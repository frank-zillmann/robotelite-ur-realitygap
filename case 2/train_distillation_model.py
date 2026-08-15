"""Distilled model that predicts ``actual_*`` channels for a commanded trajectory.

Trained on recorded real runs, the model predicts what an actual channel would be
for a given commanded trajectory. Applied to the sim's targets (whose
``actual_*`` columns read 0.0), it overwrites those columns with predictions so
the metric and the RL agent can run without hardware.

``DistillModel`` is the interface the pipeline depends on:

    fit(recordings)  -> learn from recorded real runs
    predicts()       -> which actual_* channels this model fills in
    predict(df)      -> for a commanded-trajectory DataFrame, the predicted
                        actual channels, as {channel: (n, N_JOINTS) array}

The channels a model fills must be the ones the metric reads (see metrics.py).
``LinearModel`` below predicts ``actual_current``.

    from train_distillation_model import LinearModel, augment
    from analysis import Recording
    m = LinearModel().fit([Recording("data/test-4.csv"), Recording("data/test-6.csv")])
    m.predicts()                          # ['actual_current']
    m.save("models/distill.pkl")
    augment(m, "sim_to_real.csv")         # overwrite actual_current with predictions

Run as a script to fit on a fixed train set, report error on a fixed, disjoint
test set, and save. The split is fixed by default (data/test-{1,2,3,6}.csv to
train, data/test-{4,5,7}.csv to test) so numbers stay comparable across runs no
matter which DistillModel or Preprocess is plugged in:

    python train_distillation_model.py --out models/distill.pkl

Override either list to use a different split (they must not overlap):

    python train_distillation_model.py \
        --train-csvs data/test-1.csv data/test-2.csv \
        --test-csvs data/test-3.csv \
        --out models/distill.pkl

train_rla.py and run.py depend only on the interface, so a custom subclass of
DistillModel (or LinearModel) can replace the baseline via its pickle; add it
to MODELS below to select it with --model.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocess import Identity, Preprocess, default_preprocess
from utils import (JOINT_NAMES, N_JOINTS, VEL_COL, ACC_COL, frame_dt,
                   get_block, set_block)

# Fixed file-level train/test split, used regardless of --model or the active
# Preprocess. data/test-{2,4}.csv are both acc sweeps at vel=100, {3,5} are
# both vel sweeps at acc=100, and {6,7} are both wide random vel/acc combos
# (test-1 is a standalone low-range vel/acc grid) — so holding out one file
# from each pair (4, 5, 7) tests generalization to a new run within a regime
# the model has seen, rather than extrapolation to an unseen regime.
DEFAULT_TRAIN_CSVS = ["data/test-1.csv", "data/test-2.csv", "data/test-3.csv", "data/test-6.csv"]
DEFAULT_TEST_CSVS  = ["data/test-4.csv", "data/test-5.csv", "data/test-7.csv"]


class DistillModel(ABC):
    """Interface every distilled model must implement.

    Subclasses implement ``fit``, ``predicts``, and ``predict``, and optionally
    override ``bounds``.
    """

    @abstractmethod
    def fit(self, recordings) -> "DistillModel":
        """Train on a list of ``analysis.Recording`` (real robot runs)."""

    @abstractmethod
    def predicts(self) -> list[str]:
        """Per-joint channel bases this model predicts, e.g. ``["actual_current"]``.

        These are the ``actual_*`` columns ``predict`` returns and ``augment``
        overwrites in the recording.
        """

    @abstractmethod
    def predict(self, df) -> dict:
        """Predicted actual channels for a commanded-trajectory DataFrame.

        ``df`` carries ``t``, ``target_q*``, ``target_qd*`` and the commanded
        ``vel``/``acc``. Return ``{base: (n, N_JOINTS) array}`` for every base in
        ``predicts()``. The caller overwrites those columns with the result.
        """

    def bounds(self):
        """Optional ``((vel_lo, vel_hi), (acc_lo, acc_hi))`` the model trusts.

        Return the raw-number range the training data covered so the optimizer
        stays in-distribution, or ``None`` to let the caller pick its own.
        """
        return None

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "DistillModel":
        with open(path, "rb") as f:
            return pickle.load(f)

    def params(self) -> dict:
        """Return model hyperparameters for logging. Override in subclasses."""
        return {}


class LinearModel(DistillModel):
    """Least-squares linear baseline that predicts the actual current.

    One linear model per row: for each joint the actual current is
    ``w . [target_current, qd, qdd, pos, vel, acc, joint one-hot]``, where
    ``target_current`` is the commanded current the joint is tracking, ``qd`` is
    the commanded velocity (from ``target_qd``), ``qdd`` its time derivative,
    ``pos`` the commanded angle, and ``vel``/``acc`` the raw movej numbers the
    script commanded. Fit against the measured ``actual_current`` of the real
    runs.

    The one-hot joint block gives each joint its own intercept with shared
    slopes; there is no separate bias term (it would be collinear with the
    one-hot). Being linear and smooth in its inputs, it cannot reproduce the ring
    after a stop (README "Why the optimizer stalls").

    Extension points:

    - Fit per joint or add joint-interaction terms; shared slopes leak one
      joint's behaviour onto another.
    - Normalize the features: ``pos`` (radians) and ``vel``/``acc`` (raw movej
      numbers up to ~1000) are on very different scales.
    - Add physics from ``utils.UR10e`` (gravity torque, mass matrix, Coriolis).
    - Use a non-linear regressor (MLP, trees) that can capture the ring.

    Override ``_row_features`` to change the inputs, or ``predicts``/``predict``
    to model a different channel.
    """

    FEATURE_NAMES = ["target_current", "qd", "qdd", "pos", "vel", "acc"] + \
        [f"is_{n}" for n in JOINT_NAMES]

    def __init__(self):
        self.coef = None                     # (12,) per-row weights
        self.vel_range = None                # (lo, hi) commanded vel seen in training
        self.acc_range = None                # (lo, hi) commanded acc seen in training

    def predicts(self) -> list[str]:
        return ["actual_current"]

    # --- features -------------------------------------------------------------

    def _row_features(self, joint: int, tgt_i, pos, qd, qdd, vel, acc) -> np.ndarray:
        """Feature rows for one joint over a whole trajectory, shape ``(n, 12)``.

        Every argument except ``joint`` is a length-``n`` array. ``tgt_i`` is the
        commanded ``target_current`` for this joint. Override to feed the model
        more inputs (gravity torque, mass, neighbouring joints); keep it a
        function of the commanded trajectory so it also applies to candidates.
        """
        tgt_i, pos = np.asarray(tgt_i), np.asarray(pos)
        qd, qdd = np.asarray(qd), np.asarray(qdd)
        vel, acc = np.asarray(vel), np.asarray(acc)
        onehot = np.zeros((len(pos), N_JOINTS))
        onehot[:, joint] = 1.0
        return np.column_stack([tgt_i, qd, qdd, pos, vel, acc, onehot])

    # --- fit ------------------------------------------------------------------

    def _design(self, recordings) -> tuple[np.ndarray, np.ndarray]:
        """Stack (features, measured actual_current) over every joint of every run.

        Shared by ``fit`` and the script's held-out error report, so both build
        the feature matrix the same way.
        """
        X, y = [], []
        for rec in recordings:
            if rec.vel_cmd is None or rec.acc_cmd is None:
                raise ValueError(f"{rec.path} has no vel/acc registers; record "
                                 "with `--float-register 1 vel 2 acc`")
            qdd = np.gradient(rec.target_qd, rec.dt, axis=0)     # commanded accel
            for j in range(N_JOINTS):
                X.append(self._row_features(j, rec.target_current[:, j], rec.target_q[:, j],
                                            rec.target_qd[:, j], qdd[:, j],
                                            rec.vel_cmd, rec.acc_cmd))
                y.append(rec.actual_current[:, j])
        return np.vstack(X), np.concatenate(y)

    def fit(self, recordings) -> "LinearModel":
        """Fit the row model on every row of every joint of every real run."""
        X, y = self._design(recordings)
        self.coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        # Feature columns 4 and 5 are vel and acc: their spans are the bounds.
        self.vel_range = (float(X[:, 4].min()), float(X[:, 4].max()))
        self.acc_range = (float(X[:, 5].min()), float(X[:, 5].max()))
        return self

    # --- predict --------------------------------------------------------------

    def predict(self, df) -> dict:
        dt = frame_dt(df)
        ti = get_block(df, "target_current")
        q = get_block(df, "target_q")
        qd = get_block(df, "target_qd")
        qdd = np.gradient(qd, dt, axis=0)
        vel = df[VEL_COL].to_numpy(dtype=float)
        acc = df[ACC_COL].to_numpy(dtype=float)
        out = np.zeros_like(q)
        for j in range(N_JOINTS):
            out[:, j] = self._row_features(j, ti[:, j], q[:, j], qd[:, j],
                                           qdd[:, j], vel, acc) @ self.coef
        return {"actual_current": out}

    def bounds(self):
        if self.coef is None:
            return None
        return self.vel_range, self.acc_range

    def params(self) -> dict:
        return {
            "type":         "least_squares",
            "features":     self.FEATURE_NAMES,
            "coefficients": self.coef.tolist() if self.coef is not None else None,
            "vel_range":    list(self.vel_range) if self.vel_range else None,
            "acc_range":    list(self.acc_range) if self.acc_range else None,
        }


# Models selectable via --model. Add a new DistillModel subclass here to make
# it available from the CLI without touching the train/test split logic.
MODELS = {"linear": LinearModel}


def augment(model: DistillModel, csv: str, pre: Preprocess = None):
    """Overwrite a recording's actual_* columns with the model's predictions.

    Reads ``csv``, asks ``model.predict`` for the channels in
    ``model.predicts()``, writes them back into the same columns, and saves in
    place. The schema is unchanged; the actual_* columns become predictions,
    ready for metrics.py.

    ``pre`` (a preprocess.Preprocess) is applied around the model as in training:
    ``transform_distill`` before predict, ``revert_distill`` after, so the saved
    columns land back in real units. Defaults to a no-op ``Identity``.
    """
    pre = pre or Identity()
    df = pre.transform_distill(pd.read_csv(csv))
    preds = model.predict(df)
    for base in model.predicts():
        set_block(df, base, preds[base])
    df = pre.revert_distill(df)
    df.to_csv(csv, index=False)
    print(f"overwrote {model.predicts()} with predictions -> {csv}")
    return df


# ---------------------------------------------------------------------------
# Result logging and visualisation
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _evaluate_model(model: DistillModel, recordings) -> dict:
    """Run model.predict on every recording and collect per-channel per-joint arrays.

    Returns:
        {channel: {joint_idx: {"pred": array, "actual": array, "residuals": array}}}
    """
    channels = model.predicts()
    buckets = {ch: {j: {"pred": [], "actual": []} for j in range(N_JOINTS)}
               for ch in channels}
    for rec in recordings:
        preds = model.predict(rec.df)
        for ch in channels:
            actual_block = (getattr(rec, ch) if hasattr(rec, ch)
                            else get_block(rec.df, ch))
            pred_block = preds[ch]
            for j in range(N_JOINTS):
                buckets[ch][j]["pred"].append(pred_block[:, j])
                buckets[ch][j]["actual"].append(actual_block[:, j])
    result = {}
    for ch in channels:
        result[ch] = {}
        for j in range(N_JOINTS):
            pred   = np.concatenate(buckets[ch][j]["pred"])
            actual = np.concatenate(buckets[ch][j]["actual"])
            result[ch][j] = {"pred": pred, "actual": actual, "residuals": pred - actual}
    return result


def _compute_metrics(eval_data: dict) -> dict:
    """Compute overall and per-joint RMSE / R² from _evaluate_model output."""
    metrics = {}
    for ch, joints in eval_data.items():
        all_res = np.concatenate([joints[j]["residuals"] for j in range(N_JOINTS)])
        all_act = np.concatenate([joints[j]["actual"]    for j in range(N_JOINTS)])
        ss_res  = float(np.sum(all_res ** 2))
        ss_tot  = float(np.sum((all_act - all_act.mean()) ** 2))
        per_joint = []
        for j in range(N_JOINTS):
            res_j = joints[j]["residuals"]
            act_j = joints[j]["actual"]
            ss_j  = float(np.sum(res_j ** 2))
            tot_j = float(np.sum((act_j - act_j.mean()) ** 2))
            per_joint.append({
                "joint": JOINT_NAMES[j],
                "rmse":  float(np.sqrt(np.mean(res_j ** 2))),
                "r2":    float(1 - ss_j / tot_j) if tot_j else 0.0,
            })
        metrics[ch] = {
            "overall": {
                "rmse": float(np.sqrt(np.mean(all_res ** 2))),
                "r2":   float(1 - ss_res / ss_tot) if ss_tot else 0.0,
            },
            "per_joint": per_joint,
        }
    return metrics


def _plot_residuals(eval_data: dict, metrics: dict, channels: list, run_dir: str):
    for ch in channels:
        ch_short = ch.replace("actual_", "")
        all_res  = np.concatenate([eval_data[ch][j]["residuals"] for j in range(N_JOINTS)])
        ovr      = metrics[ch]["overall"]
        fig, ax  = plt.subplots(figsize=(8, 5))
        ax.hist(all_res, bins=60, edgecolor="black", alpha=0.7, color="steelblue")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="zero error")
        ax.set_xlabel(f"Residual ({ch_short})")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Residuals — {ch}\n"
            f"RMSE={ovr['rmse']:.4f}  R²={ovr['r2']:.4f}  "
            f"mean={float(np.mean(all_res)):.4f}  std={float(np.std(all_res)):.4f}"
        )
        ax.legend()
        fig.tight_layout()
        path = os.path.join(run_dir, f"residuals_{ch}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[results] plot -> {path}")


def _plot_per_joint_rmse(metrics: dict, channels: list, run_dir: str):
    for ch in channels:
        ch_short   = ch.replace("actual_", "")
        ovr        = metrics[ch]["overall"]
        joint_rmse = [metrics[ch]["per_joint"][j]["rmse"] for j in range(N_JOINTS)]
        fig, ax    = plt.subplots(figsize=(9, 5))
        bars = ax.bar(JOINT_NAMES, joint_rmse, color="steelblue", edgecolor="black")
        ax.axhline(ovr["rmse"], color="red", linestyle="--", linewidth=1.2,
                   label=f"Overall RMSE={ovr['rmse']:.4f}")
        for bar, val in zip(bars, joint_rmse):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1e-4,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("Joint")
        ax.set_ylabel(f"RMSE ({ch_short})")
        ax.set_title(f"Per-joint RMSE — {ch}")
        ax.tick_params(axis="x", rotation=20)
        ax.legend()
        fig.tight_layout()
        path = os.path.join(run_dir, f"per_joint_rmse_{ch}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[results] plot -> {path}")


def _plot_per_joint_r2(metrics: dict, channels: list, run_dir: str):
    for ch in channels:
        ch_short = ch.replace("actual_", "")
        ovr      = metrics[ch]["overall"]
        joint_r2 = [metrics[ch]["per_joint"][j]["r2"] for j in range(N_JOINTS)]
        fig, ax  = plt.subplots(figsize=(9, 5))
        bars = ax.bar(JOINT_NAMES, joint_r2, color="steelblue", edgecolor="black")
        ax.axhline(ovr["r2"], color="red", linestyle="--", linewidth=1.2,
                   label=f"Overall R²={ovr['r2']:.4f}")
        ax.axhline(0, color="black", linewidth=0.8)
        for bar, val in zip(bars, joint_r2):
            va = "bottom" if val >= 0 else "top"
            offset = 0.01 if val >= 0 else -0.01
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                    f"{val:.4f}", ha="center", va=va, fontsize=8)
        ax.set_xlabel("Joint")
        ax.set_ylabel(f"R² ({ch_short})")
        ax.set_title(f"Per-joint R² — {ch}")
        ax.tick_params(axis="x", rotation=20)
        ax.legend()
        fig.tight_layout()
        path = os.path.join(run_dir, f"per_joint_r2_{ch}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[results] plot -> {path}")


def _update_summary(model: DistillModel, held_out_metrics: dict,
                    results_dir: str, dt_str: str):
    """Append a row to runs_summary.csv and regenerate the comparison plot."""
    summary_path = os.path.join(results_dir, "runs_summary.csv")
    row = {"datetime": dt_str, "model_class": type(model).__name__}
    for ch, m in held_out_metrics.items():
        row[f"{ch}_rmse"] = m["rmse"]
        row[f"{ch}_r2"]   = m["r2"]

    summary_df = (pd.concat([pd.read_csv(summary_path), pd.DataFrame([row])],
                             ignore_index=True)
                  if os.path.exists(summary_path) else pd.DataFrame([row]))
    summary_df.to_csv(summary_path, index=False)
    print(f"[results] summary -> {summary_path}")

    # comparison plot: RMSE and R² per run for each channel
    channels  = model.predicts()
    n_ch      = len(channels)
    x         = list(range(len(summary_df)))
    labels    = summary_df["datetime"].tolist()
    fig, axes = plt.subplots(2, n_ch, figsize=(max(6, 4 * len(x)), 8 * n_ch // n_ch),
                             squeeze=False)
    for col, ch in enumerate(channels):
        for row_idx, (metric, color, ylabel) in enumerate(
            [("rmse", "steelblue", "RMSE"), ("r2", "darkorange", "R²")]
        ):
            col_name = f"{ch}_{metric}"
            ax = axes[row_idx][col]
            if col_name in summary_df.columns:
                ax.plot(x, summary_df[col_name], marker="o", color=color)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ch} — held-out {ylabel} over runs")
            ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(results_dir, "comparison_plot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[results] comparison -> {path}")


def log_run(model: DistillModel, train_csvs: list, test_csvs: list,
            train_recordings, test_recordings,
            results_dir: str = None, dt_str: str = None):
    """Log a training run: save log.json, residual plot, per-joint RMSE plot,
    update runs_summary.csv and regenerate the comparison plot.

    Metrics that matter are computed on ``test_recordings`` only — files
    ``model.fit()`` never saw. ``train_recordings`` are evaluated too and
    logged as ``in_sample_metrics`` purely as a sanity check: it should
    always look better than the held-out numbers, and if it doesn't, the fit
    itself is broken (not a generalization problem).

    Args:
        model:             fitted DistillModel (already fit on train_recordings).
        train_csvs:        paths passed to model.fit(), for the log.
        test_csvs:         held-out paths, never passed to fit(), for the log.
        train_recordings:  Recording objects for train_csvs (preprocessed).
        test_recordings:   Recording objects for test_csvs (preprocessed).
        results_dir:       override for RESULTS_DIR.
        dt_str:            override for the run timestamp (default: now).
    """
    results_dir = results_dir or RESULTS_DIR
    dt_str      = dt_str or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir     = os.path.join(results_dir, dt_str)
    os.makedirs(run_dir, exist_ok=True)

    held_out_eval    = _evaluate_model(model, test_recordings)
    held_out_metrics = _compute_metrics(held_out_eval)
    in_sample_metrics = _compute_metrics(_evaluate_model(model, train_recordings))

    log = {
        "datetime":          dt_str,
        "model_class":       type(model).__name__,
        "params":            model.params(),
        "training": {
            "train_csvs": train_csvs,
            "test_csvs":  test_csvs,
        },
        "held_out_metrics":  held_out_metrics,
        "in_sample_metrics": in_sample_metrics,
    }
    log_path = os.path.join(run_dir, "log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[results] log  -> {log_path}")

    model_path = os.path.join(run_dir, "distill.pkl")
    model.save(model_path)           # versioned copy alongside its log and plots
    print(f"[results] model -> {model_path}")

    _plot_residuals(held_out_eval, held_out_metrics, model.predicts(), run_dir)
    _plot_per_joint_rmse(held_out_metrics, model.predicts(), run_dir)
    _plot_per_joint_r2(held_out_metrics, model.predicts(), run_dir)
    summary_row = {ch: m["overall"] for ch, m in held_out_metrics.items()}
    _update_summary(model, summary_row, results_dir, dt_str)
    print(f"[results] run complete -> {run_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Fit a DistillModel on a fixed train set and report error "
                    "on a fixed, disjoint test set (see module docstring). The "
                    "split defaults are the same no matter which --model or "
                    "Preprocess (preprocess.default_preprocess) is active, so "
                    "runs stay comparable.")
    ap.add_argument("--train-csvs", nargs="+", default=DEFAULT_TRAIN_CSVS,
                    help="recordings model.fit() sees (default: %(default)s)")
    ap.add_argument("--test-csvs", nargs="+", default=DEFAULT_TEST_CSVS,
                    help="held-out recordings, never passed to fit(); the "
                        "printed/plotted/logged metrics come from these "
                        "(default: %(default)s)")
    ap.add_argument("--model", choices=sorted(MODELS), default="linear",
                    help="DistillModel to train (default: %(default)s)")
    ap.add_argument("--out", default="models/distill.pkl", help="pickle path")
    args = ap.parse_args()

    overlap = set(args.train_csvs) & set(args.test_csvs)
    if overlap:
        raise SystemExit(f"--train-csvs and --test-csvs overlap, the test set "
                          f"would not be held out: {sorted(overlap)}")

    # Import under the real module name (not "__main__") so the saved pickle
    # loads cleanly in train_rla.py and run.py.
    from train_distillation_model import MODELS as _MODELS
    from analysis import Recording

    # Preprocess train and test the same way the model will see them later.
    pre = default_preprocess()
    train_recordings = [Recording(r.path, df=pre.transform_distill(r.df))
                        for r in (Recording(p) for p in args.train_csvs)]
    test_recordings  = [Recording(r.path, df=pre.transform_distill(r.df))
                        for r in (Recording(p) for p in args.test_csvs)]

    model = _MODELS[args.model]()
    model.fit(train_recordings)
    print(f"trained {type(model).__name__} on {len(train_recordings)} run(s): "
          f"{[os.path.basename(p) for p in args.train_csvs]}")

    held_out_metrics = _compute_metrics(_evaluate_model(model, test_recordings))
    print(f"held-out on {len(test_recordings)} run(s) never seen by fit(): "
          f"{[os.path.basename(p) for p in args.test_csvs]}")
    for ch, m in held_out_metrics.items():
        ovr = m["overall"]
        print(f"  {ch}: RMSE={ovr['rmse']:.4f}  R2={ovr['r2']:.4f}")
        for pj in m["per_joint"]:
            print(f"    {pj['joint']:10s} RMSE={pj['rmse']:.4f}  R2={pj['r2']:.4f}")

    model.save(args.out)             # models/distill.pkl — "latest" for pipeline defaults
    print(f"saved {args.out}")

    # Log this run: save results/<datetime>/{log.json, model, plots} and update summary.
    dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_run(model, args.train_csvs, args.test_csvs, train_recordings, test_recordings,
            dt_str=dt_str)


if __name__ == "__main__":
    main()
