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

Run as a script to train on the recorded runs, print held-out error, and save:

    python train_distillation_model.py --csvs data/test-*.csv --out models/distill.pkl

train_rla.py and run.py depend only on the interface, so a custom subclass of
DistillModel (or LinearModel) can replace the baseline via its pickle.
"""
from __future__ import annotations

import argparse
import glob
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


def log_run(model: DistillModel, recordings, holdout_fraction: float,
            held_out_metrics: dict, results_dir: str = None, dt_str: str = None):
    """Log a training run: save log.json, residual plot, per-joint RMSE plot,
    update runs_summary.csv and regenerate the comparison plot.

    Args:
        model:             fitted DistillModel.
        recordings:        list of Recording objects used for training (all data).
        holdout_fraction:  fraction that was held out during training evaluation.
        held_out_metrics:  {channel: {"rmse": float, "r2": float, "n_rows": int}}
                           computed on the held-out split before final refit.
        results_dir:       override for RESULTS_DIR.
        dt_str:            override for the run timestamp (default: now).
    """
    results_dir = results_dir or RESULTS_DIR
    dt_str      = dt_str or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir     = os.path.join(results_dir, dt_str)
    os.makedirs(run_dir, exist_ok=True)

    eval_data    = _evaluate_model(model, recordings)
    full_metrics = _compute_metrics(eval_data)

    log = {
        "datetime":          dt_str,
        "model_class":       type(model).__name__,
        "params":            model.params(),
        "training": {
            "n_recordings":    len(recordings),
            "holdout_fraction": holdout_fraction,
        },
        "held_out_metrics":  held_out_metrics,
        "full_data_metrics": full_metrics,
    }
    log_path = os.path.join(run_dir, "log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[results] log  -> {log_path}")

    _plot_residuals(eval_data, full_metrics, model.predicts(), run_dir)
    _plot_per_joint_rmse(full_metrics, model.predicts(), run_dir)
    _update_summary(model, held_out_metrics, results_dir, dt_str)
    print(f"[results] run complete -> {run_dir}")


def main():
    ap = argparse.ArgumentParser(description="Train the distillation model.")
    ap.add_argument("--csvs", nargs="+", default=sorted(glob.glob("data/test-*.csv")),
                    help="recorded runs to train on")
    ap.add_argument("--out", default="models/distill.pkl", help="pickle path")
    ap.add_argument("--holdout", type=float, default=0.2,
                    help="fraction of rows held out for the error report")
    args = ap.parse_args()

    # Import under the real module name (not "__main__") so the saved pickle
    # loads cleanly in train_rla.py and run.py.
    from train_distillation_model import LinearModel
    from analysis import Recording

    # Preprocess the training data the same way the model will see it later.
    pre = default_preprocess()
    recordings = [Recording(r.path, df=pre.transform_distill(r.df))
                  for r in (Recording(p) for p in args.csvs)]
    model = LinearModel()

    # Build the row matrix once (same features as fit) to estimate held-out
    # error: predict the measured actual_current on held-out rows.
    X, y = model._design(recordings)
    print(f"{len(y)} rows from {len(recordings)} runs")

    # Deterministic split (no RNG): every 1/holdout-th row is a test row.
    step = max(int(round(1 / args.holdout)), 2)
    is_test = np.arange(len(y)) % step == 0
    coef, *_ = np.linalg.lstsq(X[~is_test], y[~is_test], rcond=None)
    err = X[is_test] @ coef - y[is_test]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss = float(1 - np.sum(err ** 2) / np.sum((y[is_test] - y[is_test].mean()) ** 2))
    print(f"held-out ({is_test.sum()} rows): actual_current RMSE {rmse:.3f} A   R2 {ss:.3f}")
    print("coefficients:")
    for name, c in zip(LinearModel.FEATURE_NAMES, coef):
        print(f"  {name:12s} {c:+.4f}")

    # Refit on everything and save.
    model.fit(recordings)
    model.save(args.out)
    print(f"saved {args.out}")

    # Log this run: save results/<datetime>/{log.json, plots} and update summary.
    held_out_metrics = {
        "actual_current": {"rmse": rmse, "r2": ss, "n_rows": int(is_test.sum())}
    }
    log_run(model, recordings, args.holdout, held_out_metrics)


if __name__ == "__main__":
    main()
