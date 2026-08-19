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
import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime

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


class LinearModel(DistillModel):
    """Least-squares linear baseline that predicts the actual position.

    One linear model per row: for each joint the actual position (rad) is
    ``w . [target_current, qd, qdd, pos, vel, acc, joint one-hot]``, where
    ``target_current`` is the commanded current the joint is tracking, ``qd`` is
    the commanded velocity (from ``target_qd``), ``qdd`` its time derivative,
    ``pos`` the commanded angle, and ``vel``/``acc`` the raw movej numbers the
    script commanded. Fit against the measured ``actual_q`` of the real runs.

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
        return ["actual_q"]

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
        """Stack (features, measured actual_q) over every joint of every run.

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
                y.append(rec.actual_q[:, j])
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
        return {"actual_q": out}

    def bounds(self):
        if self.coef is None:
            return None
        return self.vel_range, self.acc_range


def _annotate_bars(ax, bars, values, fmt="{:.3f}"):
    """Print each bar's value above (or below, if negative) the bar itself."""
    for bar, v in zip(bars, values):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y, fmt.format(v),
                ha="center", va="bottom" if y >= 0 else "top", fontsize=9)
    ax.margins(y=0.15)


def plot_per_joint_metrics(rmse: np.ndarray, r2: np.ndarray):
    """RMSE and R2 of held-out ``actual_q`` predictions, one bar per joint.

    ``rmse`` is taken in radians (as computed in ``main()``) and shown in
    degrees here -- easier to read than radians, and R2 is already unitless
    so it needs no conversion.
    """
    import matplotlib.pyplot as plt
    import ur_style
    ur_style.apply()

    rmse_deg = np.degrees(rmse)
    fig, (ax_rmse, ax_r2) = plt.subplots(1, 2, figsize=(12, 5))

    bars = ax_rmse.bar(JOINT_NAMES, rmse_deg, color=ur_style.BLUE)
    _annotate_bars(ax_rmse, bars, rmse_deg, "{:.3f}°")
    ax_rmse.set_ylabel("RMSE (deg)")
    ax_rmse.set_title("Held-out actual_q RMSE per joint")

    bars = ax_r2.bar(JOINT_NAMES, r2, color=ur_style.MID_BLUE)
    _annotate_bars(ax_r2, bars, r2, "{:.3f}")
    ax_r2.set_ylabel(r"$R^2$")
    ax_r2.set_title(r"Held-out actual_q $R^2$ per joint")

    for ax in (ax_rmse, ax_r2):
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


def plot_coefficients(model: "LinearModel"):
    """Bar chart of the fitted LinearModel's per-feature weights.

    Diverging two-tone: BLUE for positive weights, NAVY for negative, so sign
    reads at a glance without leaving the brand palette.
    """
    import matplotlib.pyplot as plt
    import ur_style
    ur_style.apply()

    coef = np.asarray(model.coef)
    colors = [ur_style.BLUE if c >= 0 else ur_style.NAVY for c in coef]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(LinearModel.FEATURE_NAMES, coef, color=colors)
    _annotate_bars(ax, bars, coef, "{:+.3f}")
    ax.axhline(0, color=ur_style.GRAY, lw=0.8)
    ax.set_ylabel("weight")
    ax.set_title("LinearModel coefficients (actual_q)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


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


def main():
    ap = argparse.ArgumentParser(description="Train the distillation model.")
    ap.add_argument("--csvs", nargs="+", default=sorted(glob.glob("data/test-*.csv")),
                    help="recorded runs to train on")
    ap.add_argument("--out", default="models/distill.pkl", help="pickle path")
    ap.add_argument("--holdout", type=float, default=0.2,
                    help="fraction of rows held out for the error report")
    ap.add_argument("--no-plot", action="store_true",
                    help="skip the RMSE/R2/coefficient plots entirely")
    ap.add_argument("--no-show", action="store_true",
                    help="save plots to results/<timestamp>/ but don't open interactive windows")
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
    # error: predict the measured actual_q on held-out rows.
    X, y = model._design(recordings)
    print(f"{len(y)} rows from {len(recordings)} runs")

    # Deterministic split (no RNG): every 1/holdout-th row is a test row.
    step = max(int(round(1 / args.holdout)), 2)
    is_test = np.arange(len(y)) % step == 0
    coef, *_ = np.linalg.lstsq(X[~is_test], y[~is_test], rcond=None)
    err = X[is_test] @ coef - y[is_test]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss = float(1 - np.sum(err ** 2) / np.sum((y[is_test] - y[is_test].mean()) ** 2))
    print(f"held-out ({is_test.sum()} rows): actual_q RMSE {rmse:.4f} rad   R2 {ss:.3f}")
    print("coefficients:")
    for name, c in zip(LinearModel.FEATURE_NAMES, coef):
        print(f"  {name:12s} {c:+.4f}")

    # Per-joint held-out RMSE/R2: the last N_JOINTS columns of X are the joint
    # one-hot, so each row's joint is just its argmax there.
    joint_idx = X[:, -N_JOINTS:].argmax(axis=1)
    rmse_per_joint = np.zeros(N_JOINTS)
    r2_per_joint = np.zeros(N_JOINTS)
    print(f"{'joint':10s} {'RMSE':>8s} {'R2':>8s}")
    for j in range(N_JOINTS):
        mask = is_test & (joint_idx == j)
        err_j = X[mask] @ coef - y[mask]
        y_j = y[mask]
        rmse_per_joint[j] = float(np.sqrt(np.mean(err_j ** 2)))
        r2_per_joint[j] = float(1 - np.sum(err_j ** 2) / np.sum((y_j - y_j.mean()) ** 2))
        print(f"{JOINT_NAMES[j]:10s} {rmse_per_joint[j]:8.4f}rad {r2_per_joint[j]:8.3f}")

    # Refit on everything and save.
    model.fit(recordings)
    model.save(args.out)
    print(f"saved {args.out}")

    if not args.no_plot:
        import matplotlib.pyplot as plt

        out_dir = os.path.join("results", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(out_dir, exist_ok=True)

        fig_metrics = plot_per_joint_metrics(rmse_per_joint, r2_per_joint)
        fig_metrics.savefig(os.path.join(out_dir, "per_joint_metrics.png"),
                            dpi=150, bbox_inches="tight")

        fig_coef = plot_coefficients(model)
        fig_coef.savefig(os.path.join(out_dir, "coefficients.png"),
                         dpi=150, bbox_inches="tight")

        print(f"saved plots -> {out_dir}")

        if not args.no_show:
            plt.show()


if __name__ == "__main__":
    main()
