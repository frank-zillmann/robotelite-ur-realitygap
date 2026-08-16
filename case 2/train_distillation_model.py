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
import pickle
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from preprocess import Identity, Preprocess, default_preprocess
from utils import (JOINT_NAMES, N_JOINTS, VEL_COL, ACC_COL, UR10e, frame_dt,
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
    """Least-squares linear baseline that predicts the actual current.

    One independent linear model per joint: ``actual_current_j = w_j .
    [target_current, qd, qdd, pos, vel, acc, gravity, mass_diag, qd_lag*]``,
    where ``target_current`` is the commanded current the joint is tracking,
    ``qd`` is the commanded velocity (from ``target_qd``), ``qdd`` its time
    derivative, ``pos`` the commanded angle, ``vel``/``acc`` the raw movej
    numbers the script commanded, ``gravity`` this joint's gravity torque and
    ``mass_diag`` its own diagonal entry of the joint-space mass matrix (both
    at the arm's full commanded pose), and ``qd_lag*`` (see ``LAGS``) this
    joint's commanded velocity delayed by a few sample counts -- the model's
    only source of memory, an FIR-style tap on the *commanded* signal (real
    ``actual_current`` history is not available for a candidate motion the
    agent is scoring, only for the recordings this fits on). This can only
    capture ring behaviour for as long as the commanded velocity's own recent
    history still carries information; once it has been flat-zero for longer
    than ``max(LAGS)``, every lagged feature reads zero too and this model has
    nothing left to reconstruct a persisting oscillation from -- a genuinely
    persistent ring needs the model to feed on its own past *predictions*
    (autoregressive), which this is not (see kianna_notes/progress_log.md
    entry 5 for what was actually observed).

    Each joint's ``w_j`` is fit separately against its own measured
    ``actual_current``, so one joint's dynamics (e.g. the shoulder, which
    fights the most gravity load) no longer leak into another's the way a
    single shared-slope fit across all joints would.

    Extension points:

    - Add Coriolis from ``utils.UR10e.coriolis`` to ``_row_features`` (the
      remaining torque term not yet used; slower to compute, scales with
      velocity products).
    - Make it genuinely autoregressive (feed the model's own past predictions
      forward as an input, simulated row by row) if the ring needs to persist
      longer than ``LAGS`` reaches.
    - Normalize the features: ``pos`` (radians) and ``vel``/``acc`` (raw movej
      numbers up to ~1000) are on very different scales.
    - Use a non-linear regressor (MLP, trees) that can capture the ring.

    Override ``_row_features`` to change the inputs, or ``predicts``/``predict``
    to model a different channel.
    """

    LAGS = (1, 2, 4, 8, 16, 32)   # samples (~8-256ms @125Hz): qd delayed this many rows
    FEATURE_NAMES = (["target_current", "qd", "qdd", "pos", "vel", "acc",
                      "gravity", "mass_diag"] + [f"qd_lag{k}" for k in LAGS])
    PHYSICS_STRIDE = 10   # rows between exact UR10e physics evals; rest interpolated

    def __init__(self):
        self.coef = None                     # (N_JOINTS, len(FEATURE_NAMES)) per-joint weights
        self.vel_range = None                # (lo, hi) commanded vel seen in training
        self.acc_range = None                # (lo, hi) commanded acc seen in training
        self.ur = UR10e()                    # gravity/mass features (see _physics_block)

    def predicts(self) -> list[str]:
        return ["actual_current"]

    # --- features -------------------------------------------------------------

    def _physics_block(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gravity torque and mass-matrix diagonal per row, each ``(n, N_JOINTS)``.

        ``UR10e`` physics is too slow (~0.5ms/row) to call on every row of a
        recording, so it is evaluated exactly every ``PHYSICS_STRIDE`` rows and
        linearly interpolated in between (always including the first and last
        row exactly) -- a trajectory's pose changes smoothly, so this is close
        to exact at a fraction of the cost. Same grid-and-interpolate idea
        ``dynamics.UR10eDynamics`` uses for the same reason. Uses
        ``gravity_and_mass_diag`` (one shared ``_link_terms`` pass) rather than
        ``gravity``+``mass_matrix`` separately, which would pay for that pass
        twice.
        """
        q = np.asarray(q, dtype=float)
        n = len(q)
        idx = np.unique(np.r_[np.arange(0, n, self.PHYSICS_STRIDE), n - 1])
        sparse = [self.ur.gravity_and_mass_diag(q[i]) for i in idx]
        g_sparse = np.array([s[0] for s in sparse])
        m_sparse = np.array([s[1] for s in sparse])
        rows = np.arange(n)
        grav = np.column_stack([np.interp(rows, idx, g_sparse[:, j]) for j in range(N_JOINTS)])
        mass_diag = np.column_stack([np.interp(rows, idx, m_sparse[:, j]) for j in range(N_JOINTS)])
        return grav, mass_diag

    @classmethod
    def _lag(cls, x: np.ndarray, k: int) -> np.ndarray:
        """``x`` delayed by ``k`` samples; the first ``k`` rows edge-pad with
        ``x[0]`` (a trajectory typically starts at rest, so repeating the
        initial value is a reasonable "nothing happened yet" fill)."""
        if k <= 0:
            return x
        return np.concatenate([np.full(k, x[0]), x[:-k]])

    def _row_features(self, joint: int, tgt_i, pos, qd, qdd, vel, acc,
                      grav, mass_diag) -> np.ndarray:
        """Feature rows for one joint over a whole trajectory, shape
        ``(n, 8 + len(LAGS))``.

        Every argument except ``joint`` is a length-``n`` array. ``tgt_i`` is
        the commanded ``target_current`` for this joint; ``grav``/``mass_diag``
        are this joint's columns of ``_physics_block`` (full-pose gravity
        torque and mass-matrix diagonal); the trailing columns are ``qd``
        delayed by each of ``LAGS`` (see the class docstring for what this
        can and can't capture). ``joint`` is otherwise unused by these base
        features but kept so an override can look up more per-joint physics
        -- each joint gets its own fit, so a joint-dependent feature no
        longer needs a one-hot to matter. Keep it a function of the commanded
        trajectory so it also applies to candidates.
        """
        tgt_i, pos = np.asarray(tgt_i), np.asarray(pos)
        qd, qdd = np.asarray(qd), np.asarray(qdd)
        vel, acc = np.asarray(vel), np.asarray(acc)
        grav, mass_diag = np.asarray(grav), np.asarray(mass_diag)
        lags = [self._lag(qd, k) for k in self.LAGS]
        return np.column_stack([tgt_i, qd, qdd, pos, vel, acc, grav, mass_diag, *lags])

    # --- fit ------------------------------------------------------------------

    def _design(self, recordings) -> list[tuple[np.ndarray, np.ndarray]]:
        """Per joint: (features, measured actual_current) stacked over every run.

        Shared by ``fit`` and the script's held-out error report, so both build
        the same per-joint feature matrices. Returns one ``(X, y)`` pair per
        joint (index 0..N_JOINTS-1), since each joint gets its own fit.
        """
        per_joint = [([], []) for _ in range(N_JOINTS)]
        for rec in recordings:
            if rec.vel_cmd is None or rec.acc_cmd is None:
                raise ValueError(f"{rec.path} has no vel/acc registers; record "
                                 "with `--float-register 1 vel 2 acc`")
            qdd = np.gradient(rec.target_qd, rec.dt, axis=0)     # commanded accel
            grav, mass_diag = self._physics_block(rec.target_q)  # once per run, not per joint
            for j in range(N_JOINTS):
                Xs, ys = per_joint[j]
                Xs.append(self._row_features(j, rec.target_current[:, j], rec.target_q[:, j],
                                             rec.target_qd[:, j], qdd[:, j],
                                             rec.vel_cmd, rec.acc_cmd,
                                             grav[:, j], mass_diag[:, j]))
                ys.append(rec.actual_current[:, j])
        return [(np.vstack(Xs), np.concatenate(ys)) for Xs, ys in per_joint]

    @staticmethod
    def _zero_near_constant(X: np.ndarray) -> np.ndarray:
        """Zero out feature columns that are (almost) constant for this joint.

        E.g. a wrist joint's own mass-matrix-diagonal entry barely changes
        with pose (see kianna_notes/progress_log.md) -- real variation is
        ~1e-6 relative, floating-point noise rather than signal. Left in,
        ``lstsq``'s minimum-norm solve fits a huge coefficient to that noise
        (unstable, and it can drag other coefficients along with it); zeroed,
        ``lstsq`` correctly assigns it coefficient 0 instead (a zero column
        cannot reduce the residual, so the minimum-norm solution ignores it).
        """
        spread = X.max(axis=0) - X.min(axis=0)
        flat = spread < 1e-6 * (np.abs(X).max(axis=0) + 1e-12)
        X = X.copy()
        X[:, flat] = 0.0
        return X

    def fit(self, recordings) -> "LinearModel":
        """Fit one row model per joint, each on every row of every real run."""
        design = self._design(recordings)
        self.coef = np.array([np.linalg.lstsq(self._zero_near_constant(X), y, rcond=None)[0]
                              for X, y in design])
        # Feature columns 4 and 5 are vel and acc, shared across joints (one
        # movej sets them for every joint at once), so joint 0's span is enough.
        X0 = design[0][0]
        self.vel_range = (float(X0[:, 4].min()), float(X0[:, 4].max()))
        self.acc_range = (float(X0[:, 5].min()), float(X0[:, 5].max()))
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
        grav, mass_diag = self._physics_block(q)
        out = np.zeros_like(q)
        for j in range(N_JOINTS):
            out[:, j] = self._row_features(j, ti[:, j], q[:, j], qd[:, j], qdd[:, j],
                                           vel, acc, grav[:, j], mass_diag[:, j]) @ self.coef[j]
        return {"actual_current": out}

    def bounds(self):
        if self.coef is None:
            return None
        return self.vel_range, self.acc_range


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

    # Build the per-joint row matrices once (same features as fit) to estimate
    # held-out error per joint: predict the measured actual_current on rows
    # each joint's own fit never saw.
    design = model._design(recordings)
    total_rows = sum(len(y) for _, y in design)
    print(f"{total_rows} rows from {len(recordings)} runs")

    # Deterministic split (no RNG): every 1/holdout-th row is a test row.
    step = max(int(round(1 / args.holdout)), 2)
    print(f"{'joint':10s} {'test rows':>10s} {'RMSE':>8s} {'R2':>8s}")
    all_err, all_y = [], []
    for j, (X, y) in enumerate(design):
        is_test = np.arange(len(y)) % step == 0
        coef, *_ = np.linalg.lstsq(LinearModel._zero_near_constant(X[~is_test]), y[~is_test], rcond=None)
        err = X[is_test] @ coef - y[is_test]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        r2 = float(1 - np.sum(err ** 2) / np.sum((y[is_test] - y[is_test].mean()) ** 2))
        print(f"{JOINT_NAMES[j]:10s} {is_test.sum():10d} {rmse:7.3f}A {r2:7.3f}")
        all_err.append(err)
        all_y.append(y[is_test])
    all_err, all_y = np.concatenate(all_err), np.concatenate(all_y)
    rmse = float(np.sqrt(np.mean(all_err ** 2)))
    r2 = float(1 - np.sum(all_err ** 2) / np.sum((all_y - all_y.mean()) ** 2))
    print(f"overall held-out ({len(all_y)} rows): actual_current RMSE {rmse:.3f} A   R2 {r2:.3f}")

    # Refit on everything and save.
    model.fit(recordings)
    print("coefficients (per joint):")
    for j in range(N_JOINTS):
        print(f"  {JOINT_NAMES[j]}:")
        for name, c in zip(LinearModel.FEATURE_NAMES, model.coef[j]):
            print(f"    {name:14s} {c:+.4f}")
    model.save(args.out)
    print(f"saved {args.out}")

if __name__ == "__main__":
    main()
