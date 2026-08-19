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
``PerJointPositionModel`` below predicts ``actual_q`` (position), one
independent linear fit per joint -- see its docstring for why per-joint rather
than a shared fit across joints.

    from train_distillation_model import PerJointPositionModel, augment
    from analysis import Recording
    m = PerJointPositionModel().fit([Recording("data/test-4.csv"), Recording("data/test-6.csv")])
    m.predicts()                          # ['actual_q']
    m.save("models/distill.pkl")
    augment(m, "sim_to_real.csv")         # overwrite actual_q with predictions

Run as a script to fit on every recorded run, report a row-level held-out
error, then refit on 100% of the rows and save that (see ``DEFAULT_CSVS``/
``DEFAULT_HOLDOUT`` below for the split methodology and its caveat):

    python train_distillation_model.py --out models/distill.pkl

Override the csvs or the holdout fraction:

    python train_distillation_model.py \
        --csvs data/test-1.csv data/test-2.csv data/test-3.csv \
        --holdout 0.1 --out models/distill.pkl

train_rla.py and run.py depend only on the interface, so a custom subclass of
DistillModel (or PerJointPositionModel) can replace the baseline via its
pickle; add it to MODELS below to select it with --model. train_rla.py and
run.py still default to ``metrics.CurrentGapMetric``/``actual_current``, which
this module no longer predicts -- they need to switch to a position metric
(``metrics.PositionGapMetric``) before they'll work with a model saved from
here again.
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
from sklearn.ensemble import HistGradientBoostingRegressor

import ur_style
from preprocess import Identity, Preprocess, default_preprocess
from utils import (JOINT_NAMES, N_JOINTS, VEL_COL, ACC_COL, UR10e, frame_dt,
                   get_block, set_block)

# Shared physics instance for the gravity-torque feature (PerJointPositionModel).
# payload=0.0 (default): recordings don't log a per-run payload to plug in here.
_UR10E = UR10e()

# Fixed seed for PerJointTreeModel's HistGradientBoostingRegressor: its
# auto-triggered early stopping carves out its own internal validation split,
# which is non-deterministic without a fixed seed -- would otherwise break the
# "no RNG" determinism the row-level split (below) is built around.
_RANDOM_STATE = 0


def _gravity_block(q: np.ndarray) -> np.ndarray:
    """Gravity torque per joint, ``(n, N_JOINTS)`` Nm, for a whole trajectory.

    ``UR10e.gravity_batch`` takes the full 6-joint pose for every row at once
    -- the torque on any one joint depends on the pose of the whole chain,
    not just that joint's own angle. Computed once per recording here, not
    once per joint inside ``_design``'s/``predict``'s per-joint loop, since a
    single call already returns all six joints' torques (looping per joint
    would redo the same FK/Jacobian work six times over). Uses the batched
    (numpy-vectorized) method, not ``gravity`` in a Python loop -- the latter
    is bit-identical but ~150x slower over the ~1e6 rows a full training run
    covers (verified: 7.5s vs. several minutes for 1.08M rows).
    """
    return _UR10E.gravity_batch(q)

# Train on every recorded run; holdout is a row-level fraction, not a set of
# held-out files. Matches original_train.py's methodology: pool every row of
# every joint of every recording, mark a deterministic fraction of rows as
# "test" (no RNG), fit on the rest, then refit on 100% of the rows before
# saving -- the held-out numbers are a diagnostic on a throwaway fit, not a
# property of the saved model.
#
# Caveat worth knowing: because held-out rows sit inside the same continuous
# trajectories the fit sees (the robot streams at ~128 Hz, so row i and row
# i+1 are ~7.8 ms apart and highly autocorrelated), this measures
# interpolation within seen trajectories more than generalization to an
# unseen run. A file-level split (hold out whole recordings) is the stricter
# alternative -- see git history for the version of this file that did that.
DEFAULT_CSVS = [f"data/test-{i}.csv" for i in range(1, 8)]
DEFAULT_HOLDOUT = 0.2


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
        """Per-joint channel bases this model predicts, e.g. ``["actual_q"]``.

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

    def coefficients(self):
        """Optional ``{joint_name: (feature_names, weights)}`` for a linear
        model's coefficients bar chart.

        Return ``None`` (the default) for a model with no per-feature linear
        weight to show -- ``_plot_coefficients`` skips the plot rather than
        erroring, so a non-linear regressor just doesn't get one.
        """
        return None


class PerJointPositionModel(DistillModel):
    """Least-squares linear baseline that predicts actual position, one independent
    fit per joint.

    One linear model per joint rather than one shared-slope fit pooled across
    all six (a joint one-hot only shifting the intercept, the shape an earlier
    version of this baseline used): a held-out comparison on the current
    channel showed pooling costs the low-current wrist joints a lot of accuracy
    (R² 0.43 -> 0.60 for wrist2, 0.07 -> 0.26 for wrist3 when fit separately)
    because their true sensitivity to the inputs differs in scale and even sign
    from the big joints that dominate a pooled fit. So here each joint gets its
    own coefficient vector from its own ``lstsq`` call — six independent small
    regressions rather than one big one with a joint indicator.

    Predicts the *residual* ``actual_q - target_q`` rather than raw ``actual_q``:
    the target is then near-zero-mean and on a consistent scale across joints
    (radians of error, not radians of arbitrary joint angle), and ``predict``
    just adds ``target_q`` back before returning. Each joint's fit also gets a
    real intercept (no one-hot to be collinear with, since each joint is its own
    fit), so a joint's average static bias -- e.g. gravity sag at its typical
    poses -- has somewhere to go instead of leaking into the other slopes.

    Features: ``target_current``, ``qd``, ``qdd``, ``pos``, ``gravity_torque``,
    ``vel``, ``acc``, plus the intercept. ``gravity_torque`` is
    ``utils.UR10e.gravity(target_q)[joint]`` -- the torque needed to hold the
    whole arm against gravity at that instant's commanded pose, Nm. It
    depends on all six joints' angles at once (the torque felt at any one
    joint depends on the pose of the whole chain, not just that joint's own
    angle), which is why it's computed from the full pose and only then
    sliced to this joint's column, rather than being a function of ``pos``
    alone. Justification: direction of travel alone (a proxy for "moving
    with or against gravity") swung measured overshoot 3.4x on the same
    joint at the same speed, bigger than anything vel/acc explained
    (`bronze_tier/segment_stats.csv`); gravity torque is the physical
    quantity behind that effect, and generalizes to poses the sweeps didn't
    cover in a way a direction flag can't.

    Being linear and per-row, it still can't reproduce the settle-window
    ring (a decaying oscillation after the commanded motion stops -- see
    ``bronze_tier/trajectories/worst_move_test-4_shoulder.png``); gravity
    torque is a *static*, pose-dependent term and has nothing to say about
    that dynamic, history-dependent effect.

    Extension points:

    - Add lag features (recent ``qd``/``qdd``/jerk) or a decaying-oscillation
      term to capture the ring -- a linear-in-recent-history model still fits
      with ``lstsq``, a nonlinear one needs a different regressor.
    - Add ``utils.UR10e.mass_matrix(q)`` (effective inertia at the pose) --
      the other direct physical driver of deflection not yet included.
    - Override ``_row_features`` (kept with a ``(joint, ...)`` signature even
      though this baseline ignores ``joint``) to feed a joint-specific physics
      term, e.g. a different neighbouring-joint coupling term per joint.
    - Subclass and override ``FEATURE_NAMES`` to use a different subset of
      the columns ``_row_features`` knows how to build (see
      ``PerJointTreeModel``) -- selection is driven entirely by
      ``self.FEATURE_NAMES``, not a hard-coded column list, specifically so
      this is possible without duplicating feature-building logic.
    """

    FEATURE_NAMES = ["target_current", "qd", "qdd", "pos", "gravity_torque",
                     "vel", "acc", "bias"]

    def __init__(self):
        self.coefs = None                    # (N_JOINTS, len(FEATURE_NAMES)) one row per joint
        self.vel_range = None
        self.acc_range = None

    def predicts(self) -> list[str]:
        return ["actual_q"]

    # --- features -------------------------------------------------------------

    def _row_features(self, joint: int, tgt_i, pos, qd, qdd, vel, acc, gravity=None) -> np.ndarray:
        """Feature row for one joint over a whole trajectory, columns selected
        by ``self.FEATURE_NAMES``, shape ``(n, len(FEATURE_NAMES))``.

        ``joint`` is unused by this baseline (each joint already gets its own
        coefficients from being fit separately) but kept in the signature so a
        subclass can look up a joint-specific physics term without changing the
        call sites. ``gravity`` is this joint's own column of
        ``_gravity_block(full 6-joint pose)`` -- already sliced to one joint
        by the caller, since computing it needs the whole pose, not just this
        joint's ``pos`` -- and is only required if ``"gravity_torque"`` is in
        ``self.FEATURE_NAMES``; a subclass that omits it (e.g.
        ``PerJointTreeModel``) can leave it ``None``.
        """
        tgt_i, pos = np.asarray(tgt_i), np.asarray(pos)
        qd, qdd = np.asarray(qd), np.asarray(qdd)
        vel, acc = np.asarray(vel), np.asarray(acc)
        bias = np.ones(len(pos))
        cols = {"target_current": tgt_i, "qd": qd, "qdd": qdd, "pos": pos,
                "vel": vel, "acc": acc, "bias": bias}
        if gravity is not None:
            cols["gravity_torque"] = np.asarray(gravity)
        return np.column_stack([cols[name] for name in self.FEATURE_NAMES])

    # --- fit ------------------------------------------------------------------

    def _design(self, recordings) -> dict:
        """Per-joint (features, position-error target) design matrices.

        Returns ``{joint: (X, y)}``, ``y`` being ``actual_q - target_q`` (the
        residual this model fits). Shared by ``fit`` so joint and pooled
        evaluation build the feature matrix the same way. Only computes
        ``_gravity_block`` (the expensive full-pose FK pass) when
        ``"gravity_torque"`` is actually in ``self.FEATURE_NAMES``, so a
        subclass that omits it doesn't pay for it.
        """
        needs_gravity = "gravity_torque" in self.FEATURE_NAMES
        per_joint = {j: ([], []) for j in range(N_JOINTS)}
        for rec in recordings:
            if rec.vel_cmd is None or rec.acc_cmd is None:
                raise ValueError(f"{rec.path} has no vel/acc registers; record "
                                 "with `--float-register 1 vel 2 acc`")
            qdd = np.gradient(rec.target_qd, rec.dt, axis=0)     # commanded accel
            grav = _gravity_block(rec.target_q) if needs_gravity else None
            for j in range(N_JOINTS):
                Xs, ys = per_joint[j]
                Xs.append(self._row_features(j, rec.target_current[:, j], rec.target_q[:, j],
                                             rec.target_qd[:, j], qdd[:, j],
                                             rec.vel_cmd, rec.acc_cmd,
                                             grav[:, j] if needs_gravity else None))
                ys.append(rec.actual_q[:, j] - rec.target_q[:, j])
        return {j: (np.vstack(Xs), np.concatenate(ys)) for j, (Xs, ys) in per_joint.items()}

    def fit(self, recordings) -> "PerJointPositionModel":
        """Fit one row model per joint, independently, on every row of every real run."""
        design = self._design(recordings)
        self.coefs = np.zeros((N_JOINTS, len(self.FEATURE_NAMES)))
        vel_col = self.FEATURE_NAMES.index("vel")
        acc_col = self.FEATURE_NAMES.index("acc")
        vel_all, acc_all = [], []
        for j, (X, y) in design.items():
            self.coefs[j], *_ = np.linalg.lstsq(X, y, rcond=None)
            vel_all.append(X[:, vel_col])
            acc_all.append(X[:, acc_col])
        self.vel_range = (float(np.min(vel_all)), float(np.max(vel_all)))
        self.acc_range = (float(np.min(acc_all)), float(np.max(acc_all)))
        return self

    def _row_split_fit_predict(self, X_train, y_train, X_test) -> np.ndarray:
        """Fit a throwaway model on ``(X_train, y_train)``, predict ``X_test``.

        Returns the *residual* prediction (not the absolute value ``predict``
        returns) -- the caller (``_row_split_eval``) adds ``target_q`` back
        itself. Used only for the row-split held-out diagnostic, discarded
        after; override in a subclass whose real ``fit`` isn't linear so the
        held-out number reflects that model's actual method rather than a
        hard-coded ``lstsq`` refit on its features.
        """
        coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
        return X_test @ coef

    # --- predict --------------------------------------------------------------

    def predict(self, df) -> dict:
        dt = frame_dt(df)
        ti = get_block(df, "target_current")
        q = get_block(df, "target_q")
        qd = get_block(df, "target_qd")
        qdd = np.gradient(qd, dt, axis=0)
        vel = df[VEL_COL].to_numpy(dtype=float)
        acc = df[ACC_COL].to_numpy(dtype=float)
        needs_gravity = "gravity_torque" in self.FEATURE_NAMES
        grav = _gravity_block(q) if needs_gravity else None
        out = np.zeros_like(q)
        for j in range(N_JOINTS):
            feats = self._row_features(j, ti[:, j], q[:, j], qd[:, j], qdd[:, j], vel, acc,
                                       grav[:, j] if needs_gravity else None)
            out[:, j] = q[:, j] + feats @ self.coefs[j]     # target_q + predicted error
        return {"actual_q": out}

    def bounds(self):
        if self.vel_range is None:
            return None
        return self.vel_range, self.acc_range

    def params(self) -> dict:
        return {
            "type":         "least_squares_per_joint",
            "features":     self.FEATURE_NAMES,
            "coefficients": self.coefs.tolist() if self.coefs is not None else None,
            "vel_range":    list(self.vel_range) if self.vel_range else None,
            "acc_range":    list(self.acc_range) if self.acc_range else None,
        }

    def coefficients(self):
        if self.coefs is None:
            return None
        return {JOINT_NAMES[j]: (self.FEATURE_NAMES, self.coefs[j]) for j in range(N_JOINTS)}


class PerJointTreeModel(PerJointPositionModel):
    """Gradient-boosted-tree position model, one independent regressor per joint.

    Same target convention as ``PerJointPositionModel``: predicts the
    *residual* ``actual_q - target_q``, not raw ``actual_q``. This matters
    even more for a tree than for the linear parent: a tree approximates a
    function as piecewise-constant regions, so making it reproduce
    "output ≈ target_q" (a continuous, ~±π-range near-identity mapping)
    before it has any capacity left for the actual millirad-scale correction
    would waste most of its splits on the trivial part. Predicting the small,
    near-zero-mean residual instead means every split is doing useful work.

    Deliberately uses the pre-gravity 7-feature set (``target_current``,
    ``qd``, ``qdd``, ``pos``, ``vel``, ``acc``, ``bias``) rather than the
    parent's 8 (which adds ``gravity_torque``) -- omitting it here was a
    scope decision, not an evidence-driven one; see ``ModelReview.md``.
    Reuses the parent's ``_row_features``/``_design``/``predicts``/``bounds``
    unchanged (both are driven entirely by ``self.FEATURE_NAMES``, see the
    parent's docstring) -- only the fitting/prediction mechanism changes.

    No feature scaling needed: ``HistGradientBoostingRegressor``'s splits are
    threshold-based on one feature at a time and invariant to monotonic
    per-column transforms, unlike a true gradient-descent-trained model
    (e.g. an MLP) that this project hasn't used and doesn't need here.

    ``coefficients()`` returns ``None`` -- there's no per-feature linear
    weight to show, so ``_plot_coefficients`` skips the plot (prints, does
    not error) rather than displaying something meaningless.
    """

    FEATURE_NAMES = ["target_current", "qd", "qdd", "pos", "vel", "acc", "bias"]

    def __init__(self):
        super().__init__()
        self.models = [None] * N_JOINTS      # one fitted HistGradientBoostingRegressor per joint

    def fit(self, recordings) -> "PerJointTreeModel":
        """Fit one gradient-boosted-tree model per joint, independently."""
        design = self._design(recordings)
        vel_col = self.FEATURE_NAMES.index("vel")
        acc_col = self.FEATURE_NAMES.index("acc")
        vel_all, acc_all = [], []
        for j, (X, y) in design.items():
            self.models[j] = HistGradientBoostingRegressor(
                random_state=_RANDOM_STATE).fit(X, y)
            vel_all.append(X[:, vel_col])
            acc_all.append(X[:, acc_col])
        self.vel_range = (float(np.min(vel_all)), float(np.max(vel_all)))
        self.acc_range = (float(np.min(acc_all)), float(np.max(acc_all)))
        return self

    def _row_split_fit_predict(self, X_train, y_train, X_test) -> np.ndarray:
        return HistGradientBoostingRegressor(
            random_state=_RANDOM_STATE).fit(X_train, y_train).predict(X_test)

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
            feats = self._row_features(j, ti[:, j], q[:, j], qd[:, j], qdd[:, j], vel, acc)
            out[:, j] = q[:, j] + self.models[j].predict(feats)
        return {"actual_q": out}

    def params(self) -> dict:
        return {
            "type":         "hist_gradient_boosting_per_joint",
            "features":     self.FEATURE_NAMES,
            "random_state": _RANDOM_STATE,
            "vel_range":    list(self.vel_range) if self.vel_range else None,
            "acc_range":    list(self.acc_range) if self.acc_range else None,
        }

    def coefficients(self):
        return None


# Models selectable via --model. Add a new DistillModel subclass here to make
# it available from the CLI without touching the train/test split logic.
MODELS = {"linear_per_joint": PerJointPositionModel, "tree_per_joint": PerJointTreeModel}


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

    Also gathers each channel's ``target_*`` counterpart (e.g. ``target_q`` for
    ``actual_q``) alongside ``actual``, so ``_compute_metrics`` can score R²
    against "predict no gap" rather than "predict the mean" -- see its
    docstring for why that distinction matters for a channel like position.

    Returns:
        {channel: {joint_idx: {"pred": array, "actual": array, "target": array,
                                "residuals": array}}}
    """
    channels = model.predicts()
    buckets = {ch: {j: {"pred": [], "actual": [], "target": []} for j in range(N_JOINTS)}
               for ch in channels}
    for rec in recordings:
        preds = model.predict(rec.df)
        for ch in channels:
            actual_block = (getattr(rec, ch) if hasattr(rec, ch)
                            else get_block(rec.df, ch))
            tgt_ch = ch.replace("actual_", "target_")
            target_block = (getattr(rec, tgt_ch) if hasattr(rec, tgt_ch)
                            else get_block(rec.df, tgt_ch))
            pred_block = preds[ch]
            for j in range(N_JOINTS):
                buckets[ch][j]["pred"].append(pred_block[:, j])
                buckets[ch][j]["actual"].append(actual_block[:, j])
                buckets[ch][j]["target"].append(target_block[:, j])
    result = {}
    for ch in channels:
        result[ch] = {}
        for j in range(N_JOINTS):
            pred   = np.concatenate(buckets[ch][j]["pred"])
            actual = np.concatenate(buckets[ch][j]["actual"])
            target = np.concatenate(buckets[ch][j]["target"])
            result[ch][j] = {"pred": pred, "actual": actual, "target": target,
                             "residuals": pred - actual}
    return result


def _compute_metrics(eval_data: dict) -> dict:
    """Compute overall and per-joint RMSE / R² from _evaluate_model output.

    R² here is 1 - ss_res/ss_tot with ss_tot measured against **target**, i.e.
    against the trivial "predict no gap" (``actual = target``) baseline, not
    the usual "predict the mean of actual" baseline. For a channel like
    ``actual_q``, the mean-of-actual baseline is nearly worthless: position
    spans radians over a move while the gap being modeled is millirad-scale,
    so "predict the mean" and "predict target" score almost identically close
    to 0 residual either way and R² saturates near 1.0 regardless of whether
    the model learned anything (verified: a model that just copies target into
    actual_q scores the same R²=1.0000 the fitted model did). Scoring against
    target instead measures the thing that's actually being modeled: how much
    of the *tracking gap* the model explains, relative to assuming there is
    none. This also applies more mildly to ``actual_current``, where the gap
    is already a larger share of the signal's variance so the two definitions
    were closer to agreeing -- but "predict no gap" is the more meaningful
    null model for a distillation task either way.
    """
    metrics = {}
    for ch, joints in eval_data.items():
        all_res = np.concatenate([joints[j]["residuals"] for j in range(N_JOINTS)])
        all_act = np.concatenate([joints[j]["actual"]    for j in range(N_JOINTS)])
        all_tgt = np.concatenate([joints[j]["target"]    for j in range(N_JOINTS)])
        ss_res  = float(np.sum(all_res ** 2))
        ss_tot  = float(np.sum((all_act - all_tgt) ** 2))
        per_joint = []
        for j in range(N_JOINTS):
            res_j = joints[j]["residuals"]
            act_j = joints[j]["actual"]
            tgt_j = joints[j]["target"]
            ss_j  = float(np.sum(res_j ** 2))
            tot_j = float(np.sum((act_j - tgt_j) ** 2))
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


# Channels whose native unit is radians; plots convert these to degrees for
# readability (a fraction of a radian is hard to eyeball). log.json and
# runs_summary.csv keep radians -- SI, and what metrics.py/train_rla.py consume.
_ANGLE_CHANNELS = {"actual_q"}


def _plot_units(ch: str) -> tuple[float, str]:
    """(scale, unit label) to convert a channel's native units for plotting."""
    if ch in _ANGLE_CHANNELS:
        return 180.0 / np.pi, "deg"
    return 1.0, ch.replace("actual_", "")


def _annotate_bars(ax, bars, values, fmt: str = "{:.4f}") -> None:
    """Value labels above (below, for negative bars) each bar.

    Shared by every per-joint bar chart in this file so the annotation offset
    and font stay identical; the offset scales with the data's own range so it
    reads right whether the chart is R² (~[-1, 1]) or an RMSE in degrees.
    """
    span = max((abs(v) for v in values), default=1.0) or 1.0
    for bar, val in zip(bars, values):
        va     = "bottom" if val >= 0 else "top"
        offset = 0.015 * span * (1 if val >= 0 else -1)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                fmt.format(val), ha="center", va=va, fontsize=8)


def _plot_residuals(eval_data: dict, metrics: dict, channels: list, run_dir: str):
    """General overview: one pooled histogram plus every joint's residual
    distribution overlaid on top of it.

    The pooled histogram (grey, behind) is the "how's the model doing overall"
    number; the overlaid per-joint outlines (density-normalized, so a joint
    with fewer rows isn't dwarfed) let you spot at a glance whether the overall
    number is being carried by one joint or is representative of all six. See
    ``_plot_residuals_per_joint`` for each joint's own x-scale, since a wrist
    joint's spread can be too small to read here next to the shoulder's.
    """
    ur_style.apply()
    for ch in channels:
        scale, unit = _plot_units(ch)
        all_res  = np.concatenate([eval_data[ch][j]["residuals"] for j in range(N_JOINTS)]) * scale
        ovr      = metrics[ch]["overall"]
        fig, ax  = plt.subplots(figsize=(9, 5.5))
        ax.hist(all_res, bins=60, color=ur_style.GRID, edgecolor=ur_style.GRAY,
                label="all joints (pooled)")
        ax2 = ax.twinx()
        for j in range(N_JOINTS):
            res_j  = eval_data[ch][j]["residuals"] * scale
            rmse_j = metrics[ch]["per_joint"][j]["rmse"] * scale
            ax2.hist(res_j, bins=60, histtype="step", density=True, linewidth=1.4,
                     label=f"{JOINT_NAMES[j]} (RMSE={rmse_j:.4f})")
        ax2.set_yticks([])
        ax.axvline(0, color=ur_style.GRAY, linestyle="--", linewidth=1.2, label="zero error")
        ax.set_xlabel(f"Residual ({unit})")
        ax.set_ylabel("Count (pooled)")
        ax.set_title(
            f"Residuals — {ch}\n"
            f"Overall RMSE={ovr['rmse'] * scale:.4f}  R²={ovr['r2']:.4f}  "
            f"mean={float(np.mean(all_res)):.4f}  std={float(np.std(all_res)):.4f}"
        )
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
        fig.tight_layout()
        path = os.path.join(run_dir, f"residuals_{ch}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[results] plot -> {path}")


def _plot_residuals_per_joint(eval_data: dict, metrics: dict, channels: list, run_dir: str):
    """Per-joint detail: one histogram per joint, each on its own x-scale.

    Complements ``_plot_residuals``'s pooled/overlaid view -- a joint whose
    residuals are much smaller than the others (e.g. a wrist next to the
    shoulder) is illegible there but has full resolution here, including
    whether its distribution is centered on zero or biased.
    """
    ur_style.apply()
    for ch in channels:
        scale, unit = _plot_units(ch)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for j in range(N_JOINTS):
            ax = axes[j // 3][j % 3]
            res_j = eval_data[ch][j]["residuals"] * scale
            pj = metrics[ch]["per_joint"][j]
            ax.hist(res_j, bins=40, edgecolor=ur_style.NAVY, alpha=0.85, color=ur_style.BLUE)
            ax.axvline(0, color=ur_style.GRAY, linestyle="--", linewidth=1.0)
            mean_j, std_j = float(np.mean(res_j)), float(np.std(res_j))
            ax.set_title(
                f"{JOINT_NAMES[j]}\nRMSE={pj['rmse'] * scale:.4f}  R²={pj['r2']:.4f}  "
                f"mean={mean_j:.4f}  std={std_j:.4f}", fontsize=9)
            ax.set_xlabel(f"Residual ({unit})", fontsize=8)
        fig.suptitle(f"Residuals by joint — {ch}")
        fig.tight_layout()
        path = os.path.join(run_dir, f"residuals_per_joint_{ch}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[results] plot -> {path}")


def _plot_per_joint_metrics(metrics: dict, channels: list, run_dir: str):
    """Held-out RMSE and R² per joint, side by side in one figure.

    One combined figure per channel (RMSE left, R² right) rather than two
    separate files -- matches the comparison plots from the earlier branch's
    baseline, so a run from here drops next to one of those for an easy
    side-by-side look.
    """
    ur_style.apply()
    for ch in channels:
        scale, unit = _plot_units(ch)
        ovr        = metrics[ch]["overall"]
        joint_rmse = [metrics[ch]["per_joint"][j]["rmse"] * scale for j in range(N_JOINTS)]
        joint_r2   = [metrics[ch]["per_joint"][j]["r2"] for j in range(N_JOINTS)]

        fig, (ax_rmse, ax_r2) = plt.subplots(1, 2, figsize=(14, 5))

        bars = ax_rmse.bar(JOINT_NAMES, joint_rmse, color=ur_style.BLUE, edgecolor=ur_style.NAVY)
        ax_rmse.axhline(ovr["rmse"] * scale, color=ur_style.GRAY, linestyle="--", linewidth=1.2,
                        label=f"Overall RMSE={ovr['rmse'] * scale:.4f}")
        _annotate_bars(ax_rmse, bars, joint_rmse)
        ax_rmse.set_ylabel(f"RMSE ({unit})")
        ax_rmse.set_title(f"Held-out {ch} RMSE per joint")
        ax_rmse.tick_params(axis="x", rotation=20)
        ax_rmse.legend()

        bar_colors = [ur_style.BLUE if v >= 0 else ur_style.NAVY for v in joint_r2]
        bars = ax_r2.bar(JOINT_NAMES, joint_r2, color=bar_colors, edgecolor=ur_style.NAVY)
        ax_r2.axhline(0, color=ur_style.GRAY, linewidth=0.8)
        ax_r2.axhline(ovr["r2"], color=ur_style.GRAY, linestyle="--", linewidth=1.2,
                     label=f"Overall R²={ovr['r2']:.4f}")
        _annotate_bars(ax_r2, bars, joint_r2)
        ax_r2.set_ylabel("R²")
        ax_r2.set_title(f"Held-out {ch} R² per joint")
        ax_r2.tick_params(axis="x", rotation=20)
        ax_r2.legend()

        fig.tight_layout()
        path = os.path.join(run_dir, f"per_joint_metrics_{ch}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[results] plot -> {path}")


def _plot_coefficients(model: DistillModel, run_dir: str):
    """Bar chart of a linear model's fitted weights, one subplot per joint.

    Skipped (with a print, not an error) for a model whose ``coefficients()``
    returns ``None`` -- e.g. a non-linear regressor has no per-feature weight
    to show, so there's nothing useful to plot here.
    """
    coefs = model.coefficients()
    if coefs is None:
        print(f"[results] coefficients plot skipped -- {type(model).__name__} "
              "has no linear coefficients to show")
        return
    ur_style.apply()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for j in range(N_JOINTS):
        joint = JOINT_NAMES[j]
        feature_names, weights = coefs[joint]
        ax = axes[j // 3][j % 3]
        bar_colors = [ur_style.BLUE if w >= 0 else ur_style.NAVY for w in weights]
        bars = ax.bar(feature_names, weights, color=bar_colors, edgecolor=ur_style.NAVY)
        ax.axhline(0, color=ur_style.GRAY, linewidth=0.8)
        _annotate_bars(ax, bars, weights, fmt="{:+.4f}")
        ax.set_title(joint)
        ax.set_ylabel("weight")
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle(f"{type(model).__name__} coefficients ({model.predicts()[0]})")
    fig.tight_layout()
    path = os.path.join(run_dir, "coefficients.png")
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
    ur_style.apply()
    channels  = model.predicts()
    n_ch      = len(channels)
    x         = list(range(len(summary_df)))
    labels    = summary_df["datetime"].tolist()
    fig, axes = plt.subplots(2, n_ch, figsize=(max(6, 4 * len(x)), 8 * n_ch // n_ch),
                             squeeze=False)
    for col, ch in enumerate(channels):
        scale, unit = _plot_units(ch)
        for row_idx, (metric, color, ylabel) in enumerate(
            [("rmse", ur_style.BLUE, f"RMSE ({unit})"), ("r2", ur_style.MID_BLUE, "R²")]
        ):
            col_name = f"{ch}_{metric}"
            ax = axes[row_idx][col]
            if col_name in summary_df.columns:
                y = summary_df[col_name] * scale if metric == "rmse" else summary_df[col_name]
                ax.plot(x, y, marker="o", color=color)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ch} — held-out {ylabel} over runs")
    fig.tight_layout()
    path = os.path.join(results_dir, "comparison_plot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[results] comparison -> {path}")


def _row_split_eval(model: DistillModel, recordings, holdout: float) -> dict:
    """Row-level held-out evaluation: pool every row of every joint, mark a
    deterministic fraction as "test", fit a throwaway model on the rest.

    Matches ``original_train.py``'s split (``step = round(1/holdout)``, every
    step-th row by index, no RNG). Requires ``model._design(recordings)`` ->
    ``{joint: (X, y)}`` with ``y = actual_<channel> - target_<channel>`` (what
    ``PerJointPositionModel`` provides) and ``model._row_split_fit_predict``
    for the throwaway fit -- neither is part of the generic ``DistillModel``
    interface, so a model without both can't use this. The throwaway fit
    itself is delegated to the model (``_row_split_fit_predict``) rather than
    hard-coded here, so the held-out number reflects each model's own fitting
    method (e.g. gradient-boosted trees for ``PerJointTreeModel``), not
    always a linear refit on whatever features ``_design`` builds.

    Returns an eval_data dict shaped like ``_evaluate_model``'s output (so it
    feeds ``_compute_metrics`` and the plotting functions unchanged): rebuilds
    "actual"/"target" for the held-out rows from ``target_q`` gathered
    independently (concatenated over ``recordings`` in the same order
    ``_design`` iterates them), rather than assuming a feature-column index,
    so it doesn't silently break if the feature layout changes.
    """
    ch = model.predicts()[0]
    design = model._design(recordings)
    target_by_joint = {j: np.concatenate([rec.target_q[:, j] for rec in recordings])
                       for j in range(N_JOINTS)}

    n = len(next(iter(design.values()))[1])
    step = max(int(round(1 / holdout)), 2)
    is_test = np.arange(n) % step == 0

    eval_data = {ch: {}}
    for j, (X, y) in design.items():
        pred_res = model._row_split_fit_predict(X[~is_test], y[~is_test], X[is_test])
        tgt    = target_by_joint[j][is_test]
        actual = tgt + y[is_test]
        pred   = tgt + pred_res
        eval_data[ch][j] = {"pred": pred, "actual": actual, "target": tgt,
                            "residuals": pred - actual}
    return eval_data


def log_run(model: DistillModel, csvs: list, holdout: float,
            held_out_eval: dict, in_sample_metrics: dict,
            results_dir: str = None, dt_str: str = None):
    """Log a training run: save log.json, residual plot, per-joint RMSE plot,
    update runs_summary.csv and regenerate the comparison plot.

    ``held_out_eval`` (from ``_row_split_eval``, a throwaway fit on 80% of the
    rows) is the honest generalization number. ``in_sample_metrics`` is the
    *saved* model (refit on 100% of the rows) scored on its own training data
    -- a sanity check, not a generalization measure: it should look at least
    as good as the held-out numbers, and if it doesn't, the fit itself is
    broken.

    Args:
        model:              the DistillModel already refit on 100% of ``csvs``.
        csvs:                every recording used (train and, in this row-level
                             split, "test" alike -- see module docstring).
        holdout:             row fraction used for ``held_out_eval``, for the log.
        held_out_eval:       from ``_row_split_eval``.
        in_sample_metrics:   from ``_compute_metrics(_evaluate_model(model, recordings))``.
        results_dir:         override for RESULTS_DIR.
        dt_str:              override for the run timestamp (default: now).
    """
    results_dir = results_dir or RESULTS_DIR
    dt_str      = dt_str or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir     = os.path.join(results_dir, dt_str)
    os.makedirs(run_dir, exist_ok=True)

    held_out_metrics = _compute_metrics(held_out_eval)

    log = {
        "datetime":          dt_str,
        "model_class":       type(model).__name__,
        "params":            model.params(),
        "training": {
            "csvs":    csvs,
            "holdout": holdout,
            "split":   "row-level (every ~1/holdout-th row, deterministic); "
                      "saved model refit on 100% of rows after the held-out check",
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
    _plot_residuals_per_joint(held_out_eval, held_out_metrics, model.predicts(), run_dir)
    _plot_per_joint_metrics(held_out_metrics, model.predicts(), run_dir)
    _plot_coefficients(model, run_dir)
    summary_row = {ch: m["overall"] for ch, m in held_out_metrics.items()}
    _update_summary(model, summary_row, results_dir, dt_str)
    print(f"[results] run complete -> {run_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Fit a DistillModel on all recorded runs (original_train.py's "
                    "methodology): hold out a row-level fraction to report error, "
                    "then refit on 100% of the rows and save that. See the module "
                    "docstring for the row-level-split caveat.")
    ap.add_argument("--csvs", nargs="+", default=DEFAULT_CSVS,
                    help="recordings to train on (default: all of them)")
    ap.add_argument("--holdout", type=float, default=DEFAULT_HOLDOUT,
                    help="row fraction held out for the printed/plotted/logged "
                        "error report (default: %(default)s); the saved model "
                        "is refit on 100%% of the rows regardless")
    ap.add_argument("--model", choices=sorted(MODELS), default="linear_per_joint",
                    help="DistillModel to train (default: %(default)s)")
    ap.add_argument("--out", default="models/distill.pkl", help="pickle path")
    args = ap.parse_args()

    # Import under the real module name (not "__main__") so the saved pickle
    # loads cleanly in train_rla.py and run.py.
    from train_distillation_model import MODELS as _MODELS
    from analysis import Recording

    pre = default_preprocess()
    recordings = [Recording(r.path, df=pre.transform_distill(r.df))
                 for r in (Recording(p) for p in args.csvs)]

    model = _MODELS[args.model]()

    held_out_eval    = _row_split_eval(model, recordings, args.holdout)
    held_out_metrics = _compute_metrics(held_out_eval)
    n_rows  = len(next(iter(held_out_eval[model.predicts()[0]].values()))["actual"])
    print(f"row-level held-out ({args.holdout:.0%} of rows, {n_rows} rows/joint) "
          f"from {len(recordings)} run(s): {[os.path.basename(p) for p in args.csvs]}")
    for ch, m in held_out_metrics.items():
        ovr = m["overall"]
        print(f"  {ch}: RMSE={ovr['rmse']:.4f}  R2={ovr['r2']:.4f}")
        for pj in m["per_joint"]:
            print(f"    {pj['joint']:10s} RMSE={pj['rmse']:.4f}  R2={pj['r2']:.4f}")

    # Refit on 100% of the rows -- this is the model that ships.
    model.fit(recordings)
    in_sample_metrics = _compute_metrics(_evaluate_model(model, recordings))
    model.save(args.out)             # models/distill.pkl — "latest" for pipeline defaults
    print(f"refit on 100% of {len(recordings)} run(s), saved {args.out}")

    # Log this run: save results/<datetime>/{log.json, model, plots} and update summary.
    dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_run(model, args.csvs, args.holdout, held_out_eval, in_sample_metrics, dt_str=dt_str)


if __name__ == "__main__":
    main()
