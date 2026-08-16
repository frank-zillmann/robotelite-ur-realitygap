"""Per-run diagnostic metrics -- not the RL score (see ``metrics.py``, the single
per-row number the agent optimizes), but numbers a human reads to judge a run's
quality: tracking accuracy, overshoot/settling/smoothness of each move, cycle
time, and -- given a ``DistillModel`` and held-out recordings -- how well the
reality-gap model itself predicts the real robot. Consumed by ``report.py``.

    from analysis import Recording
    from diagnostics import per_joint_metrics, per_run_metrics
    rec = Recording("baseline.csv")
    per_joint_metrics(rec)      # {joint name: {pos_rmse, pos_max_err, current_gap_rms, jerk_rms, jerk_peak}}
    per_run_metrics(rec)        # {cycle_time, overshoot_mean/max, settling_time_mean/max, residual_vib_rms_mean/max}

Reality-gap prediction error is separate (it needs a model, not just a
recording):

    from train_distillation_model import DistillModel
    from diagnostics import reality_gap_error
    model = DistillModel.load("models/distill.pkl")
    reality_gap_error(model, [Recording("data/test-1.csv")])   # {rmse, r2}
"""
from __future__ import annotations

import numpy as np

from common import segments
from preprocess import default_preprocess
from utils import JOINT_NAMES, N_JOINTS, get_block, set_block

SETTLE_TOL = 0.01   # rad: a joint counts as "settled" once |actual - dest| stays under this


def jerk(qd: np.ndarray, dt: float) -> np.ndarray:
    """Jerk (rad/s^3), per joint: the time-derivative of acceleration, itself
    the derivative of the measured velocity ``qd`` (RTDE's ``actual_qd``,
    rather than differentiating position three times)."""
    qdd = np.gradient(qd, dt, axis=0)
    return np.gradient(qdd, dt, axis=0)


def per_joint_metrics(rec) -> dict:
    """Whole-run, per joint: position RMSE/max error, current-gap RMS, RMS/peak jerk."""
    pos_err = rec.actual_q - rec.target_q
    cur_gap = rec.actual_current - rec.target_current
    j3 = jerk(rec.actual_qd, rec.dt)
    out = {}
    for j in range(N_JOINTS):
        out[JOINT_NAMES[j]] = {
            "pos_rmse": float(np.sqrt(np.mean(pos_err[:, j] ** 2))),
            "pos_max_err": float(np.max(np.abs(pos_err[:, j]))),
            "current_gap_rms": float(np.sqrt(np.mean(cur_gap[:, j] ** 2))),
            "jerk_rms": float(np.sqrt(np.mean(j3[:, j] ** 2))),
            "jerk_peak": float(np.max(np.abs(j3[:, j]))),
        }
    return out


def _segment_diagnostics(rec, seg) -> dict:
    """Peak overshoot, settling time, and residual-vibration RMS for one segment,
    read on its own (widest-travel) joint over its settle window ``[i1, i2)``.
    """
    window = rec.actual_q[seg.i1:seg.i2, seg.joint]
    if len(window) == 0:
        return {"overshoot": 0.0, "settling_time": 0.0, "residual_vib_rms": 0.0}
    sign = 1.0 if seg.dest >= seg.start else -1.0
    overshoot = float(max(((window - seg.dest) * sign).max(), 0.0))
    err = np.abs(window - seg.dest)
    unsettled = np.where(err > SETTLE_TOL)[0]     # rows still outside the settle band
    settling_time = float((unsettled[-1] + 1) * rec.dt) if len(unsettled) else 0.0
    residual_vib_rms = float(np.sqrt(np.mean(err ** 2)))
    return {"overshoot": overshoot, "settling_time": settling_time,
            "residual_vib_rms": residual_vib_rms}


def segment_metrics(rec) -> list[dict]:
    """Per-segment (one waypoint-to-waypoint move): joint, overshoot, settling
    time, and residual-vibration RMS in its post-stop settle window."""
    return [{"joint": JOINT_NAMES[seg.joint], "i0": seg.i0, **_segment_diagnostics(rec, seg)}
            for seg in segments(rec)]


def per_run_metrics(rec) -> dict:
    """Whole-run: cycle time, plus overshoot/settling/vibration averaged (and
    peaked) over every segment in the run."""
    segs = segment_metrics(rec)
    out = {"cycle_time": float(rec.t[-1] - rec.t[0])}
    for key in ("overshoot", "settling_time", "residual_vib_rms"):
        vals = [s[key] for s in segs] or [0.0]
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_max"] = float(np.max(vals))
    return out


def reality_gap_error(model, recordings, pre=None) -> dict:
    """RMSE/R2 of ``model``'s predicted ``actual_*`` channels against the real
    recorded ones, over every joint of every recording -- whether the distilled
    model predicts the real robot, not just how well it fit its own training rows.

    ``pre`` (a preprocess.Preprocess, default ``default_preprocess()``) is applied
    the same way ``augment`` uses it: ``transform_distill`` before ``predict``,
    ``revert_distill`` after, so predictions land back in the real units the
    recorded ``actual_*`` columns are already in. Matters once ``pre`` stops being
    a no-op; harmless (an identity round trip) while it still is.
    """
    pre = pre or default_preprocess()
    err, y = [], []
    for rec in recordings:
        transformed = pre.transform_distill(rec.df)
        preds = model.predict(transformed)
        for base in model.predicts():
            pred_df = transformed.copy()
            set_block(pred_df, base, preds[base])
            pred_df = pre.revert_distill(pred_df)
            predicted = get_block(pred_df, base)
            actual = get_block(rec.df, base)
            err.append((predicted - actual).ravel())
            y.append(actual.ravel())
    err, y = np.concatenate(err), np.concatenate(y)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    r2 = float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2))
    return {"rmse": rmse, "r2": r2}
