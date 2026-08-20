"""Interactive plot of recorded UR runs.

Loads a CSV from ``record.py`` and plots up to three sources of one quantity:

    target          what the controller commanded  (target_* columns)
    actual          what the robot measured        (actual_* columns)
    model           what the distilled model predicts for those commands
                    (``--model``), with a band of ``--sd-factor`` standard
                    deviations, aleatoric and with the ensemble's disagreement added

    python analysis.py --csv baseline.csv optimized.csv --model models/distill-ur5e.pkl

Several runs can be given at once; each is traced as ``<file> - target`` and
``<file> - actual``, and their error and cycle time are printed for comparison.

Only the first ``--max-points`` rows are plotted, at the recording's full rate.

``--joint`` picks the component, which is a joint (0..5 = base..wrist3) for the
per-joint quantities and a Cartesian axis (0..5 = x, y, z, rx, ry, rz) for the
TCP ones, since those are poses in space and have nothing to do with joints.

``Recording`` is the shared CSV data loader used across the pipeline
(``train_distillation_model.py``, ``optimize.py``); it wraps a run's CSV as
numpy arrays.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from utils import (ACC_COL, JOINT_NAMES, N_JOINTS, SCL_COL, SCRIPT_COL, TIME_COL,
                   VEL_COL)

TCP_AXES = ("x", "y", "z", "rx", "ry", "rz")

# Quantities the viewer can show: label -> (target base, actual base, unit, component names).
QUANTITIES = {
    "current":   ("target_current", "actual_current", "A", JOINT_NAMES),
    "angle q":   ("target_q", "actual_q", "rad", JOINT_NAMES),
    "speed qd":  ("target_qd", "actual_qd", "rad/s", JOINT_NAMES),
    "accel qdd": ("target_qdd", "actual_qdd", "rad/s^2", JOINT_NAMES),
    "moment":    ("target_moment", "actual_current_as_torque", "Nm", JOINT_NAMES),
    "TCP pose":  ("target_TCP_pose", "actual_TCP_pose", "m, rad", TCP_AXES),
    "TCP speed": ("target_TCP_speed", "actual_TCP_speed", "m/s, rad/s", TCP_AXES),
}


class Recording:
    """One recorded run loaded from a ``record.py`` CSV, as numpy arrays."""

    def __init__(self, path: str, df=None):
        # Loads ``path``, or wraps an already-loaded table if ``df`` is given
        # (e.g. one already transformed by the caller).
        if df is None:
            df = pd.read_csv(path)
        self.path = path
        self.df = df             # raw table; callers can slice rows by index
        self.t = df[TIME_COL].to_numpy(dtype=float)
        self.target_q = np.column_stack([df[f"target_q{j}"] for j in range(6)])
        self.actual_q = np.column_stack([df[f"actual_q{j}"] for j in range(6)])
        self.target_qd = np.column_stack([df[f"target_qd{j}"] for j in range(6)])
        self.target_current = np.column_stack([df[f"target_current{j}"] for j in range(6)])
        self.actual_current = np.column_stack([df[f"actual_current{j}"] for j in range(6)])
        # Commanded movej parameters, if record.py logged them (raw URScript
        # values, e.g. 100). None when the run has no vel/acc registers.
        self.vel_cmd = df[VEL_COL].to_numpy(dtype=float) if VEL_COL in df else None
        self.acc_cmd = df[ACC_COL].to_numpy(dtype=float) if ACC_COL in df else None
        # URScript line currently executing (RTDE): nonzero while a movej runs, 0
        # during a sleep/dwell. The value is script-specific and not comparable
        # across scripts, but a change marks a move boundary. Zeros if not logged.
        self.scl = (df[SCL_COL].to_numpy() if SCL_COL in df
                    else np.zeros(len(self.t), dtype=int))
        # Source script of each row, tagged when several runs are pooled into one
        # file, so segmentation never spans two scripts. One value if not tagged.
        self.script = (df[SCRIPT_COL].to_numpy() if SCRIPT_COL in df
                       else np.zeros(len(self.t), dtype=int))

    @property
    def dt(self) -> float:
        """Sample period (s): elapsed time over the number of steps."""
        return float((self.t[-1] - self.t[0]) / (len(self.t) - 1))

    def channel(self, base: str):
        """A per-joint channel as ``(n, N_JOINTS)``, or ``None`` if not recorded.

        ``actual_qdd`` is not an RTDE field, so it is differentiated from
        ``actual_qd`` on demand; everything else is read straight from the CSV.
        """
        if base is None:
            return None
        cols = [f"{base}{j}" for j in range(N_JOINTS)]
        if all(c in self.df for c in cols):
            return self.df[cols].to_numpy(dtype=float)
        if base == "actual_qdd":
            qd = self.channel("actual_qd")
            return None if qd is None else np.gradient(qd, self.dt, axis=0)
        return None


# --- viewer -------------------------------------------------------------------

# Plotly's qualitative palette, taken in order as traces are added.
PALETTE = ("#636efa", "#ef553b", "#00cc96", "#ab63fa", "#ffa15a", "#19d3f3",
           "#ff6692", "#b6e880", "#ff97ff", "#fecb52")
_rgba = lambda c, a: f"rgba({int(c[1:3], 16)},{int(c[3:5], 16)},{int(c[5:7], 16)},{a})"


def stats(recs: list):
    """Print the tracking error and cycle time of each run, scored like optimize.py.

    Measured, not predicted: ``actual_q`` against ``target_q`` over the moves, so a
    baseline and an optimized run can be compared directly. The score uses the same
    weighting the optimizer minimizes and is relative to the first run, which
    therefore scores ``1 + ALPHA`` -- anything below that is an improvement on it.
    """
    from common import segments
    from optimize import ALPHA
    from utils import DT

    rows = []
    for rec in recs:
        segs = segments(rec)
        a, b = (segs[0].i0, segs[-1].i2) if segs else (0, len(rec.t))
        err = np.abs(rec.actual_q[a:b] - rec.target_q[a:b])
        rows.append((os.path.basename(rec.path), err.mean(), err.max(), (b - a) * DT))
    print(f"  {'run':30s} {'error [mrad]':>13s} {'worst':>8s} {'cycle [s]':>10s} {'score':>7s}")
    for name, mean, worst, T in rows:
        score = mean / rows[0][1] + ALPHA * T / rows[0][3]
        print(f"  {name:30s} {mean * 1000:13.4f} {worst * 1000:8.3f} {T:10.3f} {score:7.3f}")


def view(recs: list, quantity: str = "angle q", joint: int = 1, model=None,
         sd_factor: float = 1.0, max_points: int = 5000):
    """Plotly figure of one channel across several runs, each traced as
    ``<file> - target`` / ``- actual`` / ``- model``.

    Only the first ``max_points`` rows of each are shown, at the recording's full
    rate: a run can be ~174k rows, and every trace of it makes a page the browser
    chokes on.
    """
    import plotly.graph_objects as go

    t_base, a_base, unit, comps = QUANTITIES[quantity]
    keep = slice(None, max_points)
    colours = iter(PALETTE * 4)
    fig = go.Figure()
    for rec in recs:
        run = os.path.basename(rec.path)
        t = rec.t[keep]
        for tag, block in (("target", rec.channel(t_base)),
                           ("actual", rec.channel(a_base))):
            if block is not None:
                fig.add_scattergl(x=t, y=block[keep, joint], name=f"{run} - {tag}",
                                  line=dict(color=next(colours), width=1.5))
        if model is None or a_base not in model.predicts():
            continue
        # Predicting the shown rows only is exact, not an approximation: the model
        # is causal, so row t never depends on a row after it.
        p = model.predict(rec.df.iloc[keep])
        mean, colour = p["mean"][a_base][:, joint], next(colours)
        # Widest band first so the narrower one occludes it: the model's own noise
        # inside, the ensemble's disagreement added on top.
        for key, alpha, tag in (("var", 0.12, "+epistemic"),
                                ("var_aleatoric", 0.22, "aleatoric")):
            if key in p:
                half = sd_factor * np.sqrt(p[key][a_base][:, joint])
                fig.add_scattergl(x=np.concatenate([t, t[::-1]]),
                                  y=np.concatenate([mean + half, (mean - half)[::-1]]),
                                  fill="toself", fillcolor=_rgba(colour, alpha),
                                  line=dict(width=0), hoverinfo="skip",
                                  name=f"{run} - {sd_factor:g}sd {tag}")
        fig.add_scattergl(x=t, y=mean, name=f"{run} - model",
                          line=dict(color=colour, width=1.5))

    fig.update_layout(title=f"{quantity} - {comps[joint]}", xaxis_title="time [s]",
                      yaxis_title=f"{quantity} [{unit}]", hovermode="x unified",
                      template="plotly_white")
    return fig


def main():
    ap = argparse.ArgumentParser(description="Interactive plot of recorded UR runs.")
    ap.add_argument("--csv", required=True, nargs="+",
                    help="recorded run CSVs; the first is the one the rest are scored against")
    ap.add_argument("--quantity", choices=list(QUANTITIES), default="angle q",
                    help="which channel to plot")
    ap.add_argument("--joint", type=int, default=1, choices=range(N_JOINTS),
                    help="component 0..5: a joint (base..wrist3), or a Cartesian "
                         "axis (x, y, z, rx, ry, rz) for the TCP quantities")
    ap.add_argument("--model", default=None,
                    help="distilled model pickle; adds its prediction and an "
                         "uncertainty band for each run")
    ap.add_argument("--sd-factor", type=float, default=1.0,
                    help="width of the uncertainty bands, in standard deviations")
    ap.add_argument("--max-points", type=int, default=10000,
                    help="rows drawn per trace, from the start of the run; the page "
                         "grows by ~0.2 MB per 1000 rows and trace")
    args = ap.parse_args()

    recs = [Recording(p) for p in args.csv]
    stats(recs)
    model = None
    if args.model:
        from train_distillation_model import DistillModel      # pulls in torch
        model = DistillModel.load(args.model)
    view(recs, args.quantity, args.joint, model, args.sd_factor, args.max_points).show()


if __name__ == "__main__":
    main()
