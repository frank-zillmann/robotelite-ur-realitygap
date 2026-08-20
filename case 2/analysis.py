"""Interactive plot of recorded UR runs.

Loads a CSV from ``record.py`` and plots up to three sources of one quantity:

    target          what the controller commanded  (target_* columns)
    actual          what the robot measured        (actual_* columns)
    model           what the distilled model predicts for those commands
                    (``--model``), pale, with a band of ``--sd-factor`` standard
                    deviations

Each run gets one colour; within it the target is dashed, the measured actual solid,
and the model pale and thick behind them.

    python analysis.py --csv baseline.csv optimized.csv --model models/distill-ur5e.pkl

Several runs can be given at once; their error and cycle time are printed for
comparison.

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

# One base colour per run; within a run the three sources are told apart by style.
PALETTE = ("#d62728", "#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd", "#8c564b")
_rgba = lambda c, a: f"rgba({int(c[1:3], 16)},{int(c[3:5], 16)},{int(c[5:7], 16)},{a})"


def laps(q, tol: float = 1e-3) -> int:
    """How many times the commanded path repeats itself.

    ``send.py --loop N`` puts N repetitions in one file; dividing by this keeps the
    cycle time comparable however many were run.

    Counted as the times the path lands on the pose it *ends* at: a program that
    finished did whole laps, so that pose is on the cycle, while the one it started
    from need not be -- the first run of a batch opens with the robot travelling in
    from wherever it was. Landing is not enough on its own, since a figure eight
    begun at its crossing comes home twice a lap, so the last two stretches are
    compared and the count halved if they are not copies of each other.
    """
    home = np.abs(q - q[-1]).max(axis=1) < tol
    idx = np.flatnonzero(np.diff(home.astype(int)) > 0) + 1   # rows where it lands
    if len(idx) < 2:
        return max(1, len(idx))
    m = int(np.median(np.diff(idx)))                          # rows in one stretch
    tail = q[idx[-1] - 2 * m:idx[-1]]
    doubled = (len(tail) == 2 * m and np.abs(tail[:m] - tail[m:]).mean()
               > 0.08 * np.ptp(q, axis=0).max())
    return max(1, len(idx) // 2 if doubled else len(idx))


def stats(recs: list):
    """Print what each run measured, scored the way optimize.py scores a path.

    Measured, not predicted: ``actual_q`` against ``target_q``, so a baseline and an
    optimized run can be compared directly. The window is the rows where the command
    is moving, which trims the idle head and tail; the cycle time is that window
    divided by ``laps``, and it is what the robot *took*, not what the path asked
    for. The score weights error against time exactly as the optimizer does and is
    relative to the first run, which therefore reads ``1 + ALPHA``.

    A move the robot makes to reach the start counts as motion like any other, so
    compare runs that began from the same pose, or loop them enough that it washes
    out.
    """
    from optimize import ALPHA
    from utils import DT

    rows = []
    for rec in recs:
        mv = np.flatnonzero(np.abs(np.gradient(rec.target_q, DT, axis=0)).max(1) > 0.01)
        a, b = (mv[0], mv[-1] + 1) if len(mv) else (0, len(rec.t))
        err = np.abs(rec.actual_q[a:b] - rec.target_q[a:b])
        n = laps(rec.target_q)          # on the whole run: it starts at rest, on the cycle
        rows.append((os.path.basename(rec.path), err.mean(), err.max(), (b - a) * DT / n, n))

    w = max(len(r[0]) for r in rows)
    print(f"  {'run':{w}s} {'error [mrad]':>12s} {'worst':>8s} {'cycle [s]':>10s} "
          f"{'laps':>5s} {'score':>7s}")
    for name, mean, worst, T, n in rows:
        # A simulator tracks perfectly, so there is nothing to score against.
        score = (f"{mean / rows[0][1] + ALPHA * T / rows[0][3]:7.3f}"
                 if rows[0][1] > 0 else "      -")
        print(f"  {name:{w}s} {mean * 1000:12.4f} {worst * 1000:8.3f} {T:10.3f} "
              f"{n:5d} {score}")
    if rows[0][1] == 0:
        print("  (actual == target exactly: a simulator, so there is no error to score)")


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
    fig = go.Figure()
    for rec, colour in zip(recs, PALETTE * 4):
        run = os.path.basename(rec.path)
        t = rec.t[keep]
        for tag, block, dash in (("target", rec.channel(t_base), "dash"),
                                 ("actual", rec.channel(a_base), "solid")):
            if block is not None:
                fig.add_scattergl(x=t, y=block[keep, joint], name=f"{run} - {tag}",
                                  line=dict(color=colour, dash=dash, width=1.5))
        if model is None or a_base not in model.predicts():
            continue
        # Predicting the shown rows only is exact, not an approximation: the model
        # is causal, so row t never depends on a row after it.
        p = model.predict(rec.df.iloc[keep])
        mean = p["mean"][a_base][:, joint]
        # The bands stay out of the legend: they belong to the model line, which is
        # drawn pale so the measured one reads through it.
        for key, alpha in (("var", 0.10), ("var_aleatoric", 0.18)):
            if key in p:
                half = sd_factor * np.sqrt(p[key][a_base][:, joint])
                fig.add_scattergl(x=np.concatenate([t, t[::-1]]),
                                  y=np.concatenate([mean + half, (mean - half)[::-1]]),
                                  fill="toself", fillcolor=_rgba(colour, alpha),
                                  line=dict(width=0), hoverinfo="skip", showlegend=False)
        fig.add_scattergl(x=t, y=mean, name=f"{run} - model ({sd_factor:g}sd band)",
                          line=dict(color=_rgba(colour, 0.45), width=3))

    fig.update_layout(title=f"{quantity} - {comps[joint]}", xaxis_title="time [s]",
                      yaxis_title=f"{quantity} [{unit}]", hovermode="x unified",
                      template="plotly_white")
    return fig


def show(fig):
    """Open the figure, muting what the browser writes on its way up.

    Launching it inherits our stderr, so GTK and locale chatter lands in the middle
    of the table above. Python-level errors still raise normally.
    """
    with open(os.devnull, "w") as null:
        saved = os.dup(2)
        os.dup2(null.fileno(), 2)
        try:
            fig.show()
        finally:
            os.dup2(saved, 2)
            os.close(saved)


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
    show(view(recs, args.quantity, args.joint, model, args.sd_factor, args.max_points))


if __name__ == "__main__":
    main()
