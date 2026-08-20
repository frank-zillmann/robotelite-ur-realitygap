"""Interactive plot of recorded UR runs.

Loads a CSV from ``record.py`` and plots up to three sources of one quantity:

    target          what the controller commanded  (target_* columns)
    actual          what the robot measured        (actual_* columns)
    model           what the distilled model predicts for those commands
                    (``--model``), pale, with a band of ``--sd-factor`` standard
                    deviations

Each run gets one colour; within it the target is dashed, the measured actual solid,
and the model pale and thick behind them.

    python analysis.py --csv baseline.csv optimized.csv --path baseline.path optimized.path \
        --model models/distill-ur5e.pkl --robot UR5e

Several runs can be given at once; their gap, predicted gap, and optimizer
loss are printed for comparison. Cycle time comes from ``--path`` (one
entry per ``--csv``, same order); gap and predicted_gap are RMSE over all
recorded rows, the former measured (actual vs. target), the latter the
model's guess at it from target alone.

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

from send import load_path
from utils import (ACC_COL, DT, JOINT_NAMES, N_JOINTS, SCL_COL, SCRIPT_COL,
                   TIME_COL, VEL_COL)

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


def cycle_time(path: str) -> float:
    """One commanded cycle, from a `.path` file."""
    rows = np.asarray(load_path(path), float)
    return float(rows[:, N_JOINTS].sum()) if rows.shape[1] > N_JOINTS else len(rows) * DT


def stats(recs, paths, model, robot):
    """Print measured and predicted gap, cycle time, and loss per trajectory.

    ``paths`` matches each recording one-to-one -- a run's `.csv` no longer implies
    its `.path` by name (e.g. `foo.stream.csv` from send.py), so it's explicit.

    ``gap`` is the true RMSE between ``actual_q`` and ``target_q``; ``predicted_gap``
    is the model's guess at that from ``target_q`` alone -- what the optimizer
    actually chases, since it never sees ``actual_q``. ``loss`` mirrors the
    optimizer's training loss, built from the predicted gap. The barrier itself
    isn't scored here (an optimization device, and the trajectory is already known
    to respect the limits), so it's evaluated at weight 0.
    """
    import torch
    from optimize import loss as loss_fn
    from optimize import penalties

    q = [torch.as_tensor(rec.target_q, dtype=torch.float32) for rec in recs]
    times = [cycle_time(p) for p in paths]
    measured = [float(np.sqrt(np.mean((rec.actual_q - rec.target_q) ** 2))) for rec in recs]
    with torch.no_grad():
        base_gap, _, _ = penalties(model, robot, q[0], weight=0)
        scale = times[0] / float(base_gap.clamp_min(1e-8))
        rows = [loss_fn(model, robot, qi, cycle, scale, weight=0)
                for qi, cycle in zip(q, times)]

    names = [os.path.basename(rec.path) for rec in recs]
    width = max(map(len, names))
    print(f"  {'run':{width}s} {'gap [mrad]':>12s} {'predicted_gap [mrad]':>21s} "
          f"{'cycle [s]':>10s} {'loss':>10s}")
    for name, cycle, gap_meas, (total, gap_pred, _, _) in zip(names, times, measured, rows):
        print(f"  {name:{width}s} {gap_meas * 1000:12.4f} {float(gap_pred) * 1000:21.4f} "
              f"{cycle:10.3f} {float(total):10.4f}")


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
    ap.add_argument("--path", required=True, nargs="+",
                    help="the .path matching each --csv, one-to-one and in the same order")
    ap.add_argument("--quantity", choices=list(QUANTITIES), default="angle q",
                    help="which channel to plot")
    ap.add_argument("--joint", type=int, required=True, choices=range(N_JOINTS),
                    help="component 0..5: a joint (base..wrist3), or a Cartesian "
                         "axis (x, y, z, rx, ry, rz) for the TCP quantities")
    ap.add_argument("--model", required=True,
                    help="distilled model pickle")
    ap.add_argument("--robot", required=True, choices=("UR5e", "UR10e"),
                    help="arm whose kinematics/limits the model's predictions are checked against")
    ap.add_argument("--sd-factor", type=float, default=1.0,
                    help="width of the uncertainty bands, in standard deviations")
    ap.add_argument("--max-points", type=int, default=10000,
                    help="rows drawn per trace, from the start of the run; the page "
                         "grows by ~0.2 MB per 1000 rows and trace")
    args = ap.parse_args()
    if len(args.path) != len(args.csv):
        raise SystemExit(f"--path needs one entry per --csv ({len(args.csv)}), got {len(args.path)}")

    from train_distillation_model import DistillModel      # pulls in torch
    from utils import Robot

    recs = [Recording(p) for p in args.csv]
    model = DistillModel.load(args.model)
    stats(recs, args.path, model, Robot(args.robot))
    show(view(recs, args.quantity, args.joint, model, args.sd_factor, args.max_points))


if __name__ == "__main__":
    main()
