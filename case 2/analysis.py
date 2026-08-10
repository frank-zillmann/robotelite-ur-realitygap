"""Per-joint stats and a plot for a recorded UR run.

Load a CSV from ``record.py``, print per-joint numbers (range of motion, current
gap, position lag), and plot one joint's target vs actual current.

    python analysis.py --csv data/test-4.csv --joint 1

``--joint`` selects the joint (0=base ... 5=wrist3).

``Recording`` is the shared CSV data loader used across the pipeline
(``train_distillation_model.py``, ``train_rla.py``); it wraps a run's CSV as
numpy arrays.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from utils import ACC_COL, JOINT_NAMES, N_JOINTS, SCL_COL, SCRIPT_COL, TIME_COL, VEL_COL

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


class Recording:
    """One recorded run loaded from a ``record.py`` CSV, as numpy arrays."""

    def __init__(self, path: str, df=None):
        # Loads ``path``, or wraps an already-loaded table if ``df`` is given
        # (e.g. one a preprocess.Preprocess transformed).
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
        """Median sample period (s)."""
        return float(np.median(np.diff(self.t)))

    def current_gap(self, joint: int) -> np.ndarray:
        """Actual minus target current for one joint (A), per row."""
        return self.actual_current[:, joint] - self.target_current[:, joint]

    def plot(self, joint: int = 1):
        """Plot one joint's target vs actual current, and the current gap below.

        Top: the two current traces over the run. Bottom: actual minus target
        current.
        """
        import matplotlib.pyplot as plt

        name = JOINT_NAMES[joint]
        gap = self.current_gap(joint)
        fig, (ax_c, ax_g) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

        ax_c.plot(self.t, self.target_current[:, joint], label="target current", lw=2)
        ax_c.plot(self.t, self.actual_current[:, joint], label="actual current", lw=1)
        ax_c.set_ylabel("current (A)")
        ax_c.set_title(f"{name} joint")
        ax_c.legend(loc="best")

        ax_g.plot(self.t, gap, color="tab:red", lw=1)
        ax_g.axhline(0, color="grey", lw=0.8)
        ax_g.set_ylabel("actual - target (A)")
        ax_g.set_xlabel("time (s)")

        fig.tight_layout()
        return fig


def _per_joint_stats(rec) -> list[dict]:
    """Compute per-joint stats and return as a list of dicts."""
    stats = []
    for j in range(N_JOINTS):
        moved   = float(rec.target_q[:, j].max() - rec.target_q[:, j].min())
        gap     = rec.current_gap(j)
        stats.append({
            "joint":        JOINT_NAMES[j],
            "moved_rad":    moved,
            "gap_rms_A":    float(np.sqrt(np.mean(gap ** 2))),
            "gap_max_A":    float(np.max(np.abs(gap))),
            "pos_err_mrad": float(np.sqrt(np.mean(
                                (rec.actual_q[:, j] - rec.target_q[:, j]) ** 2))) * 1e3,
        })
    return stats


def log_analysis(rec, joint: int, stats: list, csv_path: str,
                 results_dir: str, dt_str: str):
    """Save log.json and plots to results/<datetime>_analysis_<csv_base>/."""
    import matplotlib.pyplot as plt

    base    = os.path.splitext(os.path.basename(csv_path))[0]
    run_dir = os.path.join(results_dir, f"{dt_str}_analysis_{base}")
    os.makedirs(run_dir, exist_ok=True)

    # ---- log.json ------------------------------------------------------------
    log = {
        "datetime":  dt_str,
        "csv":       csv_path,
        "n_rows":    len(rec.t),
        "dt_ms":     round(rec.dt * 1e3, 3),
        "per_joint": stats,
    }
    log_path = os.path.join(run_dir, "log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[results] log  -> {log_path}")

    # ---- single-joint current plot -------------------------------------------
    fig = rec.plot(joint)
    plot_path = os.path.join(run_dir, f"current_{JOINT_NAMES[joint]}.png")
    fig.savefig(plot_path, dpi=150)
    print(f"[results] plot -> {plot_path}")

    # ---- all-joints overview (3 bar charts) ----------------------------------
    names    = [s["joint"]        for s in stats]
    gap_rms  = [s["gap_rms_A"]    for s in stats]
    gap_max  = [s["gap_max_A"]    for s in stats]
    pos_err  = [s["pos_err_mrad"] for s in stats]

    fig2, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, vals, title, ylabel, color in [
        (axes[0], gap_rms, "Current gap RMS",     "A",    "steelblue"),
        (axes[1], gap_max, "Current gap max |A|", "A",    "darkorange"),
        (axes[2], pos_err, "Position error RMS",  "mrad", "purple"),
    ]:
        bars = ax.bar(names, vals, color=color, edgecolor="black")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
    fig2.suptitle(os.path.basename(csv_path), fontsize=10)
    fig2.tight_layout()
    overview_path = os.path.join(run_dir, "all_joints_overview.png")
    fig2.savefig(overview_path, dpi=150)
    plt.close(fig2)
    print(f"[results] plot -> {overview_path}")
    print(f"[results] run complete -> {run_dir}")

    return fig   # caller decides whether to show interactively


def main():
    ap = argparse.ArgumentParser(description="Per-joint stats and a plot for a recorded run.")
    ap.add_argument("--csv", default="data/test-4.csv", help="recorded run CSV")
    ap.add_argument("--joint", type=int, default=1,
                    help="joint index 0..5 to plot (default 1 = shoulder)")
    ap.add_argument("--no-plot", action="store_true", help="save results but do not show plots")
    args = ap.parse_args()

    rec = Recording(args.csv)
    print(f"{args.csv}  ({len(rec.t)} rows, dt {rec.dt*1e3:.1f} ms)")

    stats = _per_joint_stats(rec)
    print(f"{'joint':10s} {'moved':>9s} {'gap RMS':>9s} {'gap max':>9s} {'pos err':>10s}")
    for s in stats:
        print(f"{s['joint']:10s} {s['moved_rad']:8.3f}r {s['gap_rms_A']:8.3f}A "
              f"{s['gap_max_A']:8.3f}A {s['pos_err_mrad']:7.2f}mrad")

    dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fig = log_analysis(rec, args.joint, stats, args.csv, RESULTS_DIR, dt_str)

    if not args.no_plot:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__ == "__main__":
    main()
