"""Plot target vs actual for one value/component of a recorded run.

Generalizes analysis.py's current-only plot to any target_*/actual_* channel
present in a record.py CSV (q, qd, current, TCP_pose, TCP_speed), for one
chosen component.

    python plot_target_actual.py --csv data/test-4.csv --value current --joint 1
    python plot_target_actual.py --csv data/test-4.csv --value TCP_pose --joint 2
    python plot_target_actual.py --csv data/test-4.csv --value q --joint 0 --save q0.png --no-show

``--joint`` selects the component: joint index 0..5 for q/qd/current, or
x,y,z,rx,ry,rz (also indexed 0..5) for TCP_pose/TCP_speed.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from utils import JOINT_NAMES, N_JOINTS, TIME_COL, get_block

# Values with both target_<value><i> and actual_<value><i> columns in the CSV,
# and the per-component labels/units for each.
VALUE_LABELS = {
    "q":         JOINT_NAMES,
    "qd":        JOINT_NAMES,
    "current":   JOINT_NAMES,
    "TCP_pose":  ("x", "y", "z", "rx", "ry", "rz"),
    "TCP_speed": ("x", "y", "z", "rx", "ry", "rz"),
}

VALUE_UNITS = {
    "q":         "rad",
    "qd":        "rad/s",
    "current":   "A",
    "TCP_pose":  "m / rad",
    "TCP_speed": "m/s / rad/s",
}


def plot_target_actual(csv_path: str, value: str, component: int,
                        save: str | None = None, show: bool = True):
    """Plot target vs actual (top) and their gap (bottom) for one component."""
    import matplotlib.pyplot as plt

    if value not in VALUE_LABELS:
        raise ValueError(f"unknown --value {value!r}; choose from {sorted(VALUE_LABELS)}")
    if not 0 <= component < N_JOINTS:
        raise ValueError(f"--joint must be 0..{N_JOINTS - 1}")

    df = pd.read_csv(csv_path)
    t = df[TIME_COL].to_numpy(dtype=float)
    target = get_block(df, f"target_{value}")[:, component]
    actual = get_block(df, f"actual_{value}")[:, component]
    gap = actual - target

    label = VALUE_LABELS[value][component]
    unit = VALUE_UNITS[value]

    fig, (ax_v, ax_g) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax_v.plot(t, target, label="target", lw=2)
    ax_v.plot(t, actual, label="actual", lw=1)
    ax_v.set_ylabel(f"{value} ({unit})")
    ax_v.set_title(f"{value} — {label}  [{os.path.basename(csv_path)}]")
    ax_v.legend(loc="best")

    ax_g.plot(t, gap, color="tab:red", lw=1)
    ax_g.axhline(0, color="grey", lw=0.8)
    ax_g.set_ylabel("actual - target")
    ax_g.set_xlabel("time (s)")

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150)
        print(f"[plot] saved -> {save}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def main():
    ap = argparse.ArgumentParser(
        description="Plot target vs actual for one value/component of a recorded run.")
    ap.add_argument("--csv", required=True, help="recorded run CSV")
    ap.add_argument("--value", default="current", choices=sorted(VALUE_LABELS),
                    help="which channel to plot (default: current)")
    ap.add_argument("--joint", type=int, default=1,
                    help="component index 0..5 (joint for q/qd/current; "
                         "x,y,z,rx,ry,rz for TCP_*)")
    ap.add_argument("--save", default=None,
                    help="save the figure to this path (in addition to showing, unless --no-show)")
    ap.add_argument("--no-show", action="store_true",
                    help="don't open a window (useful headless / remote)")
    args = ap.parse_args()

    plot_target_actual(args.csv, args.value, args.joint,
                       save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
