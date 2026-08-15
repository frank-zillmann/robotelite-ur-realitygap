"""Bronze-tier exploration of the recorded RTDE dataset.

Loads every recording in ``data/test-*.csv``, segments each into individual
movej moves (``common.segments``), and computes two per-move metrics:

  - RMS position error : sqrt(mean((actual_q - target_q)^2)) over the whole
                          move, same formula ``analysis.py``'s per-joint
                          stats use (``pos_err_mrad``).
  - peak overshoot      : max |actual_q - dest| during the settle window
                          after the commanded motion nominally stops
                          (``seg.i1 .. seg.i2``) -- how far the real robot
                          rings past the target after arriving.

Each recorded run sweeps ``vel`` and/or ``acc`` down from 100 to 10
(test-1..5, one parameter at a time) or randomizes both together
(test-6, test-7), so pooling every move across every file gives a dense
(vel, acc) -> vibration dataset with no extra recording needed.

    python bronze_exploration.py

Outputs -> bronze_tier/:
    segment_stats.csv         one row per move: file, joint, vel, acc, dist,
                               rms_pos_err_mrad, peak_overshoot_mrad
    log.json                  headline numbers: counts, best/worst combos
    trajectories/*.png        target vs actual position at several vel/acc
                               settings overlaid (one plot per sweep file),
                               plus the single worst and best move in detail
    overshoot_vs_vel.png      \\
    overshoot_vs_acc.png       |  single-parameter sweeps (other held fixed)
    rms_vs_vel.png             |
    rms_vs_acc.png            /
    overshoot_heatmap.png     combined (vel, acc) grid, from the randomized
    rms_heatmap.png           test-6/test-7 runs
    per_joint_overshoot.png   bar chart, mean peak overshoot by joint
    per_run/test-N_overshoot_by_joint.png,   the presentation bar chart
    per_run/test-N_rms_by_joint.png          (degrees + % of move distance,
                               per joint) computed on one recording at a
                               time instead of pooled across all 7
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import Recording
from common import segments
from utils import JOINT_NAMES, N_JOINTS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_GLOB = os.path.join(HERE, "data", "test-*.csv")
OUT_DIR = os.path.join(HERE, "bronze_tier")

# A joint's move is "meaningful" above this travel (rad); below it, the joint
# was essentially holding position and its stats are noise, not signal.
MOVED_EPS_RAD = 0.05

RAD2DEG = 180.0 / np.pi


# --- per-move metrics ---------------------------------------------------------

def peak_overshoot_mrad(rec: Recording, seg) -> float:
    """Max |actual - dest| during the settle window after motion nominally stops.

    ``seg.i1`` is where the commanded speed returns to rest; the window
    ``i1:i2`` up to the next move is where a real robot's ringing shows up.
    """
    if seg.i2 <= seg.i1:
        return 0.0
    actual = rec.actual_q[seg.i1:seg.i2, seg.joint]
    return float(np.max(np.abs(actual - seg.dest))) * 1e3


def rms_pos_error_mrad(rec: Recording, seg) -> float:
    """RMS actual-minus-target position error over the full move (i0:i2)."""
    a = rec.actual_q[seg.i0:seg.i2, seg.joint]
    t = rec.target_q[seg.i0:seg.i2, seg.joint]
    return float(np.sqrt(np.mean((a - t) ** 2))) * 1e3


def collect_segment_stats(paths: list[str]) -> tuple[pd.DataFrame, dict]:
    """Segment every recording and return (stats DataFrame, {file: Recording})."""
    rows, recs = [], {}
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"loading {name} ...")
        rec = Recording(path)
        recs[name] = rec
        segs = segments(rec)
        print(f"  {len(segs)} segments, {len(rec.t)} rows")
        for seg in segs:
            if seg.vel is None or seg.acc is None:
                continue
            rows.append({
                "file": name,
                "joint": seg.joint,
                "joint_name": JOINT_NAMES[seg.joint],
                "vel": seg.vel,
                "acc": seg.acc,
                "dist_rad": seg.dist,
                "start_rad": seg.start,
                "dest_rad": seg.dest,
                "direction": 1 if seg.dest >= seg.start else -1,
                "rms_pos_err_mrad": rms_pos_error_mrad(rec, seg),
                "peak_overshoot_mrad": peak_overshoot_mrad(rec, seg),
                "duration_s": float(rec.t[seg.i1] - rec.t[seg.i0]),
                "i0": seg.i0, "i1": seg.i1, "i2": seg.i2,
            })
    return pd.DataFrame(rows), recs


def classify_sweep(df_file: pd.DataFrame) -> str:
    """How a file varies (vel, acc): 'vel_sweep', 'acc_sweep', 'combo', or 'fixed'."""
    n_vel, n_acc = df_file["vel"].nunique(), df_file["acc"].nunique()
    if n_vel > 1 and n_acc > 1:
        return "combo"
    if n_vel > 1:
        return "vel_sweep"
    if n_acc > 1:
        return "acc_sweep"
    return "fixed"


def dominant_joint(df_file: pd.DataFrame) -> int:
    """The joint that actually travels in this file (weighted by total distance)."""
    return int(df_file.groupby("joint")["dist_rad"].sum().idxmax())


# --- trajectory plots ----------------------------------------------------------

def plot_param_sweep_trajectories(rec, segs_sorted: list, param: str,
                                  out_path: str, file_name: str, joint_name: str,
                                  n_show: int = 6):
    """Target (dashed) vs actual (solid) position, several settings overlaid."""
    idx = np.unique(np.linspace(0, len(segs_sorted) - 1,
                                min(n_show, len(segs_sorted))).round().astype(int))
    vals = [getattr(s, param) for s in segs_sorted]
    cmap, norm = plt.get_cmap("viridis"), plt.Normalize(min(vals), max(vals))

    fig, ax = plt.subplots(figsize=(8, 5))
    for i in idx:
        seg = segs_sorted[i]
        t = rec.t[seg.i0:seg.i2] - rec.t[seg.i0]
        target = rec.target_q[seg.i0:seg.i2, seg.joint]
        actual = rec.actual_q[seg.i0:seg.i2, seg.joint]
        color = cmap(norm(getattr(seg, param)))
        ax.plot(t, target, "--", color=color, lw=1, alpha=0.6)
        ax.plot(t, actual, "-", color=color, lw=1.6,
                label=f"{param}={getattr(seg, param):.0f}")
    unit = "deg/s" if param == "vel" else "deg/s^2"
    ax.set_xlabel("time since move start (s)")
    ax.set_ylabel(f"{joint_name} position (rad)")
    ax.set_title(f"{file_name}: target (dashed) vs actual (solid) at different "
                f"{param} ({unit})")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_move_detail(rec, seg, out_path: str, title: str):
    """One move in detail: full trajectory for context, plus a zoomed panel of
    the settle window in mrad-from-destination, where the overshoot number
    actually becomes visible (it is invisible against a multi-radian move on
    a shared axis).
    """
    t = rec.t[seg.i0:seg.i2] - rec.t[seg.i0]
    target = rec.target_q[seg.i0:seg.i2, seg.joint]
    actual = rec.actual_q[seg.i0:seg.i2, seg.joint]
    t_stop = rec.t[seg.i1] - rec.t[seg.i0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.axvspan(t_stop, t[-1] if len(t) else t_stop, color="tab:red", alpha=0.08,
               label="settle window")
    ax1.plot(t, target, label="target", lw=2)
    ax1.plot(t, actual, label="actual", lw=1.2)
    ax1.axvline(t_stop, color="grey", ls="--", lw=0.8, label="nominal stop")
    ax1.set_xlabel("time since move start (s)")
    ax1.set_ylabel(f"{JOINT_NAMES[seg.joint]} position (rad)")
    ax1.set_title("full move")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    t_settle = t[t >= t_stop]
    dev_mrad = (actual[t >= t_stop] - seg.dest) * 1e3
    ax2.axhline(0, color="grey", lw=0.8)
    ax2.plot(t_settle, dev_mrad, color="tab:red", lw=1.4)
    ax2.set_xlabel("time since move start (s)")
    ax2.set_ylabel("actual - dest (mrad)")
    ax2.set_title("settle window, zoomed")
    ax2.grid(alpha=0.3)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --- summary plots ---------------------------------------------------------

def plot_metric_vs_param(df: pd.DataFrame, param: str, metric: str,
                         out_path: str, title: str, ylabel: str):
    """Scatter of a metric against vel or acc, one series per (joint, direction).

    Split by travel direction, not just joint: a movej that lifts a joint
    against gravity and the return move that lets gravity assist it can ring
    very differently even at the same vel/acc, so pooling both directions
    into one series would show two flat bands with no visible vel/acc trend.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for j in sorted(df["joint"].unique()):
        for d, marker in ((1, "^"), (-1, "v")):
            sub = df[(df["joint"] == j) & (df["direction"] == d) &
                    (df["dist_rad"] > MOVED_EPS_RAD)].sort_values(param)
            if sub.empty:
                continue
            ax.scatter(sub[param], sub[metric], s=14, alpha=0.6, marker=marker,
                      label=f"{JOINT_NAMES[j]} ({'+' if d > 0 else '-'})")
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    unit = "deg/s" if param == "vel" else "deg/s^2"
    ax.set_xlabel(f"{param} ({unit})")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heatmap(df_combo: pd.DataFrame, metric: str, out_path: str, title: str,
                 bins: int = 8):
    """Mean metric over a (vel, acc) grid, from the randomized-combo runs."""
    df_combo = df_combo[df_combo["dist_rad"] > MOVED_EPS_RAD]
    if df_combo.empty:
        return
    vel_edges = np.linspace(df_combo["vel"].min(), df_combo["vel"].max(), bins + 1)
    acc_edges = np.linspace(df_combo["acc"].min(), df_combo["acc"].max(), bins + 1)
    df_combo = df_combo.copy()
    df_combo["vel_bin"] = pd.cut(df_combo["vel"], vel_edges, include_lowest=True)
    df_combo["acc_bin"] = pd.cut(df_combo["acc"], acc_edges, include_lowest=True)
    pivot = df_combo.groupby(["acc_bin", "vel_bin"], observed=True)[metric].mean().unstack()
    pivot = pivot.reindex(index=pd.IntervalIndex.from_breaks(acc_edges),
                          columns=pd.IntervalIndex.from_breaks(vel_edges))

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="magma")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{iv.mid:.0f}" for iv in pivot.columns], rotation=45, fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{iv.mid:.0f}" for iv in pivot.index], fontsize=7)
    ax.set_xlabel("vel (deg/s)")
    ax.set_ylabel("acc (deg/s^2)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_direction_boxplot(df: pd.DataFrame, metric: str, out_path: str, title: str, ylabel: str):
    """Box plot of a metric split by (joint, travel direction), moved segments only.

    Isolates the direction effect (e.g. against vs. with gravity) from the
    vel/acc effect seen in ``plot_metric_vs_param``.
    """
    sub = df[df["dist_rad"] > MOVED_EPS_RAD].copy()
    sub["group"] = sub["joint_name"] + np.where(sub["direction"] > 0, " (+)", " (-)")
    groups = [g for g in sub["group"].unique() if sub[sub["group"] == g].shape[0] >= 5]
    if not groups:
        return
    groups = sorted(groups)
    data = [sub[sub["group"] == g][metric].to_numpy() for g in groups]

    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(groups)), 5))
    ax.boxplot(data, tick_labels=groups, showfliers=False)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def direction_effect(df: pd.DataFrame) -> list[dict]:
    """Mean overshoot/RMS error for each (joint, direction) with >=5 moved segments."""
    sub = df[df["dist_rad"] > MOVED_EPS_RAD]
    out = []
    for (j, d), g in sub.groupby(["joint_name", "direction"]):
        if len(g) < 5:
            continue
        out.append({
            "joint": j, "direction": "+" if d > 0 else "-", "n": int(len(g)),
            "mean_overshoot_mrad": float(g["peak_overshoot_mrad"].mean()),
            "mean_rms_pos_err_mrad": float(g["rms_pos_err_mrad"].mean()),
        })
    return sorted(out, key=lambda r: (r["joint"], r["direction"]))


def plot_per_joint_overshoot(df: pd.DataFrame, out_path: str):
    """Bar chart: mean peak overshoot per joint, moved segments only."""
    sub = df[df["dist_rad"] > MOVED_EPS_RAD]
    means = sub.groupby("joint_name")["peak_overshoot_mrad"].mean().reindex(JOINT_NAMES)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(means.index, means.values, color="darkorange", edgecolor="black")
    for bar, v in zip(bars, means.values):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("mean peak overshoot (mrad)")
    ax.set_title("Peak overshoot by joint (moved segments, all files)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --- presentation plots -------------------------------------------------------
# A curated subset (4 plots) of the analysis above, in degrees and % of move
# distance instead of mrad, with the value for each joint / (vel, acc) cell
# printed directly on the plot -- meant to go straight into a slide, and to be
# regenerated unchanged (just point --data-glob at the new recordings and
# --presentation-name at a new folder) once the optimized trajectories are
# recorded, so baseline and optimized are directly comparable.

def _add_deg_pct_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with degree and %-of-move-distance columns added.

    % is each row's error as a fraction of that row's own commanded travel
    (``dist_rad``) -- comparable across joints/moves of very different size,
    unlike the raw mrad number.
    """
    df = df.copy()
    df["overshoot_deg"] = df["peak_overshoot_mrad"] / 1e3 * RAD2DEG
    df["rms_deg"] = df["rms_pos_err_mrad"] / 1e3 * RAD2DEG
    df["overshoot_pct"] = df["peak_overshoot_mrad"] / 1e3 / df["dist_rad"] * 100
    df["rms_pct"] = df["rms_pos_err_mrad"] / 1e3 / df["dist_rad"] * 100
    return df


def plot_metric_by_joint(df: pd.DataFrame, deg_col: str, pct_col: str, out_path: str,
                         title: str):
    """Bar chart, one bar per joint, labeled with both degrees and % of move distance."""
    sub = df[df["dist_rad"] > MOVED_EPS_RAD]
    deg = sub.groupby("joint_name")[deg_col].mean().reindex(JOINT_NAMES).dropna()
    pct = sub.groupby("joint_name")[pct_col].mean().reindex(deg.index)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(deg.index, deg.values, color="darkorange", edgecolor="black")
    for bar, d, p in zip(bars, deg.values, pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, d, f"{d:.2f}°\n({p:.2f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("degrees")
    ax.set_title(title)
    ax.set_ylim(0, deg.max() * 1.25)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_duration_by_joint(df: pd.DataFrame, out_path: str):
    """Bar chart, mean move duration (motion-only window, i0:i1) per joint, in
    seconds -- the "did it actually get faster" companion to the overshoot/RMS
    by-joint plots. Vibration is trivial to reduce by slowing down; this is
    what catches that.
    """
    sub = df[df["dist_rad"] > MOVED_EPS_RAD]
    dur = sub.groupby("joint_name")["duration_s"].mean().reindex(JOINT_NAMES).dropna()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(dur.index, dur.values, color="steelblue", edgecolor="black")
    for bar, d in zip(bars, dur.values):
        ax.text(bar.get_x() + bar.get_width() / 2, d, f"{d:.2f}s",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("seconds")
    ax.set_title("Mean move duration by joint (motion-only window)")
    ax.set_ylim(0, dur.max() * 1.25)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heatmap_labeled(df_combo: pd.DataFrame, deg_col: str, pct_col: str, out_path: str,
                         title: str, bins: int = 4):
    """(vel, acc) grid, coarse enough that every cell's mean value (degrees, then
    % of move distance) can be printed inside it -- the same data as
    ``plot_heatmap`` but sized for reading off numbers rather than spotting a
    gradient by eye.

    Bins are quantile-based (equal segment count per bin), not equal-width:
    the combo runs mix a narrow 20-100 sweep (test-1) with a 10-990 randomized
    one (test-6/7), so equal-width bins leave most of the grid empty. Equal-
    count bins keep every cell populated.
    """
    df_combo = df_combo[df_combo["dist_rad"] > MOVED_EPS_RAD]
    if df_combo.empty:
        return
    df_combo = df_combo.copy()
    df_combo["vel_bin"] = pd.qcut(df_combo["vel"], bins, duplicates="drop")
    df_combo["acc_bin"] = pd.qcut(df_combo["acc"], bins, duplicates="drop")
    deg = df_combo.groupby(["acc_bin", "vel_bin"], observed=True)[deg_col].mean().unstack()
    pct = df_combo.groupby(["acc_bin", "vel_bin"], observed=True)[pct_col].mean().unstack()
    acc_index = deg.index.sort_values()
    vel_cols = deg.columns.sort_values()
    deg = deg.reindex(index=acc_index, columns=vel_cols)
    pct = pct.reindex(index=acc_index, columns=vel_cols)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(deg.values, origin="lower", aspect="auto", cmap="magma")
    vmin, vmax = np.nanmin(deg.values), np.nanmax(deg.values)
    for i in range(deg.shape[0]):
        for j in range(deg.shape[1]):
            d, p = deg.values[i, j], pct.values[i, j]
            if np.isnan(d):
                continue
            color = "black" if (d - vmin) / max(vmax - vmin, 1e-9) > 0.6 else "white"
            ax.text(j, i, f"{d:.2f}°\n{p:.2f}%", ha="center", va="center",
                    fontsize=7.5, color=color)
    ax.set_xticks(range(len(vel_cols)))
    ax.set_xticklabels([f"{iv.left:.0f}-{iv.right:.0f}" for iv in vel_cols], fontsize=7.5)
    ax.set_yticks(range(len(acc_index)))
    ax.set_yticklabels([f"{iv.left:.0f}-{iv.right:.0f}" for iv in acc_index], fontsize=7.5)
    ax.set_xlabel("vel (deg/s)")
    ax.set_ylabel("acc (deg/s^2)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="degrees")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_presentation_plots(df: pd.DataFrame, combo_files: list[str], out_dir: str,
                             folder_name: str):
    """Write the curated 4-plot presentation set to ``out_dir/folder_name/``."""
    pres_dir = os.path.join(out_dir, folder_name)
    os.makedirs(pres_dir, exist_ok=True)
    df = _add_deg_pct_cols(df)

    plot_metric_by_joint(df, "overshoot_deg", "overshoot_pct",
                         os.path.join(pres_dir, "overshoot_by_joint.png"),
                         "Peak overshoot by joint")
    plot_metric_by_joint(df, "rms_deg", "rms_pct",
                         os.path.join(pres_dir, "rms_by_joint.png"),
                         "RMS position error by joint")
    plot_duration_by_joint(df, os.path.join(pres_dir, "duration_by_joint.png"))

    combo = df[df["file"].isin(combo_files)]
    if not combo.empty:
        plot_heatmap_labeled(combo, "overshoot_deg", "overshoot_pct",
                             os.path.join(pres_dir, "overshoot_heatmap.png"),
                             "Peak overshoot over (vel, acc)")
        plot_heatmap_labeled(combo, "rms_deg", "rms_pct",
                             os.path.join(pres_dir, "rms_heatmap.png"),
                             "RMS position error over (vel, acc)")
    print(f"[results] presentation plots -> {pres_dir}")


def build_per_file_bar_plots(df: pd.DataFrame, out_dir: str):
    """Same bar chart as ``overshoot_by_joint.png``/``rms_by_joint.png``, run once
    per recording instead of once on the pooled ``df`` -- the un-averaged
    complement. Each file gets its own scale (a joint that barely moves in one
    file shouldn't be squashed by another file's larger bars).
    """
    run_dir = os.path.join(out_dir, "per_run")
    os.makedirs(run_dir, exist_ok=True)
    for file_name, df_file in df.groupby("file"):
        d = _add_deg_pct_cols(df_file)
        plot_metric_by_joint(d, "overshoot_deg", "overshoot_pct",
                             os.path.join(run_dir, f"{file_name}_overshoot_by_joint.png"),
                             f"{file_name}: peak overshoot by joint")
        plot_metric_by_joint(d, "rms_deg", "rms_pct",
                             os.path.join(run_dir, f"{file_name}_rms_by_joint.png"),
                             f"{file_name}: RMS position error by joint")
    print(f"[results] per-file bar charts -> {run_dir}")


# --- orchestration -----------------------------------------------------------

def build_trajectory_plots(df: pd.DataFrame, recs: dict, out_dir: str, n_show: int):
    traj_dir = os.path.join(out_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    for file_name, df_file in df.groupby("file"):
        sweep = classify_sweep(df_file)
        if sweep not in ("vel_sweep", "acc_sweep"):
            continue
        param = "vel" if sweep == "vel_sweep" else "acc"
        j = dominant_joint(df_file)
        rec = recs[file_name]
        segs = segments(rec)
        segs = sorted([s for s in segs if s.joint == j and s.vel is not None],
                      key=lambda s: getattr(s, param))
        if len(segs) < 2:
            continue
        out_path = os.path.join(traj_dir, f"{file_name}_{JOINT_NAMES[j]}_{param}_sweep.png")
        plot_param_sweep_trajectories(rec, segs, param, out_path, file_name,
                                      JOINT_NAMES[j], n_show=n_show)
        print(f"[results] plot -> {out_path}")

    # Single worst / best move overall (moved segments only), in detail.
    moved = df[df["dist_rad"] > MOVED_EPS_RAD]
    if moved.empty:
        return
    worst = moved.loc[moved["peak_overshoot_mrad"].idxmax()]
    best = moved.loc[moved["peak_overshoot_mrad"].idxmin()]
    for row, tag in [(worst, "worst"), (best, "best")]:
        rec = recs[row["file"]]
        seg = next(s for s in segments(rec)
                  if s.joint == row["joint"] and s.i0 == row["i0"] and s.i2 == row["i2"])
        title = (f"{tag}: {row['file']} {row['joint_name']} "
                f"vel={row['vel']:.0f} acc={row['acc']:.0f} "
                f"overshoot={row['peak_overshoot_mrad']:.2f} mrad")
        out_path = os.path.join(traj_dir, f"{tag}_move_{row['file']}_{row['joint_name']}.png")
        plot_move_detail(rec, seg, out_path, title)
        print(f"[results] plot -> {out_path}")


def _corr_within(df: pd.DataFrame, files: list[str], param: str, metric: str) -> float:
    """Pearson r restricted to one sweep's files, one (joint, direction) at a time,
    then averaged -- so a joint/direction mix at very different vel/acc ranges
    (e.g. a 10-100 deg/s sweep next to a 10-1000 deg/s random run) can't flip the
    sign the way a single pooled correlation over everything would.
    """
    sub = df[df["file"].isin(files) & (df["dist_rad"] > MOVED_EPS_RAD)]
    rs = []
    for _, g in sub.groupby(["joint_name", "direction"]):
        if len(g) < 5 or g[param].nunique() < 2:
            continue
        r = g[param].corr(g[metric])
        if not np.isnan(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def duration_range_by_file(df: pd.DataFrame) -> dict:
    """(min, max) motion-only samples per move (i1-i0), dominant joint, per file.

    Uses ``i1-i0`` (commanded speed nonzero, i.e. the move itself), not
    ``i2-i0``: the last movej of a script is bounded by the recording's end
    (see common.segments docstring), so its settle window balloons to
    whatever tail was recorded after the script halted -- an artifact, not a
    real duration change. i1-i0 is immune to that and is the honest check of
    whether the swept parameter actually changed move duration the way a
    trapezoidal-profile model predicts.
    """
    out = {}
    for f, g in df.groupby("file"):
        j = dominant_joint(g)
        sub = g[g["joint"] == j]
        n = (sub["i1"] - sub["i0"])
        out[f] = {"joint": JOINT_NAMES[j], "n_samples_min": int(n.min()),
                  "n_samples_max": int(n.max()), "ratio": round(float(n.max() / max(n.min(), 1)), 2)}
    return out


def log_summary(df: pd.DataFrame, out_dir: str, vel_sweep_files: list[str],
                acc_sweep_files: list[str]):
    moved = df[df["dist_rad"] > MOVED_EPS_RAD]
    by_file = (df.groupby("file")
               .apply(lambda g: classify_sweep(g), include_groups=False)
               .to_dict())

    def top(metric, n, ascending):
        cols = ["file", "joint_name", "vel", "acc", "direction", "dist_rad", metric]
        return (moved.sort_values(metric, ascending=ascending)
                .head(n)[cols].to_dict(orient="records"))

    log = {
        "n_files": int(df["file"].nunique()),
        "n_segments": int(len(df)),
        "n_segments_moved": int(len(moved)),
        "sweep_type_by_file": by_file,
        "worst_overshoot": top("peak_overshoot_mrad", 5, ascending=False),
        "best_overshoot": top("peak_overshoot_mrad", 5, ascending=True),
        "worst_rms_pos_err": top("rms_pos_err_mrad", 5, ascending=False),
        "best_rms_pos_err": top("rms_pos_err_mrad", 5, ascending=True),
        "direction_effect": direction_effect(df),
        "duration_range_by_file": duration_range_by_file(df),
        "note": "correlations below are computed within each (joint, direction) "
                "group in the single-parameter sweep files, then averaged -- "
                "pooling every file/joint/direction into one Pearson r mixed "
                "different speed regimes and flipped the sign. duration_range_by_file "
                "uses the motion-only window (i1-i0); it shows the swept vel/acc register "
                "(20-100 for vel, 11-100 for acc) barely changes how long the dominant "
                "joint's move actually takes (ratio ~1.0-1.35) -- so over this range the "
                "controller is not simply running a trapezoidal profile capped at the "
                "commanded vel/acc the way dynamics.py's model assumes; something else "
                "(a lower default speed limit, a blend/safety cap) is binding instead. "
                "Worth flagging for the Dynamics swap point.",
        "corr_vel_overshoot": _corr_within(df, vel_sweep_files, "vel", "peak_overshoot_mrad"),
        "corr_acc_overshoot": _corr_within(df, acc_sweep_files, "acc", "peak_overshoot_mrad"),
        "corr_vel_rms": _corr_within(df, vel_sweep_files, "vel", "rms_pos_err_mrad"),
        "corr_acc_rms": _corr_within(df, acc_sweep_files, "acc", "rms_pos_err_mrad"),
    }
    log_path = os.path.join(out_dir, "log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[results] log  -> {log_path}")
    return log


def main():
    ap = argparse.ArgumentParser(description="Bronze-tier RTDE dataset exploration.")
    ap.add_argument("--data-glob", default=DATA_GLOB, help="glob of recorded CSVs")
    ap.add_argument("--out", default=OUT_DIR, help="output directory")
    ap.add_argument("--n-traj", type=int, default=6,
                    help="settings overlaid per trajectory sweep plot")
    ap.add_argument("--presentation-name", default="presentation_plots_baseline",
                    help="subfolder (under --out) for the curated 4-plot presentation "
                         "set; pass e.g. presentation_plots_optimized when re-running "
                         "against the optimized-trajectory recordings")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.data_glob))
    if not paths:
        raise SystemExit(f"no CSVs matched {args.data_glob}")
    os.makedirs(args.out, exist_ok=True)

    df, recs = collect_segment_stats(paths)
    csv_path = os.path.join(args.out, "segment_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"[results] csv  -> {csv_path}  ({len(df)} segments)")

    build_trajectory_plots(df, recs, args.out, args.n_traj)

    vel_sweep_files = [f for f, g in df.groupby("file") if classify_sweep(g) == "vel_sweep"]
    acc_sweep_files = [f for f, g in df.groupby("file") if classify_sweep(g) == "acc_sweep"]
    combo_files = [f for f, g in df.groupby("file") if classify_sweep(g) == "combo"]

    if vel_sweep_files:
        sub = df[df["file"].isin(vel_sweep_files)]
        plot_metric_vs_param(sub, "vel", "peak_overshoot_mrad",
                             os.path.join(args.out, "overshoot_vs_vel.png"),
                             "Peak overshoot vs vel (acc held fixed)", "peak overshoot (mrad)")
        plot_metric_vs_param(sub, "vel", "rms_pos_err_mrad",
                             os.path.join(args.out, "rms_vs_vel.png"),
                             "RMS position error vs vel (acc held fixed)", "RMS pos error (mrad)")
    if acc_sweep_files:
        sub = df[df["file"].isin(acc_sweep_files)]
        plot_metric_vs_param(sub, "acc", "peak_overshoot_mrad",
                             os.path.join(args.out, "overshoot_vs_acc.png"),
                             "Peak overshoot vs acc (vel held fixed)", "peak overshoot (mrad)")
        plot_metric_vs_param(sub, "acc", "rms_pos_err_mrad",
                             os.path.join(args.out, "rms_vs_acc.png"),
                             "RMS position error vs acc (vel held fixed)", "RMS pos error (mrad)")
    if combo_files:
        sub = df[df["file"].isin(combo_files)]
        plot_heatmap(sub, "peak_overshoot_mrad", os.path.join(args.out, "overshoot_heatmap.png"),
                    "Mean peak overshoot over (vel, acc)")
        plot_heatmap(sub, "rms_pos_err_mrad", os.path.join(args.out, "rms_heatmap.png"),
                    "Mean RMS position error over (vel, acc)")

    plot_per_joint_overshoot(df, os.path.join(args.out, "per_joint_overshoot.png"))
    plot_direction_boxplot(df, "peak_overshoot_mrad",
                           os.path.join(args.out, "overshoot_by_direction.png"),
                           "Peak overshoot by joint and travel direction", "peak overshoot (mrad)")
    plot_direction_boxplot(df, "rms_pos_err_mrad",
                           os.path.join(args.out, "rms_by_direction.png"),
                           "RMS position error by joint and travel direction", "RMS pos error (mrad)")

    build_presentation_plots(df, combo_files, args.out, args.presentation_name)
    build_per_file_bar_plots(df, args.out)

    log = log_summary(df, args.out, vel_sweep_files, acc_sweep_files)
    print(f"[results] run complete -> {args.out}")
    print(f"  worst overshoot: {log['worst_overshoot'][0]}")
    print(f"  best  overshoot: {log['best_overshoot'][0]}")
    print(f"  corr(vel, overshoot)={log['corr_vel_overshoot']:.3f}  "
         f"corr(acc, overshoot)={log['corr_acc_overshoot']:.3f}")


if __name__ == "__main__":
    main()
