"""Bar charts: reality gap (|actual - target|) by channel, plus drill-downs.

Three related outputs, all under ``bronze_tier/channel_gap/``:

  full_run/             peak + RMS gap per channel, whole recordings
  settling_window/      same, but rows restricted to each move's settle
                         window (``common.segments``' ``i1:i2`` -- the
                         post-move ringing window, not the motion itself)
  worst_channels_by_component/
                         peak + RMS gap per joint/axis, for the 3 channels
                         with the worst full-run peak %FS -- the other two
                         folders say *which channel*; this says *where in
                         it*
  worst_files/           full per-channel breakdown (same chart as full_run/,
                         same %FS denominators) for the N individual
                         recordings with the worst full-run peak %FS on their
                         own worst channel -- full_run/ pools every file
                         together, which dilutes RMS (and can bury which
                         *file* is actually the problem) once you have many
                         calm recordings alongside a few bad ones; this
                         folder answers "which recordings actually show the
                         gap" instead of only the all-files-pooled average.

Each channel's bar height is the gap as a percentage of that channel's own
full-scale range (pooled target+actual, over the *whole run*, computed once
so full-run and settling-window numbers stay directly comparable -- using a
settle-window-only range as the denominator would shrink it and inflate the
settle-window percentages for no real reason). Channels sit on different
units/scales (A, mm, deg, m/s) so that percentage is the only axis every
channel/component shares. The label above each bar gives the raw value in
its native unit (degrees for every angle-valued channel) followed by that
percentage again.

    python channel_gap_bar_chart.py
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

import ur_style
from analysis import Recording
from common import segments
from utils import JOINT_NAMES

ur_style.apply()

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_GLOB = os.path.join(HERE, "data", "*.csv")
OUT_DIR = os.path.join(HERE, "bronze_tier", "channel_gap")

RAD2DEG = 180.0 / np.pi
TCP_AXIS_NAMES = ("x", "y", "z", "rx", "ry", "rz")

# display name -> (target col base, actual col base, component indices,
# factor raw(rad | rad/s | m) -> display unit, display unit, component name
# lookup indexable by the raw column index 0..5).
#
# "TCP orientation" is handled separately (orientation_gap_stats, further
# down): UR's TCP_pose3..5 is a rotation *vector* (axis * angle), a 2-to-1
# representation -- a rotation by angle pi about axis n is the same physical
# orientation as angle pi about -n, so the logged vector can flip sign
# between adjacent samples with no real motion. Naive per-component
# subtraction picks up a spurious ~360 degree spike whenever target and
# actual cross that flip a sample apart (observed in test-6 row 138273, true
# gap there is 0.085 degrees); the geodesic (quaternion) distance is immune
# to it. It's also why "TCP orientation" is deliberately left out of this
# table -- the worst-3-by-component breakdown only knows how to break down
# channels in here, and a per-axis geodesic split doesn't mean anything.
CHANNELS = [
    ("q (joint position)",  "target_q",          "actual_q",          range(6),    RAD2DEG, "deg",   JOINT_NAMES),
    ("qd (joint velocity)", "target_qd",          "actual_qd",         range(6),    RAD2DEG, "deg/s", JOINT_NAMES),
    ("current",              "target_current",     "actual_current",   range(6),    1.0,     "A",     JOINT_NAMES),
    ("TCP position",         "target_TCP_pose",    "actual_TCP_pose",  range(0, 3), 1000.0,  "mm",    TCP_AXIS_NAMES),
    ("TCP speed (linear)",   "target_TCP_speed",   "actual_TCP_speed", range(0, 3), 1.0,      "m/s",   TCP_AXIS_NAMES),
    ("TCP speed (angular)",  "target_TCP_speed",   "actual_TCP_speed", range(3, 6), RAD2DEG,  "deg/s", TCP_AXIS_NAMES),
]


# --- data loading --------------------------------------------------------

def load_recordings(paths: list[str]) -> dict:
    recs = {}
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"[load] {name} ...")
        recs[name] = Recording(path)
    return recs


def settle_masks(recs: dict) -> dict:
    """Per-file boolean row mask: True during any move's settle window
    (post-motion ringing, ``seg.i1:seg.i2``), False during the motion
    itself, dwells, and anything outside a recognized move.
    """
    masks = {}
    for name, rec in recs.items():
        mask = np.zeros(len(rec.t), dtype=bool)
        for seg in segments(rec):
            mask[seg.i1:seg.i2] = True
        masks[name] = mask
    return masks


# --- gap stats -------------------------------------------------------------

def channel_gap_stats(recs: dict, target_base: str, actual_base: str, cols,
                       masks: dict | None = None, fs_range_raw: float | None = None) -> dict:
    """Pool one channel's target/actual columns over every file (optionally
    restricted to each file's settle-window rows).

    Returns the largest |actual - target| anywhere in the pooled data (with
    the file/row/component it came from), its RMS, and the percentage each
    represents of ``fs_range_raw`` -- the channel's full-scale range, passed
    in so a settle-window call can reuse the full-run figure instead of
    computing its own (much narrower, and not comparable) range. If not
    given, it's computed from the rows actually pooled here.
    """
    best_gap, best_where = -1.0, None
    lo, hi = np.inf, -np.inf
    sq_sum, n = 0.0, 0
    for name, rec in recs.items():
        idx_all = np.nonzero(masks[name])[0] if masks is not None else None
        for c in cols:
            t = rec.df[f"{target_base}{c}"].to_numpy(dtype=float)
            a = rec.df[f"{actual_base}{c}"].to_numpy(dtype=float)
            if idx_all is not None:
                t, a, idx = t[idx_all], a[idx_all], idx_all
            else:
                idx = np.arange(len(t))
            if len(t) == 0:
                continue
            gap = np.abs(a - t)
            i = int(np.argmax(gap))
            if gap[i] > best_gap:
                best_gap, best_where = float(gap[i]), (name, int(idx[i]), c)
            lo, hi = min(lo, t.min(), a.min()), max(hi, t.max(), a.max())
            sq_sum += float(np.sum(gap ** 2))
            n += len(gap)
    local_fs_range = float(hi - lo) if np.isfinite(hi) else 0.0
    denom = fs_range_raw if fs_range_raw is not None else local_fs_range
    pct = (lambda v: v / denom * 100) if denom > 0 else (lambda v: float("nan"))
    rms = float(np.sqrt(sq_sum / n)) if n else float("nan")
    return {
        "max_gap_raw": best_gap,
        "rms_gap_raw": rms,
        "full_scale_range_raw": denom,
        "pct_max_of_full_scale": pct(best_gap),
        "pct_rms_of_full_scale": pct(rms),
        "source_file": best_where[0] if best_where else None,
        "row_index": best_where[1] if best_where else None,
        "component_index": best_where[2] if best_where else None,
        "n_samples": n,
    }


def _axis_angle_to_quat(v: np.ndarray) -> np.ndarray:
    """Rotation vectors ``(n, 3)`` (axis * angle, radians) to quaternions ``(n, 4)`` (w, x, y, z)."""
    theta = np.linalg.norm(v, axis=1)
    safe_theta = np.where(theta < 1e-12, 1.0, theta)
    axis = v / safe_theta[:, None]
    half = theta / 2.0
    return np.column_stack([np.cos(half), axis * np.sin(half)[:, None]])


def orientation_gap_stats(recs: dict, masks: dict | None = None) -> dict:
    """Max/RMS geodesic (true SO(3)) angle between target and actual TCP
    orientation, pooled over every file -- see the CHANNELS docstring for
    why this can't be a per-axis subtraction like the other channels.
    Percentage is of the largest angle two orientations can ever differ by
    (180 degrees), a fixed bound rather than a pooled data range -- it
    doesn't vary by recording or by scope, so full-run and settle-window
    numbers are already on the same footing with no override needed.
    """
    best_gap, best_where = -1.0, None
    sq_sum, n = 0.0, 0
    for name, rec in recs.items():
        t = rec.df[[f"target_TCP_pose{c}" for c in range(3, 6)]].to_numpy(dtype=float)
        a = rec.df[[f"actual_TCP_pose{c}" for c in range(3, 6)]].to_numpy(dtype=float)
        if masks is not None:
            idx = np.nonzero(masks[name])[0]
            t, a = t[idx], a[idx]
        else:
            idx = np.arange(len(t))
        if len(t) == 0:
            continue
        qt, qa = _axis_angle_to_quat(t), _axis_angle_to_quat(a)
        dot = np.clip(np.abs(np.sum(qt * qa, axis=1)), -1.0, 1.0)
        gap = 2.0 * np.arccos(dot)  # radians, in [0, pi]
        i = int(np.argmax(gap))
        if gap[i] > best_gap:
            best_gap, best_where = float(gap[i]), (name, int(idx[i]))
        sq_sum += float(np.sum(gap ** 2))
        n += len(gap)
    rms = float(np.sqrt(sq_sum / n)) if n else float("nan")
    return {
        "max_gap_raw": best_gap,
        "rms_gap_raw": rms,
        "full_scale_range_raw": np.pi,
        "pct_max_of_full_scale": best_gap / np.pi * 100,
        "pct_rms_of_full_scale": rms / np.pi * 100,
        "source_file": best_where[0] if best_where else None,
        "row_index": best_where[1] if best_where else None,
        "component_index": None,
        "n_samples": n,
    }


def collect_channel_rows(recs: dict, masks: dict | None = None,
                          fs_ranges: dict | None = None) -> list[dict]:
    """One row per channel (the 6 in CHANNELS + TCP orientation)."""
    rows = []
    for name, target_base, actual_base, cols, factor, unit, _comp_names in CHANNELS:
        fs = fs_ranges[name] if fs_ranges else None
        stats = channel_gap_stats(recs, target_base, actual_base, cols, masks, fs_range_raw=fs)
        rows.append({
            "name": name,
            "unit": unit,
            "max_gap_display": stats["max_gap_raw"] * factor,
            "rms_gap_display": stats["rms_gap_raw"] * factor,
            **stats,
        })

    stats = orientation_gap_stats(recs, masks)
    rows.append({
        "name": "TCP orientation",
        "unit": "deg",
        "max_gap_display": stats["max_gap_raw"] * RAD2DEG,
        "rms_gap_display": stats["rms_gap_raw"] * RAD2DEG,
        **stats,
    })
    return rows


def component_breakdown_rows(recs: dict, target_base: str, actual_base: str, cols,
                              comp_names, factor: float, unit: str,
                              fs_range_raw: float) -> list[dict]:
    """Per-joint/axis peak+RMS gap within one channel, as a % of that
    channel's own (full-run) full-scale range -- same denominator for every
    component so they stay comparable to each other and to the channel-level
    bar this drills into.
    """
    rows = []
    for c in cols:
        s = channel_gap_stats(recs, target_base, actual_base, [c], masks=None, fs_range_raw=fs_range_raw)
        rows.append({
            "component": comp_names[c],
            "unit": unit,
            "max_gap_display": s["max_gap_raw"] * factor,
            "rms_gap_display": s["rms_gap_raw"] * factor,
            "pct_max_of_full_scale": s["pct_max_of_full_scale"],
            "pct_rms_of_full_scale": s["pct_rms_of_full_scale"],
            "source_file": s["source_file"],
            "row_index": s["row_index"],
        })
    return rows


def rank_worst_files(recs: dict, fs_ranges: dict) -> list[tuple[str, float, list[dict]]]:
    """Per-file channel-gap rows, ranked worst-first by each file's own worst
    peak %FS across channels.

    ``fs_ranges`` (the full-run, all-files %FS denominators from
    ``collect_channel_rows``) is passed through to every per-file call so
    each file's percentage is on the *same* scale as the pooled chart and as
    every other file -- computing it per-file instead would use that file's
    own (much narrower) target/actual range as the denominator, making the
    percentages incomparable across files.

    Ranked by peak, not RMS: ``channel_gap_stats`` computes peak as the
    single worst sample across whatever's pooled into one call, so a
    per-file peak is already an honest "how bad does this file get," not
    diluted the way pooling every file's RMS together is.
    """
    ranked = []
    for name, rec in recs.items():
        rows = collect_channel_rows({name: rec}, fs_ranges=fs_ranges)
        worst_pct = max((r["pct_max_of_full_scale"] for r in rows
                         if np.isfinite(r["pct_max_of_full_scale"])), default=0.0)
        ranked.append((name, worst_pct, rows))
    return sorted(ranked, key=lambda t: t[1], reverse=True)


# --- plotting ----------------------------------------------------------------

def _grouped_bar(ax, labels: list[str], peak_pct: list[float], rms_pct: list[float]):
    """Shared grouped-bar body (peak + RMS series) used by both plot functions.

    Log y-axis: peak/RMS %FS span several orders of magnitude across
    channels (22% down to 0.004% here), so a linear axis makes the small
    channels invisible and crushes their peak/RMS labels into each other.
    Same fix the project's earlier target_selection_analysis.py used for
    this same cross-channel %FS comparison (see CLAUDE.md's change log).
    """
    x = np.arange(len(labels))
    w = 0.34
    bars_peak = ax.bar(x - w / 2, peak_pct, width=w, color=ur_style.BLUE, label="peak")
    bars_rms = ax.bar(x + w / 2, rms_pct, width=w, color=ur_style.MID_BLUE, label="RMS")
    ax.set_yscale("log")
    all_pct = [v for v in peak_pct + rms_pct if v > 0]
    floor = min(all_pct) * 0.5 if all_pct else 1e-3
    ax.set_ylim(floor, max(peak_pct) * 4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.3, axis="y", which="major")
    return bars_peak, bars_rms


def plot_channel_gap_bar(rows: list[dict], out_path: str, title: str):
    """One channel per x position, peak + RMS bars, height = % of the
    channel's full-scale range. Label above each bar gives the raw value in
    its native unit, then that percentage again.
    """
    rows = sorted(rows, key=lambda r: r["pct_max_of_full_scale"], reverse=True)
    names = [r["name"] for r in rows]
    peak_pct = [r["pct_max_of_full_scale"] for r in rows]
    rms_pct = [r["pct_rms_of_full_scale"] for r in rows]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    bars_peak, bars_rms = _grouped_bar(ax, names, peak_pct, rms_pct)
    for bar, r in zip(bars_peak, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{r['max_gap_display']:.3g} {r['unit']}\n({r['pct_max_of_full_scale']:.2f}%)",
                ha="center", va="bottom", fontsize=7.3, color=ur_style.NAVY)
    for bar, r in zip(bars_rms, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{r['rms_gap_display']:.3g} {r['unit']}\n({r['pct_rms_of_full_scale']:.2f}%)",
                ha="center", va="bottom", fontsize=7.3, color=ur_style.NAVY)

    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("reality gap (% of channel's full-scale range, log scale)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[results] plot -> {out_path}")


def plot_component_breakdown(rows: list[dict], out_path: str, title: str):
    """Same grouped peak+RMS bars as ``plot_channel_gap_bar``, one bar pair
    per joint/axis within a single channel instead of one per channel."""
    rows = sorted(rows, key=lambda r: r["pct_max_of_full_scale"], reverse=True)
    labels = [r["component"] for r in rows]
    peak_pct = [r["pct_max_of_full_scale"] for r in rows]
    rms_pct = [r["pct_rms_of_full_scale"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars_peak, bars_rms = _grouped_bar(ax, labels, peak_pct, rms_pct)
    for bar, r in zip(bars_peak, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{r['max_gap_display']:.3g} {r['unit']}\n({r['pct_max_of_full_scale']:.2f}%)",
                ha="center", va="bottom", fontsize=7.5)
    for bar, r in zip(bars_rms, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{r['rms_gap_display']:.3g} {r['unit']}\n({r['pct_rms_of_full_scale']:.2f}%)",
                ha="center", va="bottom", fontsize=7.5)

    ax.set_ylabel("reality gap (% of the channel's full-scale range, log scale)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[results] plot -> {out_path}")


# --- orchestration -----------------------------------------------------------

def _slug(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "")


def main():
    ap = argparse.ArgumentParser(
        description="Reality-gap bar charts by channel: full-run, settling-window, "
                    "and worst-3-by-component drill-down.")
    ap.add_argument("--data-glob", default=DATA_GLOB, help="glob of recorded CSVs")
    ap.add_argument("--out", default=OUT_DIR, help="output directory (subfolders created under it)")
    ap.add_argument("--n-worst-files", type=int, default=3,
                    help="how many individual recordings to break out in worst_files/ "
                        "(default: %(default)s)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.data_glob))
    if not paths:
        raise SystemExit(f"no CSVs matched {args.data_glob}")

    recs = load_recordings(paths)
    masks = settle_masks(recs)

    # --- full run -----------------------------------------------------------
    full_dir = os.path.join(args.out, "full_run")
    os.makedirs(full_dir, exist_ok=True)
    full_rows = collect_channel_rows(recs)
    plot_channel_gap_bar(full_rows, os.path.join(full_dir, "channel_gap_bar_chart.png"),
                        "Peak & RMS reality gap by channel (whole run)")
    with open(os.path.join(full_dir, "channel_gap_summary.json"), "w") as f:
        json.dump(sorted(full_rows, key=lambda r: r["pct_max_of_full_scale"], reverse=True), f, indent=2)
    print(f"[results] log  -> {full_dir}/channel_gap_summary.json")

    # --- settling window only, reusing the full-run %FS denominator --------
    fs_ranges = {r["name"]: r["full_scale_range_raw"] for r in full_rows}
    settle_dir = os.path.join(args.out, "settling_window")
    os.makedirs(settle_dir, exist_ok=True)
    settle_rows = collect_channel_rows(recs, masks, fs_ranges=fs_ranges)
    plot_channel_gap_bar(settle_rows, os.path.join(settle_dir, "channel_gap_bar_chart.png"),
                        "Peak & RMS reality gap by channel (settle window only)")
    with open(os.path.join(settle_dir, "channel_gap_summary.json"), "w") as f:
        json.dump(sorted(settle_rows, key=lambda r: r["pct_max_of_full_scale"], reverse=True), f, indent=2)
    print(f"[results] log  -> {settle_dir}/channel_gap_summary.json")

    # --- worst N individual files (own worst-channel peak %FS) -------------
    # full_run/ pools every recording together, which both dilutes RMS and
    # can hide which *file* actually has the gap once many calm recordings
    # sit alongside a few bad ones -- this breaks out the worst files on
    # their own, at the same %FS scale as every other chart (fs_ranges).
    files_dir = os.path.join(args.out, "worst_files")
    os.makedirs(files_dir, exist_ok=True)
    files_ranked = rank_worst_files(recs, fs_ranges)[:args.n_worst_files]

    worst_files_summary = {}
    for name, worst_pct, rows in files_ranked:
        out_path = os.path.join(files_dir, f"{_slug(name)}_channel_gap_bar_chart.png")
        plot_channel_gap_bar(rows, out_path,
                            f"{name}: peak & RMS reality gap by channel "
                            f"(worst file, {worst_pct:.2f}% peak)")
        worst_files_summary[name] = sorted(rows, key=lambda r: r["pct_max_of_full_scale"], reverse=True)
    with open(os.path.join(files_dir, "worst_files_summary.json"), "w") as f:
        json.dump(worst_files_summary, f, indent=2)
    print(f"[results] log  -> {files_dir}/worst_files_summary.json")
    print(f"[results] worst {args.n_worst_files} files (own worst-channel peak %FS): "
         f"{[(n, round(p, 2)) for n, p, _ in files_ranked]}")

    # --- worst 3 channels (full-run peak %FS), broken down by component ----
    comp_dir = os.path.join(args.out, "worst_channels_by_component")
    os.makedirs(comp_dir, exist_ok=True)
    lookup = {c[0]: c for c in CHANNELS}   # excludes "TCP orientation" on purpose, see CHANNELS docstring
    ranked = sorted(full_rows, key=lambda r: r["pct_max_of_full_scale"], reverse=True)
    worst3 = [r for r in ranked if r["name"] in lookup][:3]

    breakdown_summary = {}
    for r in worst3:
        name, target_base, actual_base, cols, factor, unit, comp_names = lookup[r["name"]]
        comp_rows = component_breakdown_rows(recs, target_base, actual_base, cols, comp_names,
                                              factor, unit, r["full_scale_range_raw"])
        out_path = os.path.join(comp_dir, f"{_slug(name)}_by_component.png")
        plot_component_breakdown(comp_rows, out_path, f"{name}: gap by component (whole run)")
        breakdown_summary[name] = comp_rows
    with open(os.path.join(comp_dir, "worst_channels_summary.json"), "w") as f:
        json.dump(breakdown_summary, f, indent=2)
    print(f"[results] log  -> {comp_dir}/worst_channels_summary.json")

    # --- drop the flat single-file outputs the previous version of this
    # script wrote directly under bronze_tier/, now superseded by the
    # subfolders above.
    for stale in ("channel_gap_bar_chart.png", "channel_gap_summary.json"):
        stale_path = os.path.join(HERE, "bronze_tier", stale)
        if os.path.exists(stale_path):
            os.remove(stale_path)
            print(f"[cleanup] removed superseded {stale_path}")

    print(f"[results] worst 3 channels (full-run peak %FS): {[r['name'] for r in worst3]}")


if __name__ == "__main__":
    main()
