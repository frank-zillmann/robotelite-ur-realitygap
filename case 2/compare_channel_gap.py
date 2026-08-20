"""Compare baseline vs. optimized reality-gap results, channel by channel.

Takes two sets of recorded CSVs -- baseline and optimized, each anything
``channel_gap_bar_chart.py --data-glob`` itself would accept: individual
file paths, a directory (globbed for ``*.csv``), or a glob pattern -- computes
each run's own peak/RMS-per-channel gap (reusing
``channel_gap_bar_chart.py``'s ``load_recordings``/``collect_channel_rows``
directly, not a reimplementation), and produces two grouped-bar charts: one
for peak %FS, one for RMS %FS, each with a baseline bar and an optimized bar
side by side per channel.

Both runs' percentages share the *baseline's* full-scale range per channel
(computed once, reused for optimized) -- the same "shared denominator" fix
``channel_gap_bar_chart.py`` uses to keep its own full-run/settling-window
numbers comparable. Without it, an optimized trajectory that happens to
cover a different range of motion would shift %FS for reasons having
nothing to do with the gap actually shrinking.

    python compare_channel_gap.py --baseline data/*.csv --optimized data_optimized/*.csv
    python compare_channel_gap.py --baseline data --optimized data_optimized

Outputs -> bronze_tier/channel_gap/comparison/ (or --out):
    peak_gap_comparison.png            grouped bars, baseline vs optimized peak %FS
    rms_gap_comparison.png             same, RMS %FS
    comparison_summary.json            per-channel baseline/optimized values and
                                       % change, for a table slide
    baseline_channel_gap_summary.json  each run's own full per-channel
    optimized_channel_gap_summary.json breakdown, same shape
                                       channel_gap_bar_chart.py writes
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

import ur_style
from channel_gap_bar_chart import collect_channel_rows, load_recordings

ur_style.apply()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "bronze_tier", "channel_gap", "comparison")


def _resolve_csvs(patterns: list[str]) -> list[str]:
    """``--baseline``/``--optimized`` arguments -> a sorted list of CSV paths.

    Each argument can be a literal file path, a directory (globbed for
    ``*.csv`` inside it), or a glob pattern -- whichever's most convenient
    to pass on the command line, matching ``channel_gap_bar_chart.py --data-glob``'s
    own flexibility.
    """
    paths = []
    for p in patterns:
        if os.path.isdir(p):
            paths.extend(glob.glob(os.path.join(p, "*.csv")))
        elif any(ch in p for ch in "*?["):
            paths.extend(glob.glob(p))
        else:
            paths.append(p)
    return sorted(set(paths))


def _rows_by_channel(csvs: list[str], fs_ranges: dict | None = None) -> tuple[dict, list[dict]]:
    """Load ``csvs`` and compute this run's per-channel gap rows.

    Returns ``({name: row}, rows)`` -- the dict for the comparison logic
    below, the list (JSON-shaped like ``channel_gap_bar_chart.py``'s own
    ``channel_gap_summary.json``) for the per-run summary file.
    """
    recs = load_recordings(csvs)
    rows = collect_channel_rows(recs, fs_ranges=fs_ranges)
    return {r["name"]: r for r in rows}, rows


def _grouped_compare_bar(ax, labels: list[str], baseline_vals: list[float],
                         optimized_vals: list[float], baseline_label: str, optimized_label: str):
    """Shared grouped-bar body: one baseline + one optimized bar per label.

    Log y-axis for the same reason ``channel_gap_bar_chart.py``'s own
    peak/RMS grouped bar uses one -- %FS spans several orders of magnitude
    across channels.
    """
    x = np.arange(len(labels))
    w = 0.34
    bars_base = ax.bar(x - w / 2, baseline_vals, width=w, color=ur_style.BLUE, label=baseline_label)
    bars_opt = ax.bar(x + w / 2, optimized_vals, width=w, color=ur_style.MID_BLUE, label=optimized_label)
    ax.set_yscale("log")
    all_vals = [v for v in baseline_vals + optimized_vals if v > 0]
    floor = min(all_vals) * 0.5 if all_vals else 1e-3
    ceiling = max(all_vals) * 4 if all_vals else 1.0
    ax.set_ylim(floor, ceiling)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.3, axis="y", which="major")
    return bars_base, bars_opt


def _annotate_pct_change(ax, bars_base, bars_opt, baseline_vals: list[float], optimized_vals: list[float]):
    """Percent change (optimized vs. baseline), centered above each bar
    pair -- above whichever of the two bars is taller, well clear of their
    own raw-value labels (placing it directly on the optimized bar collided
    with that bar's own label). Multiplicative offset since the axis is log
    scale, not linear.

    Bold for an improvement (lower gap, the expected/hoped-for direction),
    normal weight for a regression -- so a slide reader doesn't have to do
    the arithmetic or guess which sign is good.
    """
    for bar_b, bar_o, b, o in zip(bars_base, bars_opt, baseline_vals, optimized_vals):
        if b <= 0:
            continue
        pct_change = (o - b) / b * 100
        x = (bar_b.get_x() + bar_b.get_width() / 2 + bar_o.get_x() + bar_o.get_width() / 2) / 2
        y = max(bar_b.get_height(), bar_o.get_height()) * 1.6
        ax.text(x, y, f"{pct_change:+.0f}%", ha="center", va="bottom", fontsize=8.5,
                fontweight="bold" if pct_change <= 0 else "normal", color=ur_style.NAVY)


def plot_metric_comparison(baseline: dict, optimized: dict, pct_key: str, raw_key: str,
                           out_path: str, title: str, baseline_label: str, optimized_label: str) -> list[dict]:
    """One grouped-bar chart (one metric, e.g. peak %FS) for every channel
    present in both files. Channels present in only one are skipped (with a
    warning) rather than silently plotted as zero -- a missing channel
    usually means the two runs logged different data, not that the gap is
    actually zero.

    Returns the per-channel comparison rows (for the summary JSON).
    """
    names = sorted(set(baseline) & set(optimized), key=lambda n: baseline[n][pct_key], reverse=True)
    missing = (set(baseline) | set(optimized)) - set(names)
    if missing:
        print(f"[warn] channel(s) only in one file, skipped: {sorted(missing)}")

    baseline_vals = [baseline[n][pct_key] for n in names]
    optimized_vals = [optimized[n][pct_key] for n in names]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    bars_base, bars_opt = _grouped_compare_bar(ax, names, baseline_vals, optimized_vals,
                                               baseline_label, optimized_label)
    for bar, n in zip(bars_base, names):
        r = baseline[n]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{r[raw_key]:.3g} {r['unit']}", ha="center", va="bottom",
                fontsize=7, color=ur_style.NAVY)
    for bar, n in zip(bars_opt, names):
        r = optimized[n]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{r[raw_key]:.3g} {r['unit']}", ha="center", va="bottom",
                fontsize=7, color=ur_style.NAVY)
    _annotate_pct_change(ax, bars_base, bars_opt, baseline_vals, optimized_vals)

    ax.set_ylabel("reality gap (% of channel's full-scale range, log scale)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[results] plot -> {out_path}")

    return [{"channel": n, "baseline_pct": baseline[n][pct_key], "optimized_pct": optimized[n][pct_key],
            "pct_change": (optimized[n][pct_key] - baseline[n][pct_key]) / baseline[n][pct_key] * 100
            if baseline[n][pct_key] > 0 else None}
           for n in names]


def main():
    ap = argparse.ArgumentParser(
        description="Compare baseline vs. optimized recordings (peak + RMS reality gap %FS per channel).")
    ap.add_argument("--baseline", nargs="+", required=True,
                    help="baseline recordings: CSV file(s), a directory of CSVs, or a glob pattern")
    ap.add_argument("--optimized", nargs="+", required=True,
                    help="optimized recordings: CSV file(s), a directory of CSVs, or a glob pattern")
    ap.add_argument("--baseline-label", default="baseline", help="legend/label for --baseline")
    ap.add_argument("--optimized-label", default="optimized", help="legend/label for --optimized")
    ap.add_argument("--out", default=OUT_DIR, help="output directory")
    args = ap.parse_args()

    baseline_csvs = _resolve_csvs(args.baseline)
    optimized_csvs = _resolve_csvs(args.optimized)
    if not baseline_csvs:
        raise SystemExit(f"no CSVs matched --baseline {args.baseline}")
    if not optimized_csvs:
        raise SystemExit(f"no CSVs matched --optimized {args.optimized}")

    print(f"[baseline] {len(baseline_csvs)} file(s)")
    baseline, baseline_rows = _rows_by_channel(baseline_csvs)
    # Optimized reuses baseline's full-scale range per channel so both runs'
    # percentages are on the same scale -- see module docstring.
    fs_ranges = {r["name"]: r["full_scale_range_raw"] for r in baseline_rows}
    print(f"[optimized] {len(optimized_csvs)} file(s)")
    optimized, optimized_rows = _rows_by_channel(optimized_csvs, fs_ranges=fs_ranges)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "baseline_channel_gap_summary.json"), "w") as f:
        json.dump(sorted(baseline_rows, key=lambda r: r["pct_max_of_full_scale"], reverse=True), f, indent=2)
    with open(os.path.join(args.out, "optimized_channel_gap_summary.json"), "w") as f:
        json.dump(sorted(optimized_rows, key=lambda r: r["pct_max_of_full_scale"], reverse=True), f, indent=2)

    peak_rows = plot_metric_comparison(
        baseline, optimized, "pct_max_of_full_scale", "max_gap_display",
        os.path.join(args.out, "peak_gap_comparison.png"),
        "Peak reality gap by channel: baseline vs optimized",
        args.baseline_label, args.optimized_label)
    rms_rows = plot_metric_comparison(
        baseline, optimized, "pct_rms_of_full_scale", "rms_gap_display",
        os.path.join(args.out, "rms_gap_comparison.png"),
        "RMS reality gap by channel: baseline vs optimized",
        args.baseline_label, args.optimized_label)

    summary = {"peak": peak_rows, "rms": rms_rows}
    summary_path = os.path.join(args.out, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[results] log  -> {summary_path}")


if __name__ == "__main__":
    main()
