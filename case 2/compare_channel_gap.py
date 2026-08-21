"""Compare baseline vs. optimized reality-gap results, channel by channel.

Takes two sets of recorded CSVs -- baseline and optimized, each anything
``channel_gap_bar_chart.py --data-glob`` itself would accept: individual
file paths, a directory (globbed for ``*.csv``), or a glob pattern -- computes
each run's own peak/RMS-per-channel gap (reusing
``channel_gap_bar_chart.py``'s ``load_recordings``/``collect_channel_rows``/
``settle_masks`` directly, not a reimplementation), and produces, for both the
**full run** and the **settling window only** (each move's post-motion
ringing window, ``common.segments``' ``i1:i2`` -- same scope
``channel_gap_bar_chart.py``'s own ``settling_window/`` output uses), two
grouped-bar charts: one for peak %FS, one for RMS %FS, each with a baseline
bar and an optimized bar side by side per channel.

Both runs' percentages share the *baseline's full-run* full-scale range per
channel (computed once, reused for optimized *and* for both scopes) -- the
same "shared denominator" fix ``channel_gap_bar_chart.py`` uses to keep its
own full-run/settling-window numbers comparable. Without it, an optimized
trajectory that happens to cover a different range of motion (or the
settling window's inherently narrower range) would shift %FS for reasons
having nothing to do with the gap actually shrinking.

A third chart compares **cycle time** (move duration, ``i1 - i0`` per
``common.segments`` -- the motion itself, not the post-move settle window,
matching ``train_rla.py``'s own ``cycle_time`` definition so it's directly
comparable to ``run.py``'s "predicted cycle time" output) move by move.
Baseline and optimized are expected to be the *same* script(s) run at
different speed settings (e.g. ``foo.script`` vs ``foo.optimized.script``),
so their segments pair up 1:1 in recording order; a segment-count mismatch
is reported as a warning and only the first N of each are paired. Cycle time
has no settling-window variant -- it's defined as the motion itself, which is
the settling window's complement, not something to further restrict.

    python compare_channel_gap.py --baseline data/*.csv --optimized data_optimized/*.csv
    python compare_channel_gap.py --baseline data --optimized data_optimized

Outputs -> bronze_tier/channel_gap/comparison/ (or --out):
    full_run/peak_gap_comparison.png          grouped bars, baseline vs optimized
    full_run/rms_gap_comparison.png           peak/RMS %FS, whole recordings
    full_run/baseline_channel_gap_summary.json  each run's own full per-channel
    full_run/optimized_channel_gap_summary.json breakdown, same shape
                                               channel_gap_bar_chart.py writes
    settling_window/peak_gap_comparison.png   same, but rows restricted to each
    settling_window/rms_gap_comparison.png    move's post-motion settle window
    settling_window/baseline_channel_gap_summary.json
    settling_window/optimized_channel_gap_summary.json
    cycle_time_comparison.png          grouped bars, baseline vs optimized cycle
                                       time (s), move by move + total in the title
    baseline_cycle_times.json          per-move cycle time, each run
    optimized_cycle_times.json
    comparison_summary.json            per-channel (full-run + settle-window) and
                                       per-move baseline/optimized values and
                                       % change, for a table slide
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
from channel_gap_bar_chart import collect_channel_rows, load_recordings, settle_masks
from common import segments

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


def _rows_by_channel(csvs: list[str], fs_ranges: dict | None = None) -> tuple[dict, list[dict], dict]:
    """Load ``csvs`` and compute this run's per-channel gap rows.

    Returns ``({name: row}, rows, recs)`` -- the dict and the list (JSON-shaped
    like ``channel_gap_bar_chart.py``'s own ``channel_gap_summary.json``) for
    the per-channel comparison logic and per-run summary file, plus the raw
    ``Recording`` objects (``recs``) so cycle-time comparison can reuse them
    without reloading every CSV a second time.
    """
    recs = load_recordings(csvs)
    rows = collect_channel_rows(recs, fs_ranges=fs_ranges)
    return {r["name"]: r for r in rows}, rows, recs


def _settle_rows_by_channel(recs: dict, fs_ranges: dict) -> tuple[dict, list[dict]]:
    """Settle-window-only per-channel gap rows for already-loaded ``recs``.

    Takes ``recs`` (not CSV paths) so it reuses each run's already-loaded
    ``Recording`` objects instead of re-reading every CSV a second time.
    ``fs_ranges`` is required (not optional, unlike ``_rows_by_channel``) --
    it must be the *full-run* denominator, same as
    ``channel_gap_bar_chart.py``'s own settling-window output reuses, so a
    settle-window %FS is never computed against the settle window's own
    (narrower, not comparable) range.
    """
    masks = settle_masks(recs)
    rows = collect_channel_rows(recs, masks=masks, fs_ranges=fs_ranges)
    return {r["name"]: r for r in rows}, rows


def _segment_cycle_times(recs: dict) -> list[dict]:
    """Per-move cycle time (``i1 - i0``, seconds -- the motion itself, not the
    post-move settle window) for every segment in every recording, in
    encounter order.

    Matches ``train_rla.py``'s own ``cycle_time`` (move duration only) so
    this stays directly comparable to ``run.py``'s "predicted cycle time"
    output, rather than inventing a different definition here.
    """
    out = []
    idx = 0
    for name, rec in recs.items():
        for seg in segments(rec):
            idx += 1
            out.append({"index": idx, "file": name,
                        "cycle_time": float(rec.t[seg.i1] - rec.t[seg.i0])})
    return out


def _grouped_compare_bar(ax, labels: list[str], baseline_vals: list[float],
                         optimized_vals: list[float], baseline_label: str, optimized_label: str,
                         log_scale: bool = True, legend_below: bool = False):
    """Shared grouped-bar body: one baseline + one optimized bar per label.

    Log y-axis by default, for the same reason ``channel_gap_bar_chart.py``'s
    own peak/RMS grouped bar uses one -- %FS spans several orders of
    magnitude across channels. ``log_scale=False`` (used for cycle time,
    where baseline/optimized are the same order of magnitude and a log axis
    would just make similar-sized bars harder to compare) uses a plain
    zero-based linear axis instead.

    ``legend_below`` places the legend under the axes instead of the default
    upper-right-inside-axes corner. The peak/RMS charts sort channels tallest
    first, so their tallest bar (and its label) always sits at the left, away
    from an upper-right legend; cycle time's bars stay in move order, so its
    tallest bar can land anywhere, including under the legend -- confirmed
    directly (move 6 in a real run did). Moving the legend below sidesteps
    that regardless of which bar ends up tallest, rather than continuing to
    special-case corners.
    """
    x = np.arange(len(labels))
    w = 0.34
    bars_base = ax.bar(x - w / 2, baseline_vals, width=w, color=ur_style.BLUE, label=baseline_label)
    bars_opt = ax.bar(x + w / 2, optimized_vals, width=w, color=ur_style.MID_BLUE, label=optimized_label)
    all_vals = baseline_vals + optimized_vals
    if log_scale:
        ax.set_yscale("log")
        pos_vals = [v for v in all_vals if v > 0]
        floor = min(pos_vals) * 0.5 if pos_vals else 1e-3
        ceiling = max(pos_vals) * 4 if pos_vals else 1.0
        ax.set_ylim(floor, ceiling)
    else:
        # 1.6x: leaves room for _annotate_pct_change's label, which sits at
        # 1.6x the taller bar's height -- too little headroom clips it off
        # the top of the axes (confirmed visually before this fix).
        ax.set_ylim(0, max(all_vals) * 1.6 if all_vals else 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    if legend_below:
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    else:
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
    # Capped below the axes' top ~12% so a label over the tallest bar can
    # never land inside the "upper right" legend box, whichever label that
    # ends up being -- unlike the peak/RMS charts (bars sorted tallest-first,
    # so the tallest bar and its label sit at the left, away from the
    # legend), this helper is also used for cycle time's natural move order,
    # where the tallest bar can land anywhere, including under the legend
    # (confirmed directly: move 6 in a real comparison did, before this cap).
    label_cap = ax.get_ylim()[1] * 0.88
    for bar_b, bar_o, b, o in zip(bars_base, bars_opt, baseline_vals, optimized_vals):
        if b <= 0:
            continue
        pct_change = (o - b) / b * 100
        x = (bar_b.get_x() + bar_b.get_width() / 2 + bar_o.get_x() + bar_o.get_width() / 2) / 2
        y = min(max(bar_b.get_height(), bar_o.get_height()) * 1.6, label_cap)
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


def plot_cycle_time_comparison(baseline_segs: list[dict], optimized_segs: list[dict],
                               out_path: str, baseline_label: str, optimized_label: str) -> list[dict]:
    """One grouped-bar chart, baseline vs optimized cycle time (s) per move.

    ``baseline_segs``/``optimized_segs`` are pooled across every file in
    recording order (``_segment_cycle_times``). Paired positionally -- move 1
    of baseline against move 1 of optimized, etc. -- which only means what it
    should if both sets are the same script(s) at different speeds. A
    segment-count mismatch is not silently reshaped; it's reported and only
    the first N of each (N = the smaller count) are plotted.
    """
    n = min(len(baseline_segs), len(optimized_segs))
    if len(baseline_segs) != len(optimized_segs):
        print(f"[warn] baseline has {len(baseline_segs)} move(s), optimized has "
              f"{len(optimized_segs)} -- pairing the first {n} in recording order, "
              f"extra move(s) dropped from this chart")

    labels = [f"move {i + 1}" for i in range(n)]
    baseline_vals = [baseline_segs[i]["cycle_time"] for i in range(n)]
    optimized_vals = [optimized_segs[i]["cycle_time"] for i in range(n)]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    bars_base, bars_opt = _grouped_compare_bar(ax, labels, baseline_vals, optimized_vals,
                                               baseline_label, optimized_label, log_scale=False,
                                               legend_below=True)
    for bar, v in zip(bars_base, baseline_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.3g}s",
                ha="center", va="bottom", fontsize=7, color=ur_style.NAVY)
    for bar, v in zip(bars_opt, optimized_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.3g}s",
                ha="center", va="bottom", fontsize=7, color=ur_style.NAVY)
    _annotate_pct_change(ax, bars_base, bars_opt, baseline_vals, optimized_vals)

    total_base, total_opt = sum(baseline_vals), sum(optimized_vals)
    total_pct = (total_opt - total_base) / total_base * 100 if total_base else float("nan")
    ax.set_ylabel("cycle time (s)")
    ax.set_title(f"Cycle time by move: {baseline_label} vs {optimized_label}\n"
                f"total: {total_base:.2f}s -> {total_opt:.2f}s ({total_pct:+.1f}%)")
    # bbox_inches="tight" (not fig.tight_layout(), which doesn't reliably
    # account for a legend placed outside the axes) so the below-axes legend
    # doesn't get clipped off the saved image.
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[results] plot -> {out_path}")

    return [{"move": labels[i], "baseline_s": baseline_vals[i], "optimized_s": optimized_vals[i],
            "pct_change": (optimized_vals[i] - baseline_vals[i]) / baseline_vals[i] * 100
            if baseline_vals[i] > 0 else None}
           for i in range(n)]


def _write_scope_comparison(baseline: dict, optimized: dict, baseline_rows: list[dict],
                            optimized_rows: list[dict], out_dir: str, scope_title: str,
                            baseline_label: str, optimized_label: str) -> tuple[list[dict], list[dict]]:
    """Summaries + peak/RMS grouped-bar charts for one scope (full run, or
    settle window only) -- the block ``main`` runs once per scope, out of
    ``full_run/`` and ``settling_window/`` respectively, so the two scopes
    stay structurally identical and only differ in which rows were pooled.
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "baseline_channel_gap_summary.json"), "w") as f:
        json.dump(sorted(baseline_rows, key=lambda r: r["pct_max_of_full_scale"], reverse=True), f, indent=2)
    with open(os.path.join(out_dir, "optimized_channel_gap_summary.json"), "w") as f:
        json.dump(sorted(optimized_rows, key=lambda r: r["pct_max_of_full_scale"], reverse=True), f, indent=2)

    peak_rows = plot_metric_comparison(
        baseline, optimized, "pct_max_of_full_scale", "max_gap_display",
        os.path.join(out_dir, "peak_gap_comparison.png"),
        f"Peak reality gap by channel ({scope_title}): baseline vs optimized",
        baseline_label, optimized_label)
    rms_rows = plot_metric_comparison(
        baseline, optimized, "pct_rms_of_full_scale", "rms_gap_display",
        os.path.join(out_dir, "rms_gap_comparison.png"),
        f"RMS reality gap by channel ({scope_title}): baseline vs optimized",
        baseline_label, optimized_label)
    return peak_rows, rms_rows


def main():
    ap = argparse.ArgumentParser(
        description="Compare baseline vs. optimized recordings (peak + RMS reality gap %FS "
                    "per channel, full run and settle window only).")
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
    baseline, baseline_rows, baseline_recs = _rows_by_channel(baseline_csvs)
    # Optimized reuses baseline's full-scale range per channel so both runs'
    # -- and both scopes' -- percentages are on the same scale, see module
    # docstring.
    fs_ranges = {r["name"]: r["full_scale_range_raw"] for r in baseline_rows}
    print(f"[optimized] {len(optimized_csvs)} file(s)")
    optimized, optimized_rows, optimized_recs = _rows_by_channel(optimized_csvs, fs_ranges=fs_ranges)

    os.makedirs(args.out, exist_ok=True)
    peak_rows, rms_rows = _write_scope_comparison(
        baseline, optimized, baseline_rows, optimized_rows,
        os.path.join(args.out, "full_run"), "whole run",
        args.baseline_label, args.optimized_label)

    baseline_settle, baseline_settle_rows = _settle_rows_by_channel(baseline_recs, fs_ranges)
    optimized_settle, optimized_settle_rows = _settle_rows_by_channel(optimized_recs, fs_ranges)
    peak_settle_rows, rms_settle_rows = _write_scope_comparison(
        baseline_settle, optimized_settle, baseline_settle_rows, optimized_settle_rows,
        os.path.join(args.out, "settling_window"), "settle window only",
        args.baseline_label, args.optimized_label)

    baseline_segs = _segment_cycle_times(baseline_recs)
    optimized_segs = _segment_cycle_times(optimized_recs)
    with open(os.path.join(args.out, "baseline_cycle_times.json"), "w") as f:
        json.dump(baseline_segs, f, indent=2)
    with open(os.path.join(args.out, "optimized_cycle_times.json"), "w") as f:
        json.dump(optimized_segs, f, indent=2)
    cycle_rows = plot_cycle_time_comparison(
        baseline_segs, optimized_segs, os.path.join(args.out, "cycle_time_comparison.png"),
        args.baseline_label, args.optimized_label)

    summary = {"peak": peak_rows, "rms": rms_rows,
              "peak_settle": peak_settle_rows, "rms_settle": rms_settle_rows,
              "cycle_time": cycle_rows}
    summary_path = os.path.join(args.out, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[results] log  -> {summary_path}")


if __name__ == "__main__":
    main()
