"""Multi-run, multi-joint comparison report -- a single self-contained HTML file.

``analysis.py`` shows one joint of one run at a time in a matplotlib pop-up and
saves nothing. This lays out every joint of every run you give it side by side
(current + gap), the full ``diagnostics.py`` metric set (position RMSE/max error,
current-gap RMS, RMS/peak jerk per joint; cycle time, overshoot, settling time,
residual vibration per run), and, with ``--profiles``, the ``trapezoidal`` vs
``s_curve`` speed profiles from ``dynamics.py`` -- into one HTML file you can open
in a browser or send to someone. No new dependencies: matplotlib (already
required) rendered to embedded PNGs, so the file needs no server and no internet
connection to view.

    # compare baseline vs optimized (or any recordings) across all 6 joints
    python report.py --csvs baseline.csv optimized.csv --out report.html

    # also compare the two speed-profile strategies for a given move
    python report.py --csvs baseline.csv optimized.csv --profiles \
        --distance 1.0 --vel 1.0 --acc 2.0 --out report.html

    # add reality-gap prediction error: how well models/distill.pkl predicts
    # these specific recordings (not just its own training rows)
    python report.py --csvs baseline.csv optimized.csv --distill models/distill.pkl

``--labels`` names the runs in the report (default: each CSV's filename).
"""
from __future__ import annotations

import argparse
import base64
import io
import os

import matplotlib
matplotlib.use("Agg")            # render to a buffer only, no display needed
import matplotlib.pyplot as plt
import numpy as np

from analysis import Recording
from diagnostics import per_joint_metrics, per_run_metrics, reality_gap_error
from dynamics import s_curve, trapezoidal
from utils import JOINT_NAMES, N_JOINTS

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]


def _b64_png(fig) -> str:
    """Render a matplotlib figure to a base64 PNG data URI, then close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def current_figure(recs: dict, joint: int):
    """One joint: the first run's target current, every run's actual current
    and current gap, side by side."""
    fig, (ax_c, ax_g) = plt.subplots(1, 2, figsize=(10, 2.4))
    first = next(iter(recs.values()))
    ax_c.plot(first.t, first.target_current[:, joint], color="black", lw=1.2,
              ls="--", label="target")
    for (label, rec), color in zip(recs.items(), PALETTE):
        ax_c.plot(rec.t, rec.actual_current[:, joint], color=color, lw=1, label=label)
        ax_g.plot(rec.t, rec.current_gap(joint), color=color, lw=1, label=label)
    ax_c.set_title(f"{JOINT_NAMES[joint]}  current (A)")
    ax_g.set_title(f"{JOINT_NAMES[joint]}  actual - target (A)")
    ax_g.axhline(0, color="grey", lw=0.6)
    for ax in (ax_c, ax_g):
        ax.set_xlabel("time (s)")
        ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    return fig


def profile_figure(distance: float, vel: float, acc: float, dt: float):
    """Position / velocity / acceleration of trapezoidal vs s_curve, same move."""
    fig, (ax_p, ax_v, ax_a) = plt.subplots(1, 3, figsize=(10, 2.6))
    for name, fn, color in (("trapezoidal", trapezoidal, PALETTE[1]),
                            ("s_curve", s_curve, PALETTE[0])):
        s = fn(distance, vel, acc, dt)
        t = np.arange(len(s)) * dt
        pos = s * distance
        v = np.gradient(pos, dt)
        a = np.gradient(v, dt)
        ax_p.plot(t, pos, color=color, label=name)
        ax_v.plot(t, v, color=color, label=name)
        ax_a.plot(t, a, color=color, label=name)
    for ax, title in ((ax_p, "position (rad)"), (ax_v, "velocity (rad/s)"),
                      (ax_a, "acceleration (rad/s^2)")):
        ax.set_title(title)
        ax.set_xlabel("time (s)")
        ax.legend(fontsize=7)
    fig.suptitle(f"speed profile: distance={distance:.3f} rad  "
                 f"vel={vel:.2f} rad/s  acc={acc:.2f} rad/s^2")
    fig.tight_layout()
    return fig


def per_joint_table(recs: dict) -> str:
    """One HTML table row per (run, joint): position RMSE/max error,
    current-gap RMS, RMS/peak jerk (diagnostics.per_joint_metrics)."""
    rows = []
    for label, rec in recs.items():
        for j in range(N_JOINTS):
            m = per_joint_metrics(rec)[JOINT_NAMES[j]]
            rows.append(f"<tr><td>{label}</td><td>{JOINT_NAMES[j]}</td>"
                        f"<td>{m['pos_rmse']*1e3:.3f}</td><td>{m['pos_max_err']*1e3:.3f}</td>"
                        f"<td>{m['current_gap_rms']:.3f}</td><td>{m['jerk_rms']:.1f}</td>"
                        f"<td>{m['jerk_peak']:.1f}</td></tr>")
    return ("<table><tr><th>run</th><th>joint</th><th>position RMSE (mrad)</th>"
            "<th>max position error (mrad)</th><th>current-gap RMS (A)</th>"
            "<th>RMS jerk (rad/s^3)</th><th>peak jerk (rad/s^3)</th></tr>"
            + "".join(rows) + "</table>")


def per_run_table(recs: dict) -> str:
    """One HTML table row per run: cycle time, and overshoot/settling/vibration
    averaged (and peaked) over every move in the run (diagnostics.per_run_metrics)."""
    rows = []
    for label, rec in recs.items():
        m = per_run_metrics(rec)
        rows.append(f"<tr><td>{label}</td><td>{m['cycle_time']:.2f}</td>"
                    f"<td>{m['overshoot_mean']*1e3:.3f} / {m['overshoot_max']*1e3:.3f}</td>"
                    f"<td>{m['settling_time_mean']*1e3:.1f} / {m['settling_time_max']*1e3:.1f}</td>"
                    f"<td>{m['residual_vib_rms_mean']*1e3:.3f} / {m['residual_vib_rms_max']*1e3:.3f}"
                    "</td></tr>")
    return ("<table><tr><th>run</th><th>cycle time (s)</th>"
            "<th>peak overshoot mean/max (mrad)</th>"
            "<th>settling time mean/max (ms)</th>"
            "<th>residual vibration RMS mean/max (mrad)</th></tr>"
            + "".join(rows) + "</table>")


def reality_gap_section(model, recs: dict) -> str:
    """RMSE/R2 of ``model``'s predictions against these specific recordings
    (diagnostics.reality_gap_error) -- whether the model predicts the real
    robot, not just how well it fit its own training rows."""
    m = reality_gap_error(model, list(recs.values()))
    return (f"<h2>Reality-gap prediction error</h2>"
            f"<p>Model {model.predicts()} vs {', '.join(recs)}: "
            f"RMSE {m['rmse']:.3f}, R2 {m['r2']:.3f}</p>")


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Case 2 report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; max-width: 1100px; }}
h1, h2 {{ margin-top: 2rem; }}
table {{ border-collapse: collapse; margin: 1rem 0; }}
th, td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: right; font-size: 0.9rem; }}
th {{ background: #f2f2f2; }}
td:first-child, td:nth-child(2), th:first-child, th:nth-child(2) {{ text-align: left; }}
img {{ max-width: 100%; display: block; margin: 0.5rem 0 1.5rem; }}
</style></head>
<body>
<h1>Case 2 report</h1>
<p>Runs: {runs}</p>
<h2>Per-joint metrics</h2>
{per_joint_table}
<h2>Per-run metrics</h2>
{per_run_table}
{reality_gap_section}
<h2>Per-joint current</h2>
{current_imgs}
{profile_section}
</body></html>
"""


def main():
    ap = argparse.ArgumentParser(description="Multi-run, multi-joint HTML comparison report.")
    ap.add_argument("--csvs", nargs="+", required=True, help="recording CSVs to compare")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="name for each CSV (default: filename without extension)")
    ap.add_argument("--out", default="report.html", help="output HTML path")
    ap.add_argument("--profiles", action="store_true",
                    help="add a trapezoidal vs s_curve speed-profile comparison")
    ap.add_argument("--distance", type=float, default=1.0, help="--profiles: move distance (rad)")
    ap.add_argument("--vel", type=float, default=1.0, help="--profiles: peak speed (rad/s)")
    ap.add_argument("--acc", type=float, default=2.0, help="--profiles: peak accel (rad/s^2)")
    ap.add_argument("--dt", type=float, default=0.008, help="--profiles: sample period (s)")
    ap.add_argument("--distill", default=None,
                    help="a DistillModel pickle; adds a reality-gap prediction-error "
                         "section (RMSE/R2 of the model against these --csvs)")
    args = ap.parse_args()

    labels = args.labels or [os.path.splitext(os.path.basename(c))[0] for c in args.csvs]
    if len(labels) != len(args.csvs):
        raise SystemExit("--labels must match --csvs in count")
    recs = {label: Recording(csv) for label, csv in zip(labels, args.csvs)}

    current_imgs = "".join(f'<img src="{_b64_png(current_figure(recs, j))}">'
                           for j in range(N_JOINTS))

    profile_section = ""
    if args.profiles:
        img = _b64_png(profile_figure(args.distance, args.vel, args.acc, args.dt))
        profile_section = f'<h2>Speed profile: trapezoidal vs s_curve</h2><img src="{img}">'

    gap_section = ""
    if args.distill:
        from train_distillation_model import DistillModel
        gap_section = reality_gap_section(DistillModel.load(args.distill), recs)

    html = HTML.format(runs=", ".join(labels), per_joint_table=per_joint_table(recs),
                       per_run_table=per_run_table(recs), reality_gap_section=gap_section,
                       current_imgs=current_imgs, profile_section=profile_section)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
