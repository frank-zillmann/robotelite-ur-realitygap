"""Aggregate every results/hardware_tests/<name>/ pair into one chart -- the
real measured comparison across all scripts you've run on hardware, not a
prediction.

    python plot_hardware_results.py --model models/distill-ur5e-v2.pkl

Scans results/hardware_tests/*/ for baseline_result.csv + optimized_result.csv
pairs (skips any folder missing one), scores each with the distill model the
same way evaluate.py does, and plots cycle time + error, baseline vs optimized,
one bar pair per script.
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import Recording
from evaluate import evaluate
from train_distillation_model import DistillModel


def main():
    ap = argparse.ArgumentParser(description="Plot all hardware A/B results together.")
    ap.add_argument("--model", default="models/distill-ur5e-v2.pkl", help="distilled model pickle")
    ap.add_argument("--data-dir", default=os.path.join("results", "hardware_tests"))
    ap.add_argument("--out", default=os.path.join("results", "hardware_summary.png"))
    args = ap.parse_args()

    model = DistillModel.load(args.model)
    names, base_err, opt_err, base_t, opt_t = [], [], [], [], []

    for folder in sorted(glob.glob(os.path.join(args.data_dir, "*"))):
        base_csv = os.path.join(folder, "baseline_result.csv")
        opt_csv = os.path.join(folder, "optimized_result.csv")
        if not (os.path.exists(base_csv) and os.path.exists(opt_csv)):
            continue
        name = os.path.basename(folder)
        for csv, err_list, t_list in ((base_csv, base_err, base_t), (opt_csv, opt_err, opt_t)):
            rec = Recording(csv)
            err_rows = evaluate(model, [rec])["actual_q"][0]
            err_list.append(float(np.mean(err_rows)) * 1000)
            t_list.append(float(rec.t[-1] - rec.t[0]))
        names.append(name)

    if not names:
        raise SystemExit(f"no complete baseline/optimized pairs found under {args.data_dir}/ yet")

    x = np.arange(len(names))
    w = 0.35
    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(names) * 1.1), 9), sharex=True)

    axes[0].bar(x - w / 2, base_err, w, label="baseline", color="#888888")
    axes[0].bar(x + w / 2, opt_err, w, label="optimized", color="#1f77b4")
    axes[0].set_ylabel("Measured error (mrad)")
    axes[0].set_title(f"Hardware results: {len(names)} scripts, baseline vs PPO-optimized")
    axes[0].legend()

    axes[1].bar(x - w / 2, base_t, w, label="baseline", color="#888888")
    axes[1].bar(x + w / 2, opt_t, w, label="optimized", color="#ff7f0e")
    axes[1].set_ylabel("Cycle time (s)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right")
    axes[1].legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out} ({len(names)} scripts)")


if __name__ == "__main__":
    main()
