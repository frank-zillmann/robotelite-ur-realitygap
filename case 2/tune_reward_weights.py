"""Calibrate and visualize train_rla.py's reward-objective weight balance.

``train_rla.py``'s ``OBJECTIVE = SCORE_WEIGHT*score + CYCLE_WEIGHT*cycle_time``
(``PATH_OBJECTIVE`` in path mode) needs ``SCORE_WEIGHT`` recalibrated
whenever the active distill model/metric changes the scale of ``score`` --
its own code comment documents a one-off manual exercise (sample random
actions, measure mean score/cycle_time, pick a weight) that produced the
current ``SCORE_WEIGHT=2500.0``. This script automates that exercise and
adds a second check that exercise never covered: a *trained* policy's own
score/cycle_time distribution can drift away from the random-action
baseline the weight was calibrated against (concretely: this project's
``results/2026-08-20_12-31-40_rla_params/`` run trained until cycle_time
dominated the weighted cost ~3.5:1, not the ~1:1 the calibration assumed --
scoring barely improved because the agent had far more reward gradient
available from getting faster than from reducing vibration).

Two ways to get (score, cycle_time) samples:

  live sampling (default) -- builds the same GapEnv/PathEnv train_rla.py
      trains against and samples random actions through it:

          python tune_reward_weights.py --model models/distill.pkl \\
              --scripts scripts/shoulder_swing.script --robot-ip 127.0.0.1 \\
              --loop 5 --mode params --n-samples 500

  --episodes <path> -- reuses a completed run's own episodes.csv (written by
      train_rla.py's log_training_run) instead of sampling live -- checks
      the *trained* policy's actual distribution, not just the pre-training
      random baseline:

          python tune_reward_weights.py \\
              --episodes results/2026-08-20_12-31-40_rla_params/episodes.csv

Outputs -> results/<timestamp>_reward_tuning/ (or --out):
    samples.csv              score, cycle_time, cost (at the current weight)
    weight_sensitivity.png   score term's share of total cost vs candidate weight
    score_vs_cycle.png       score vs cycle_time scatter with iso-cost lines
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ur_style
from metrics import PositionGapMetric
from preprocess import default_preprocess
from train_distillation_model import DistillModel
from train_rla import (CYCLE_WEIGHT, PATH_CYCLE_WEIGHT, PATH_SCORE_WEIGHT,
                       RESULTS_DIR, SCORE_WEIGHT, GapEnv, PathEnv,
                       build_dataset, default_dynamics)

ur_style.apply()


def sample_live(model, scripts: list, robot_ip: str, loop, mode: str,
                n_samples: int, sim_csv: str) -> pd.DataFrame:
    """Sample ``n_samples`` random actions through the real env, one episode
    each, collecting ``(score, cycle_time)`` -- the automated version of the
    manual calibration exercise documented in train_rla.py's SCORE_WEIGHT
    comment. Uses ``env.action_space.sample()`` + ``env.step()`` (not a
    hand-rolled vel/acc loop) so this exercises the exact same code path
    real training does, action-unmapping included.
    """
    metric = PositionGapMetric()
    pre = default_preprocess()
    rec = build_dataset(model, metric, scripts, robot_ip, loop, pre, sim_csv)
    dyn = default_dynamics(rec)
    Env = GapEnv if mode == "params" else PathEnv
    env = Env(model, metric, rec, dyn=dyn, pre=pre)

    rows = []
    env.reset(seed=0)
    for _ in range(n_samples):
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        rows.append({"score": info["score"], "cycle_time": info["cycle_time"]})
        env.reset()
    return pd.DataFrame(rows)


def analyze(df: pd.DataFrame, score_weight: float, cycle_weight: float,
           target_ratio: float, out_dir: str, weight_label: str) -> float:
    """Print the current weight's balance, recommend one for ``target_ratio``,
    and save the two diagnostic plots. Returns the recommended weight.
    """
    mean_score = float(df["score"].mean())
    mean_cycle = float(df["cycle_time"].mean())

    current_score_term = score_weight * mean_score
    current_cycle_term = cycle_weight * mean_cycle
    current_ratio = current_score_term / current_cycle_term if current_cycle_term else float("nan")
    recommended = target_ratio * cycle_weight * mean_cycle / mean_score if mean_score else float("nan")

    print(f"samples: {len(df)}")
    print(f"mean score:      {mean_score:.6g}")
    print(f"mean cycle_time: {mean_cycle:.6g} s")
    print(f"current {weight_label}={score_weight:.6g}  ->  weighted (score:cycle) "
         f"= {current_ratio:.3g}:1")
    print(f"target ratio {target_ratio:.3g}:1  ->  recommended {weight_label} "
         f"= {recommended:.6g}")

    # ---- weight_sensitivity.png ----------------------------------------------
    span_lo = min(recommended, score_weight) / 50
    span_hi = max(recommended, score_weight) * 50
    candidates = np.logspace(np.log10(max(span_lo, 1e-9)), np.log10(span_hi), 300)
    score_share = (candidates * mean_score) / (candidates * mean_score + cycle_weight * mean_cycle)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(candidates, score_share, color=ur_style.BLUE, lw=2)
    ax.axhline(0.5, color=ur_style.GRAY, linestyle="--", linewidth=1.0, label="balanced (50/50)")
    target_share = target_ratio / (target_ratio + 1)
    ax.axhline(target_share, color=ur_style.MID_BLUE, linestyle="--", linewidth=1.2,
              label=f"target ratio {target_ratio:.2g}:1 ({target_share:.0%})")
    ax.axvline(score_weight, color=ur_style.NAVY, linestyle=":", linewidth=1.8,
              label=f"current {weight_label}={score_weight:.4g}")
    ax.axvline(recommended, color=ur_style.DARK_BLUE, linestyle=":", linewidth=1.8,
              label=f"recommended {weight_label}={recommended:.4g}")
    ax.set_xscale("log")
    ax.set_xlabel(f"candidate {weight_label} (log scale)")
    ax.set_ylabel("score term's share of mean total weighted cost")
    ax.set_ylim(0, 1)
    ax.set_title("Reward weight sensitivity")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    p1 = os.path.join(out_dir, "weight_sensitivity.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"[results] plot -> {p1}")

    # ---- score_vs_cycle.png ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["cycle_time"], df["score"], s=14, alpha=0.5, color=ur_style.GRAY,
              label="samples", zorder=2)
    x = np.linspace(df["cycle_time"].min() * 0.9, df["cycle_time"].max() * 1.1, 200)
    for w, color, tag in [(score_weight, ur_style.NAVY, f"current {weight_label}={score_weight:.4g}"),
                          (recommended, ur_style.BLUE, f"recommended {weight_label}={recommended:.4g}")]:
        # Iso-cost line through the mean point: w*score + cycle_weight*cycle = const.
        # Points below/left of a line cost less under that weight than the mean
        # sample does -- shows how "good" shifts as the weight changes.
        const = w * mean_score + cycle_weight * mean_cycle
        y = (const - cycle_weight * x) / w
        mask = y >= 0
        ax.plot(x[mask], y[mask], color=color, linewidth=1.8, label=f"iso-cost, {tag}", zorder=3)
    ax.scatter([mean_cycle], [mean_score], marker="x", s=60, color=ur_style.NAVY, zorder=4)
    ax.set_xlabel("cycle_time (s)")
    ax.set_ylabel("score (rad)")
    ax.set_title("Score vs cycle_time — iso-cost lines under different weights")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    p2 = os.path.join(out_dir, "score_vs_cycle.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"[results] plot -> {p2}")

    return recommended


def main():
    ap = argparse.ArgumentParser(
        description="Calibrate and visualize train_rla.py's SCORE_WEIGHT/CYCLE_WEIGHT balance.")
    ap.add_argument("--episodes", default=None,
                    help="reuse a completed run's episodes.csv instead of live-sampling")
    ap.add_argument("--model", default="models/distill.pkl",
                    help="distilled model (live-sampling mode)")
    ap.add_argument("--scripts", nargs="+", default=["scripts/shoulder_swing.script"],
                    help="URScript(s) to sample moves from (live-sampling mode)")
    ap.add_argument("--robot-ip", default="127.0.0.1",
                    help="URSim IP (live-sampling mode)")
    ap.add_argument("--loop", type=int, default=None,
                    help="repeat each script N times when collecting moves (live-sampling mode)")
    ap.add_argument("--mode", choices=("params", "path"), default="params",
                    help="which env/objective to check: params -> GapEnv/SCORE_WEIGHT, "
                        "path -> PathEnv/PATH_SCORE_WEIGHT (must match --episodes' source "
                        "if given)")
    ap.add_argument("--n-samples", type=int, default=500,
                    help="random actions to sample (live-sampling mode, default: %(default)s)")
    ap.add_argument("--target-ratio", type=float, default=2.0,
                    help="target weighted (score term : cycle_time term) ratio "
                        "(default: %(default)s -- score counts twice cycle_time)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: results/<timestamp>_reward_tuning)")
    args = ap.parse_args()

    dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = args.out or os.path.join(RESULTS_DIR, f"{dt_str}_reward_tuning")
    os.makedirs(out_dir, exist_ok=True)

    score_weight, cycle_weight, weight_label = (
        (SCORE_WEIGHT, CYCLE_WEIGHT, "SCORE_WEIGHT") if args.mode == "params"
        else (PATH_SCORE_WEIGHT, PATH_CYCLE_WEIGHT, "PATH_SCORE_WEIGHT"))

    if args.episodes:
        df = pd.read_csv(args.episodes)[["score", "cycle_time"]]
        print(f"loaded {len(df)} episodes from {args.episodes}")
    else:
        model = DistillModel.load(args.model)
        sim_csv = os.path.join(out_dir, "sim_to_real.csv")
        df = sample_live(model, args.scripts, args.robot_ip, args.loop, args.mode,
                         args.n_samples, sim_csv)
        print(f"sampled {len(df)} random actions")

    df["cost"] = score_weight * df["score"] + cycle_weight * df["cycle_time"]
    samples_path = os.path.join(out_dir, "samples.csv")
    df.to_csv(samples_path, index=False)
    print(f"[results] samples -> {samples_path}")

    recommended = analyze(df, score_weight, cycle_weight, args.target_ratio, out_dir, weight_label)
    print(f"\nTo apply: set {weight_label} = {recommended:.6g} in train_rla.py")


if __name__ == "__main__":
    main()
