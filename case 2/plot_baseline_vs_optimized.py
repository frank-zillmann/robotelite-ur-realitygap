"""Plot the recorded (baseline) movej against the trained PPO policy's choice.

Reuses train_ppo.py's own evaluate() -- so this is exactly the same before/after
numbers the training run prints at the end, just visualized: mean error and
total cycle time (bar, %-change titles), and a per-move scatter against y=x so
individual wins/losses are visible, not just the aggregate.

    python plot_baseline_vs_optimized.py --agent models/agent_ppo_movej.zip \\
        --model models/distill-ur5e-v2.pkl --robot UR5e --data data/ur5e
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from train_distillation_model import DistillModel, load_recordings
from train_ppo import build_moves, MoveEnv, evaluate
from utils import Robot


def plot(rows, out: str, title: str):
    before_err = np.array([r[0] for r in rows]) * 1000   # mrad
    err        = np.array([r[1] for r in rows]) * 1000
    before_T   = np.array([r[2] for r in rows])
    cyc        = np.array([r[3] for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    m_before, m_after = before_err.mean(), err.mean()
    axes[0, 0].bar(["recorded", "PPO"], [m_before, m_after], color=["#888888", "#1f77b4"])
    axes[0, 0].set_title(f"Mean error: {m_before:.3f} -> {m_after:.3f} mrad "
                         f"({(m_after - m_before) / m_before * 100:+.1f}%)")
    axes[0, 0].set_ylabel("Error (mrad, lower = better)")

    t_before, t_after = before_T.sum(), cyc.sum()
    axes[0, 1].bar(["recorded", "PPO"], [t_before, t_after], color=["#888888", "#ff7f0e"])
    axes[0, 1].set_title(f"Total cycle time: {t_before:.1f} -> {t_after:.1f} s "
                         f"({(t_after - t_before) / t_before * 100:+.1f}%)")
    axes[0, 1].set_ylabel("Cycle time (s, lower = faster)")

    lim = max(before_err.max(), err.max())
    axes[1, 0].scatter(before_err, err, s=8, alpha=0.4)
    axes[1, 0].plot([0, lim], [0, lim], "k--", linewidth=1, label="y = x")
    axes[1, 0].set_xlabel("recorded error (mrad)")
    axes[1, 0].set_ylabel("PPO error (mrad)")
    axes[1, 0].set_title("Per-move error -- above the line = worse than recorded")
    axes[1, 0].legend()

    lim = max(before_T.max(), cyc.max())
    axes[1, 1].scatter(before_T, cyc, s=8, alpha=0.4, color="#ff7f0e")
    axes[1, 1].plot([0, lim], [0, lim], "k--", linewidth=1, label="y = x")
    axes[1, 1].set_xlabel("recorded cycle time (s)")
    axes[1, 1].set_ylabel("PPO cycle time (s)")
    axes[1, 1].set_title("Per-move cycle time -- below the line = faster")
    axes[1, 1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot baseline vs PPO-optimized error/cycle time.")
    ap.add_argument("--agent", default="models/agent_ppo_movej.zip", help="trained PPO agent")
    ap.add_argument("--model", default="models/distill-ur5e-v2.pkl", help="distilled model pickle")
    ap.add_argument("--robot", default="UR5e", choices=list(Robot.MODELS))
    ap.add_argument("--data", default="data/ur5e", help="folder of recordings")
    ap.add_argument("--out", default="results/baseline_vs_optimized.png")
    args = ap.parse_args()

    model = DistillModel.load(args.model)
    for p in model.parameters():
        p.requires_grad_(False)
    robot = Robot(args.robot)
    recordings = load_recordings(args.data)
    moves = build_moves(recordings, robot)
    env = MoveEnv(model, robot, moves)
    agent = PPO.load(args.agent, device="cpu")

    rows = evaluate(agent, env)
    plot(rows, args.out, f"Baseline vs PPO-optimized movej -- {len(rows)} moves ({args.data})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
