"""Plot one move's actual joint-angle trajectory: recorded vs PPO-optimized.

Where plot_baseline_vs_optimized.py shows the aggregate error/cycle-time
numbers, this shows the *shape* of one move -- the recorded movej's q(t)
against the trained policy's chosen (vel, acc) movej, for whichever joint moves
the most, so a speed-up (or slow-down) is visible as an actual curve, not just
a summary number.

    python plot_move_trajectory.py --agent models/agent_ppo_movej.zip \\
        --model models/distill-ur5e-v2.pkl --robot UR5e \\
        --recording data/ur5e/T01_fast_r1.csv --move 0
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from analysis import Recording
from common import segments
from train_distillation_model import DistillModel
from train_ppo import _Move, candidate_q, obs_of, unmap_action
from utils import DT, JOINT_NAMES, Robot


def main():
    ap = argparse.ArgumentParser(description="Plot one move's recorded vs PPO-optimized trajectory.")
    ap.add_argument("--agent", default="models/agent_ppo_movej.zip", help="trained PPO agent")
    ap.add_argument("--model", default="models/distill-ur5e-v2.pkl", help="distilled model pickle")
    ap.add_argument("--robot", default="UR5e", choices=list(Robot.MODELS))
    ap.add_argument("--recording", required=True, help="a data/ur5e/*.csv recording")
    ap.add_argument("--move", type=int, default=0, help="which move in the recording (0-indexed)")
    ap.add_argument("--out", default=None, help="default: <recording>.move<N>.png")
    args = ap.parse_args()

    agent = PPO.load(args.agent, device="cpu")
    model = DistillModel.load(args.model)
    for p in model.parameters():
        p.requires_grad_(False)
    robot = Robot(args.robot)
    rec = Recording(args.recording)

    segs = [s for s in segments(rec) if s.i1 - s.i0 >= 3]
    if not (0 <= args.move < len(segs)):
        raise SystemExit(f"{args.recording} has {len(segs)} usable moves, --move must be in [0, {len(segs) - 1}]")
    s = segs[args.move]
    m = _Move(rec, s, robot)

    action, _ = agent.predict(obs_of(m), deterministic=True)
    vel, acc = unmap_action(action)
    q_ppo = candidate_q(m, robot, vel, acc)
    q_rec = m.baseline_q

    j = s.joint   # the widest-travel joint, most visibly showing the timing difference
    t_rec = np.arange(len(q_rec)) * DT
    t_ppo = np.arange(len(q_ppo)) * DT

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)

    axes[0].plot(t_rec, np.rad2deg(q_rec[:, j]), label=f"recorded ({len(q_rec) * DT:.2f} s)",
                color="#888888", linewidth=2)
    axes[0].plot(t_ppo, np.rad2deg(q_ppo[:, j]), label=f"PPO, vel={vel:.2f} acc={acc:.2f} "
                f"rad/s(^2) ({len(q_ppo) * DT:.2f} s)", color="#1f77b4", linewidth=2)
    axes[0].set_ylabel(f"{JOINT_NAMES[j]} angle (deg)")
    axes[0].set_xlabel("time (s)")
    axes[0].set_title(f"{args.recording}  move {args.move}  ({s.dist:.3f} rad travel)")
    axes[0].legend()

    qd_rec = np.gradient(q_rec[:, j], DT)
    qd_ppo = np.gradient(q_ppo[:, j], DT)
    axes[1].plot(t_rec, np.rad2deg(qd_rec), color="#888888", linewidth=2, label="recorded")
    axes[1].plot(t_ppo, np.rad2deg(qd_ppo), color="#1f77b4", linewidth=2, label="PPO")
    axes[1].set_ylabel(f"{JOINT_NAMES[j]} speed (deg/s)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend()

    fig.tight_layout()
    out = args.out or args.recording.rsplit(".", 1)[0] + f".move{args.move}.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
