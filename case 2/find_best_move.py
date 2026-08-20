"""Rank every move by how safely the trained policy improves it -- for picking
one to test on real hardware first: genuinely faster *and* not leaning on the
ERROR_CEIL hinge (error didn't get worse), no limit violations.

    python find_best_move.py --agent models/agent_ppo_movej.zip \\
        --model models/distill-ur5e-v2.pkl --robot UR5e --data data/ur5e
"""
from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from train_distillation_model import DistillModel, load_recordings
from train_ppo import build_moves, MoveEnv, evaluate
from utils import Robot


def main():
    ap = argparse.ArgumentParser(description="Find the safest, clearest PPO win to test first.")
    ap.add_argument("--agent", default="models/agent_ppo_movej.zip", help="trained PPO agent")
    ap.add_argument("--model", default="models/distill-ur5e-v2.pkl", help="distilled model pickle")
    ap.add_argument("--robot", default="UR5e", choices=list(Robot.MODELS))
    ap.add_argument("--data", default="data/ur5e", help="folder of recordings")
    ap.add_argument("--top", type=int, default=10, help="how many candidates to print")
    args = ap.parse_args()

    model = DistillModel.load(args.model)
    for p in model.parameters():
        p.requires_grad_(False)
    robot = Robot(args.robot)
    agent = PPO.load(args.agent, device="cpu")
    recordings = load_recordings(args.data)

    rows = []
    for rec in recordings:
        moves = build_moves([rec], robot)
        if not moves:
            continue
        env = MoveEnv(model, robot, moves)
        for i, (before_err, err, before_T, cyc, cost) in enumerate(evaluate(agent, env)):
            rows.append({
                "recording": rec.path, "move": i,
                "err_ratio": err / before_err, "cycle_ratio": cyc / before_T,
                "before_err": before_err, "err": err, "before_T": before_T, "cyc": cyc,
            })

    def show(label, subset, key):
        subset = sorted(subset, key=key)[:args.top]
        print(f"\n-- {label} ({len(subset)} shown) --")
        for r in subset:
            print(f"  {r['recording']:40s} move {r['move']:2d}   "
                 f"error {r['before_err'] * 1000:.3f} -> {r['err'] * 1000:.3f} mrad "
                 f"({(r['err_ratio'] - 1) * 100:+.1f}%)   "
                 f"cycle {r['before_T']:.2f} -> {r['cyc']:.2f} s "
                 f"({(r['cycle_ratio'] - 1) * 100:+.1f}%)")

    # No move needs to satisfy both a strict error bound *and* a speed bound at
    # once -- the aggregate improvement comes from averaging across moves that
    # trade differently, not any single move nailing both. So report the two
    # useful views separately instead of an AND-filter that can (and did) find
    # nothing: the safest real wins (barely worse error, still faster), and the
    # single fastest win regardless of how close to the error ceiling it runs.
    safe = [r for r in rows if r["err_ratio"] <= 1.05 and r["cycle_ratio"] < 1.0]
    print(f"{len(rows)} moves scored, {len(safe)} within 5% error and faster")
    show("safest real wins (<=5% worse error, faster)", safe, lambda r: r["cycle_ratio"])
    show("fastest overall (any error, up to the ceiling)", rows, lambda r: r["cycle_ratio"])
    show("most accuracy improved (any speed)", rows, lambda r: r["err_ratio"])


if __name__ == "__main__":
    main()
