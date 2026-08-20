"""Run the recorded baseline and the PPO-optimized version of one recording back
to back on a real robot/URSim, one command -- the actual hardware A/B test.

    python run_ab_test.py --agent models/agent_ppo_movej.zip \\
        --recording data/ur5e/T01_fast_r1.csv --model models/distill-ur5e-v2.pkl \\
        --robot UR5e --robot-ip 10.54.5.147

Writes ``<recording>.baseline.csv`` and ``<recording>.ppo.csv`` (measured, real
RTDE data -- not predictions), and prints the model's *prediction* for both so
you can see how it compares once you look at what actually happened.

Requires the robot/URSim reachable at --robot-ip, in Remote Control mode -- see
send.py's docstring for the details this doesn't repeat.
"""
from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO

from analysis import Recording
from collect_data import check_robot_mode
from common import segments
from convert import write_path
from export_ppo_path import export_path
from send import record_path
from train_distillation_model import DistillModel
from train_ppo import _Move, error
from utils import DT, Robot
import torch


def _run(label: str, host: str, path: str, out: str, dt: float, loop: int | None):
    """record_path, but fails loudly instead of silently "succeeding" with 0
    samples -- send.py returns a stop_reason string rather than raising, so
    nothing stops a script that only prints it (this is exactly what happened:
    the robot wasn't in Remote Control, both runs recorded 0 samples, and the
    script finished normally having moved nothing)."""
    print(f"running {label} -> {out}")
    n, stop = record_path(host, path, out, dt=dt, loop=loop)
    print(f"  {n} samples" + (f"  ({stop})" if stop else ""))
    if stop and "never started" in stop:
        raise SystemExit(
            f"\n{label} never ran -- robot at {host} is not in Remote Control mode "
            f"(or the IP/port is wrong). Fix that on the teach pendant and re-run; "
            f"nothing moved yet, so there is nothing to redo except this.")


def baseline_path(rec: Recording, robot: Robot, move: int | None = None) -> np.ndarray:
    """The recording's own commanded trajectory, move by move -- what
    export_path builds candidates from, unmodified, so the two paths are
    directly comparable (same moves, same order, only the speed differs).

    ``move`` restricts this to one 0-indexed move -- see export_path's
    docstring for why (a whole recording can build a URScript program too
    large for the controller to parse within send.py's start timeout)."""
    segs = [s for s in segments(rec) if s.i1 - s.i0 >= 3]
    if move is not None:
        segs = [segs[move]]
    return np.vstack([_Move(rec, s, robot).baseline_q for s in segs])


def predicted_err(model, q) -> float:
    return float(error(model, torch.as_tensor(np.asarray(q, np.float32))).mean())


def main():
    ap = argparse.ArgumentParser(
        description="A/B test: run the recorded baseline and PPO's version back to back.")
    ap.add_argument("--agent", default="models/agent_ppo_movej.zip", help="trained PPO agent")
    ap.add_argument("--recording", required=True, help="a data/ur5e/*.csv recording")
    ap.add_argument("--model", default="models/distill-ur5e-v2.pkl", help="distilled model pickle")
    ap.add_argument("--robot", default="UR5e", choices=list(Robot.MODELS))
    ap.add_argument("--robot-ip", required=True, help="robot/URSim IP, in Remote Control mode")
    ap.add_argument("--move", type=int, default=None,
                    help="only this 0-indexed move, not the whole recording -- "
                         "keeps the generated URScript program small (recommended)")
    ap.add_argument("--loop", type=int, default=None, help="repeat each path N times")
    args = ap.parse_args()

    print("probing controller...")
    check_robot_mode(args.robot_ip)   # fails fast if not powered on, before doing any work

    agent = PPO.load(args.agent, device="cpu")
    model = DistillModel.load(args.model)
    for p in model.parameters():
        p.requires_grad_(False)
    robot = Robot(args.robot)
    rec = Recording(args.recording)

    q_base = baseline_path(rec, robot, args.move)
    q_ppo = export_path(agent, model, robot, rec, args.move)
    print(f"model's prediction: error {predicted_err(model, q_base) * 1000:.3f} -> "
         f"{predicted_err(model, q_ppo) * 1000:.3f} mrad   "
         f"cycle {len(q_base) * DT:.2f} -> {len(q_ppo) * DT:.2f} s")

    stem = args.recording.rsplit(".", 1)[0]
    base_path, ppo_path = f"{stem}.baseline.path", f"{stem}.ppo.path"
    write_path(base_path, q_base, DT)
    write_path(ppo_path, q_ppo, DT)

    base_out, ppo_out = f"{stem}.baseline.result.csv", f"{stem}.ppo.result.csv"
    _run("baseline", args.robot_ip, base_path, base_out, DT, args.loop)
    _run("PPO-optimized", args.robot_ip, ppo_path, ppo_out, DT, args.loop)

    print(f"\ndone. Score the real measurements with:\n"
         f"  python evaluate.py --model {args.model} --data <folder with both .result.csv>")


if __name__ == "__main__":
    main()
