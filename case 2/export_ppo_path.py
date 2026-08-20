"""Turn a trained PPO policy's choices into a runnable path -- closes the loop
from "the distill model predicts this is better" to "a real robot measured it".

Everything train_ppo.py reports is scored against the distill model's
*prediction*, never a real measurement. This replays a trained agent's chosen
speed profile for every move of one recording, in order, and stitches them into
one servoj path -- the same format convert.py/optimize.py write, so send.py can
stream it to a real UR5e or URSim exactly like an optimize.py result:

    python export_ppo_path.py --agent models/agent_ppo.zip \\
        --recording data/ur5e/T01_fast_r1.csv --model models/distill-ur5e-v2.pkl \\
        --robot UR5e --out T01_fast_r1.ppo.path

    python send.py --path T01_fast_r1.ppo.path --robot-ip <ip> --out result.csv
    python evaluate.py --model models/distill-ur5e-v2.pkl --data <folder with result.csv>

The last step is the actual point: it scores the *real* recorded result the
same way the held-out check does, so you can see whether the model's prediction
(what train_ppo.py optimized against) matches what the robot really did.
"""
from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO

from analysis import Recording
from common import segments
from convert import write_path
from train_distillation_model import DistillModel
from train_ppo import _Move, candidate_q, obs_of, unmap_action
from utils import DT, Robot


def export_path(agent: PPO, model: DistillModel, robot: Robot, rec: Recording,
                move: int | None = None) -> np.ndarray:
    """Stitch the agent's chosen candidate for every move of ``rec``, in order.

    Mirrors ``_Move``/``MoveEnv``'s obs/action/candidate mapping exactly (via the
    free functions ``train_ppo.py`` exposes for this), so this is precisely what
    the trained policy would pick if it saw these moves during training/eval --
    not a re-derivation, a replay.

    ``move``, if given, restricts this to just that one 0-indexed move instead
    of the whole recording -- ``wrap_path`` unrolls one servoj line per row, so
    a whole multi-move recording (thousands of rows) can build a URScript
    program large enough that the controller never finishes parsing it within
    send.py's start timeout, which looks exactly like a Remote Control problem
    but isn't one. One move keeps the program small.
    """
    segs = [s for s in segments(rec) if s.i1 - s.i0 >= 3]
    if move is not None:
        if not (0 <= move < len(segs)):
            raise SystemExit(f"{rec.path} has {len(segs)} usable moves, --move must be in [0, {len(segs) - 1}]")
        segs = [segs[move]]
    rows = []
    for s in segs:
        m = _Move(rec, s, robot)
        action, _ = agent.predict(obs_of(m), deterministic=True)
        vel, acc = unmap_action(action)
        rows.append(candidate_q(m, robot, vel, acc))
    if not rows:
        raise SystemExit(f"{rec.path}: no usable moves")
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Export a trained PPO policy's chosen path for one recording.")
    ap.add_argument("--agent", default="models/agent_ppo.zip", help="trained PPO agent")
    ap.add_argument("--recording", required=True, help="a data/ur5e/*.csv recording")
    ap.add_argument("--model", default="models/distill-ur5e-v2.pkl", help="distilled model pickle")
    ap.add_argument("--robot", default="UR5e", choices=list(Robot.MODELS))
    ap.add_argument("--move", type=int, default=None,
                    help="only this 0-indexed move, not the whole recording -- "
                         "keeps the generated URScript program small (see export_path)")
    ap.add_argument("--out", default=None, help="default: <recording>.ppo.path")
    args = ap.parse_args()

    agent = PPO.load(args.agent, device="cpu")
    model = DistillModel.load(args.model)
    robot = Robot(args.robot)
    rec = Recording(args.recording)

    q = export_path(agent, model, robot, rec, args.move)
    out = args.out or args.recording.rsplit(".", 1)[0] + ".ppo.path"
    write_path(out, q, DT)
    print(f"wrote {out} ({len(q)} setpoints, {len(q) * DT:.2f} s)")
    print(f"  run it:   python send.py --path {out} --robot-ip <ip> --out result.csv")
    print(f"  score it: python evaluate.py --model {args.model} --data <folder with result.csv>")


if __name__ == "__main__":
    main()
