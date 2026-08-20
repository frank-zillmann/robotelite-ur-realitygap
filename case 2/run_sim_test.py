"""Run every optimized/baseline pair on the robot (or URSim) and report what happened.

    python run_sim_test.py --robot-ip 127.0.0.1

Checks the arm is actually live first, because a UR controller accepts a program on
its socket and silently discards it when it is not -- which looks identical to a
program that ran and did nothing. Then streams each pair, records both, and reports
the measured duration against the one the optimizer predicted.

On URSim the tracking error is meaningless (measured equals commanded there), so it
is not reported. What this does establish is what a model cannot: that the controller
accepts each path, runs it without clamping or faulting, and takes the predicted time.
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np

import record
import send
from utils import DT

MODE = {0: "DISCONNECTED", 1: "CONFIRM_SAFETY", 2: "BOOTING", 3: "POWER_OFF",
        4: "POWER_ON", 5: "IDLE", 6: "BACKDRIVE", 7: "RUNNING", 8: "UPDATING_FIRMWARE"}


def robot_state(host: str) -> tuple[int, int]:
    recipe = record.build_recipe([])
    s = record.open_stream(host, 125.0, recipe)
    try:
        smp = record.read_sample(s, recipe)
        return int(smp["robot_mode"]), int(smp["runtime_state"])
    finally:
        s.close()


def moved(csv: str) -> float:
    """Total joint travel in a recording (rad). 0 means the program never ran."""
    import pandas as pd
    d = pd.read_csv(csv)
    q = d[[f"actual_q{j}" for j in range(6)]].to_numpy()
    return float(np.abs(np.diff(q, axis=0)).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--robot-ip", default="127.0.0.1")
    ap.add_argument("--dir", default="optimized", help="folder of .path pairs")
    ap.add_argument("--loop", type=int, default=5, help="cycles per run")
    ap.add_argument("--out-dir", default="runs_sim", help="where recordings go")
    ap.add_argument("--only", nargs="+", default=None, help="specific path stems")
    args = ap.parse_args()

    mode, rt = robot_state(args.robot_ip)
    print(f"robot_mode = {mode} ({MODE.get(mode, '?')}), runtime_state = {rt}")
    if mode != 7:
        raise SystemExit(
            f"  the arm is {MODE.get(mode, '?')}, not RUNNING.\n"
            f"  Power it on and release the brakes, and enable Remote Control, or the\n"
            f"  controller will take each program and quietly never run it.")

    stems = args.only or sorted({os.path.basename(p).rsplit(".", 2)[0]
                                 for p in glob.glob(f"{args.dir}/*.path")})
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n{len(stems)} pairs, {args.loop} cycles each\n")
    print(f"{'path':18s} {'lane':9s} {'predicted':>10s} {'measured':>9s} {'travel':>8s} {'ok':>4s}")
    rows = []
    for stem in stems:
        got = {}
        for lane in ("baseline", "ppo"):
            src = f"{args.dir}/{stem}.{lane}.path"
            if not os.path.exists(src):
                continue
            n = len(send.load_path(src))
            want = n * DT * args.loop
            out = f"{args.out_dir}/{stem}.{lane}.csv"
            t0 = time.perf_counter()
            _, stop = send.record_path(args.robot_ip, src, out, dt=DT, loop=args.loop)
            wall = time.perf_counter() - t0
            trav = moved(out)
            # "It moved" is not "it ran". A program the controller drops halfway
            # still moves the arm, and reads as a success unless the recording is
            # checked against the length it should have been -- one run here stopped
            # at 28% of its rows and passed a travel-only test.
            rows_want = n * args.loop
            short = len(__import__("pandas").read_csv(out)) < 0.9 * rows_want
            ok = (trav > 1e-3 and "never started" not in str(stop) and not short)
            if short:
                stop = f"truncated: {int(0.008 * rows_want)}s of motion expected"
            got[lane] = (want, wall, trav, ok)
            print(f"{stem if lane == 'baseline' else '':18s} {lane:9s} {want:9.2f}s "
                  f"{wall:8.2f}s {trav:7.3f}r {'yes' if ok else 'NO':>4s}"
                  + ("" if ok else f"   <- {stop}"))
        if len(got) == 2:
            rows.append((stem, got["baseline"], got["ppo"]))

    print()
    if not rows:
        raise SystemExit("nothing ran")
    cuts = [(1 - p[1] / b[1]) * 100 for _, b, p in rows]
    pred = [(1 - p[0] / b[0]) * 100 for _, b, p in rows]
    allok = all(b[3] and p[3] for _, b, p in rows)
    print(f"  {len(rows)} pairs ran, every program executed: {allok}")
    print(f"  cycle-time cut, predicted {np.median(pred):.1f}% -> measured "
          f"{np.median(cuts):.1f}% (median)")
    print(f"  worst |measured - predicted| duration: "
          f"{max(max(abs(b[1]-b[0]), abs(p[1]-p[0])) for _, b, p in rows):.2f} s")
    print("\n  A simulator's measured angle equals its commanded one, so this says the\n"
          "  controller runs these paths at the predicted speed -- not what the real\n"
          "  arm's tracking error will be.")


if __name__ == "__main__":
    main()
