"""Run a URScript on the controller and keep the trajectory it commanded, as a path.

The controller is the only thing that knows exactly how it turns a script into
motion -- the speed profile, the Cartesian caps, the pauses, how one move blends
into the next. So instead of reimplementing that, run the script on URSim and
record it: ``target_q`` is the commanded trajectory, sampled at the RTDE rate.

    python convert.py --script scripts/triangle.script --out scripts/triangle.path

The result is a servoj path (``q0..q5`` plus a per-row ``dt``) that reproduces the
script's own motion, and is what optimize.py starts from. The robot must be on and
in Remote Control; nothing is optimized here, this only observes.
"""
from __future__ import annotations

import argparse
import csv
import tempfile

import numpy as np

import send
from analysis import Recording
from common import segments
from utils import N_JOINTS


def convert(script: str, robot_ip: str = "127.0.0.1", hz: float = 125.0):
    """``(q, dt)``: the commanded trajectory of one pass of ``script``.

    The first segment is dropped -- it is the robot travelling from wherever it
    happened to be to the script's first waypoint, which is not part of the cycle.
    What is kept runs from the second move to the end of the last one's pause, so
    streaming it in a loop repeats the same cycle the script does.
    """
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        send.record_run(robot_ip, script, tmp.name, hz=hz)
        rec = Recording(tmp.name)
    segs = segments(rec)
    if len(segs) < 2:
        raise SystemExit(f"{script}: recorded {len(segs)} moves, need at least 2")
    return rec.target_q[segs[1].i0:segs[-1].i2], rec.dt


def write_path(path: str, q, dt: float):
    """Write joint setpoints plus their step time, the format send.py streams."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"q{j}" for j in range(N_JOINTS)] + ["dt"])
        w.writerows([f"{v:.6f}" for v in [*row, dt]] for row in q)


def main():
    ap = argparse.ArgumentParser(description="Record what a script commands, as a path.")
    ap.add_argument("--script", default="scripts/triangle.script", help="URScript to run")
    ap.add_argument("--out", default=None, help="path CSV (default: <script>.path)")
    ap.add_argument("--robot-ip", default="127.0.0.1", help="controller address")
    ap.add_argument("--hz", type=float, default=125.0, help="RTDE sample rate")
    args = ap.parse_args()

    q, dt = convert(args.script, args.robot_ip, args.hz)
    out = args.out or args.script.rsplit(".", 1)[0] + ".path"
    write_path(out, q, dt)
    print(f"wrote {out}: {len(q)} setpoints, {len(q) * dt:.2f} s at {dt * 1000:.2f} ms")


if __name__ == "__main__":
    main()
