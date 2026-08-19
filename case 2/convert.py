"""Run a URScript on the controller and keep the trajectory it commanded, as a path.

Only the controller knows exactly how it turns a script into motion: the speed
profile, the Cartesian caps, the pauses, the blends. So rather than reimplement it,
run the script and record ``target_q``.

    python convert.py --script scripts/triangle.script --out scripts/triangle.path

Out comes a servoj path (``q0..q5`` plus a per-row ``dt``) that reproduces the
script and closes on itself, so it can be streamed in a loop. The robot must be on
and in Remote Control; nothing is optimized here.
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
    """``(q, dt)``: the commanded trajectory of one cycle of ``script``.

    The script runs twice and only the second pass is kept: it begins where the
    first ended, so it is the cycle in its steady state, closing on itself and
    independent of where the robot started. A single pass would open with the robot
    travelling in from that pose.

    ``dt`` is the nominal ``1/hz``, not the recorded timestamps: record.py stamps
    rows with the host clock, which runs long under scheduling jitter, while the
    controller streams on its own and drops a client that falls behind rather than
    skipping rows -- so a run that finished has every row.
    """
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        send.record_run(robot_ip, script, tmp.name, hz=hz, loop=2)
        rec = Recording(tmp.name)
    segs = segments(rec)
    if len(segs) < 2:
        raise SystemExit(f"{script}: recorded {len(segs)} moves, need at least 2")
    cycle = segs[-(len(segs) // 2):]                       # the second pass
    q = rec.target_q[cycle[0].i0:cycle[-1].i2]
    print(f"  {len(rec.t)} samples over two passes, {len(segs)} moves, keeping the last "
          f"{len(cycle)}; host clock read {rec.dt * 1000:.2f} ms per row against "
          f"{1000 / hz:.2f} ms")
    if np.abs(q[0] - q[-1]).max() > 1e-3:
        print("  note: it does not end where it starts, so streaming it in a loop jumps")
    return q, 1.0 / hz


def write_path(path: str, q, dt: float):
    """Write joint setpoints plus their step time, the format send.py streams."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"q{j}" for j in range(N_JOINTS)] + ["dt"])
        w.writerows([f"{v:.6f}" for v in [*row, dt]] for row in q)


def main():
    ap = argparse.ArgumentParser(description="Record what a script commands, as a path.")
    ap.add_argument("--script", default="scripts/triangle.script", help="URScript to run")
    ap.add_argument("--out", default=None, help="default: <script>.path")
    ap.add_argument("--robot-ip", default="127.0.0.1", help="controller address")
    args = ap.parse_args()

    q, dt = convert(args.script, args.robot_ip)
    out = args.out or args.script.rsplit(".", 1)[0] + ".path"
    write_path(out, q, dt)
    print(f"wrote {out}: {len(q)} setpoints, {len(q) * dt:.2f} s at {dt * 1000:.2f} ms")


if __name__ == "__main__":
    main()
