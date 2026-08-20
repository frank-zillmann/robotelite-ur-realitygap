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
from utils import DT, N_JOINTS


def convert(script: str, robot_ip: str):
    """``(q, dt)``: the commanded trajectory of one cycle of ``script``.

    The script runs twice and only the second pass is kept: it begins where the
    first ended, so it is the cycle in its steady state, closing on itself and
    independent of where the robot started. A single pass would open with the robot
    travelling in from that pose.

    The rate is ``utils.DT`` and the path is written at it, rather than at the
    recorded timestamps: record.py stamps rows with the host clock, which runs long
    under scheduling jitter, while the controller streams on its own and drops a
    client that falls behind rather than skipping rows -- so a run that finished has
    every row, exactly ``DT`` apart.
    """
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        send.record_run(robot_ip, script, tmp.name, hz=1 / DT, loop=2)
        rec = Recording(tmp.name)
    segs = segments(rec)
    if len(segs) < 2:
        raise SystemExit(f"{script}: recorded {len(segs)} moves, need at least 2")
    cycle = segs[-(len(segs) // 2):]                       # the second pass
    q = rec.target_q[cycle[0].i0:cycle[-1].i2]
    print(f"  {len(rec.t)} samples over two passes, {len(segs)} moves, keeping the last "
          f"{len(cycle)}; host clock read {rec.dt * 1000:.2f} ms per row against "
          f"{DT * 1000:.2f} ms")
    if np.abs(q[0] - q[-1]).max() > 1e-3:
        print("  note: it does not end where it starts, so streaming it in a loop jumps")
    return q, DT


def write_path(path: str, q, dt: float):
    """Write joint setpoints plus their step time, the format send.py streams."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"q{j}" for j in range(N_JOINTS)] + ["dt"])
        w.writerows([f"{v:.6f}" for v in [*row, dt]] for row in q)


def write_script(path: str, q, bnd, mult, dt: float = DT, name: str = ""):
    """Write a retimed trajectory as a URScript program of ``movej`` and ``sleep``.

    A ``.path`` has to be streamed: thousands of ``servoj`` rows pushed over a socket,
    which costs upload time, breaks if a program is sent before the last one ended,
    and cannot be loaded on the pendant. A retiming does not need any of that, because
    it never changes the geometry -- each move is still the straight joint-space line
    the original ``movej`` drew, and only its duration differs.

    That conversion is exact rather than approximate. Scaling a trapezoidal speed
    profile in time by ``k`` gives another trapezoid with ``v/k`` and ``a/k**2``, so a
    block the optimizer shortened to ``k`` of its recorded duration is reproduced by
    ``movej(dest, a=a0/k**2, v=v0/k)``; a pause it shortened is ``sleep(t0*k)``. The
    controller then generates the motion itself, at its own control rate.

    ``v0``/``a0`` are measured from the recorded block rather than read from the source
    script, so this works for any recording, including ones whose script is not to hand.
    """
    rows = np.round(np.asarray(bnd) / dt).astype(int).clip(0, len(q) - 1)
    qd = np.gradient(np.asarray(q, float), dt, axis=0)
    qdd = np.gradient(qd, dt, axis=0)
    out = [f"# {name or 'optimized'}: retimed by rl_optimize, geometry unchanged.",
           "#",
           "# Each movej is the same straight line the recording drew; only its speed",
           "# and acceleration differ, and each sleep is the original scaled. Run it",
           "# with send.py --script, or load it on the pendant.",
           ""]
    held = 0.0
    for b, (i0, i1) in enumerate(zip(rows[:-1], rows[1:])):
        k = float(mult[b]) if b < len(mult) else 1.0
        span = np.abs(q[i1] - q[i0]).max()
        if span < 1e-3:                                   # a pause: scale the wait
            held += (i1 - i0) * dt * k
            continue
        if held > 1e-3:
            out.append(f"sleep({held:.4f})")
            held = 0.0
        v0 = float(np.abs(qd[i0:i1 + 1]).max())
        a0 = float(np.abs(qdd[i0:i1 + 1]).max())
        dest = ", ".join(f"{x:.6f}" for x in q[i1])
        out.append(f"movej([{dest}], a={a0 / k ** 2:.4f}, v={v0 / k:.4f})")
    if held > 1e-3:
        out.append(f"sleep({held:.4f})")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    return sum(1 for line in out if line.startswith("movej"))


def main():
    ap = argparse.ArgumentParser(description="Record what a script commands, as a path.")
    ap.add_argument("--script", required=True, help="URScript to run")
    ap.add_argument("--robot-ip", default="127.0.0.1", help="controller address")
    ap.add_argument("--out", default=None, help="default: <script>.path")
    args = ap.parse_args()

    q, dt = convert(args.script, args.robot_ip)
    out = args.out or args.script.rsplit(".", 1)[0] + ".path"
    write_path(out, q, dt)
    print(f"wrote {out}: {len(q)} setpoints, {len(q) * dt:.2f} s at {dt * 1000:.2f} ms")


if __name__ == "__main__":
    main()
