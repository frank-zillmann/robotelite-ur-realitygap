"""Collect a speed sweep: run each motion at many vel/acc values and record it.

The recorded runs in ``data/`` were all taken at the same speed. Their scripts ask
for ``vel = 100`` down to ``vel = 20``, but a URScript ``movej`` reads ``v`` in
**rad/s**, not deg/s, and the controller silently clamps anything above the joint
limit (~2.09 rad/s on base/shoulder/elbow, ~3.14 on the wrists). Every one of
those numbers is far above the limit, so every run came out at maximum speed:
``target_qd`` peaks at exactly 2.094 / 3.142 in every run while the logged ``vel``
register reads 20, 50, 100. A gap model fitted on that sees a ``vel`` column that
varies and a trajectory that does not, so it can only learn that speed does not
matter.

This script fixes the collection. It sweeps ``vel``/``acc`` over values that are
actually below the limit, runs each combination on the robot, and records it:

    python collect.py --robot-ip 127.0.0.1                 # the default sweep
    python collect.py --dry-run                            # plan + time estimate
    python collect.py --scripts scripts/short_moves.script --acc 0.4 0.8 1.6 3.2

One CSV per (script, vel, acc) lands in ``--out-dir``, and ``--pool`` concatenates
them into one file for ``train_distillation_model.py``. Every row is tagged with
its source run (``utils.SCRIPT_COL``) so ``common.segments`` never runs a segment
across two runs. Existing CSVs are skipped unless ``--overwrite`` is given, so an
interrupted sweep resumes where it stopped.

Requires the robot (or URSim) in Remote Control mode, as ``send.py`` does.
"""
from __future__ import annotations

import argparse
import os
import re
import time

import numpy as np
import pandas as pd

import send
from dynamics import trapezoidal
from record import build_recipe, parse_registers
from utils import ACC_COL, SCRIPT_COL, VEL_COL, load_script, set_param

# --- sweep grid ---------------------------------------------------------------
# rad/s and rad/s^2, the units URScript movej actually takes. The vel grid stays
# well under the 3.1416 rad/s joint limit so no run clamps (there's more headroom
# above 1.8 on a UR5e than there was under the UR10e's 2.094 rad/s limit, if you
# want to widen it); below it the values are spaced roughly geometrically, because
# the gap grows non-linearly with speed and an even grid wastes samples at the
# fast end.
DEFAULT_VEL = (0.3, 0.6, 1.0, 1.4, 1.8)
DEFAULT_ACC = (0.5, 1.5, 3.0)

DEFAULT_SCRIPTS = (
    "scripts/workspace_sweep.script",   # reach x height grid, all six joints
    "scripts/reach_extend.script",      # extended <-> folded, inertia contrast
    "scripts/wrist_sweep.script",       # the three wrist joints on their own
    "scripts/short_moves.script",       # triangular profiles, acc-dominated
)

# UR5e joint limits. movej clamps `v` and `a` to these, and a clamped run
# records the same trajectory as every other clamped run.
MAX_VEL = np.pi         # rad/s,   180 deg/s on every joint (uniform on the UR5e,
                        # unlike the UR10e's 120/180 split)
MAX_ACC = 10.0          # rad/s^2, a conservative ceiling for a movej

SETTLE_S = 2.0          # pause between runs, so the arm is at rest before the next


# --- planning -----------------------------------------------------------------

_POSE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*\[([^\]]*)\]")
_MOVEJ = re.compile(r"^\s*movej\(\s*([A-Za-z_]\w*)")
_SLEEP = re.compile(r"^\s*sleep\(\s*([0-9.]+)")


def script_moves(text: str) -> tuple[list[float], float]:
    """Per-move joint travel (rad) and total sleep time (s) of a flat script.

    Reads the ``NAME = [...]`` pose assignments and the ``movej(NAME, ...)`` order,
    and returns the widest joint travel of each move -- the distance that sets the
    move's duration, the same one ``common.Segment.dist`` reports. Used only to
    estimate how long a sweep will take; a script whose poses this cannot parse
    (an exported PolyScope program, say) simply returns no moves.
    """
    poses, order, dwell = {}, [], 0.0
    for line in text.splitlines():
        if line.strip().startswith("#"):
            continue
        m = _POSE.match(line)
        if m:
            try:
                poses[m.group(1)] = [float(v) for v in m.group(2).split(",")]
            except ValueError:
                pass
            continue
        m = _MOVEJ.match(line)
        if m and m.group(1) in poses:
            order.append(np.asarray(poses[m.group(1)], dtype=float))
            continue
        m = _SLEEP.match(line)
        if m:
            dwell += float(m.group(1))
    travel = [float(np.abs(order[i + 1] - order[i]).max()) for i in range(len(order) - 1)]
    return travel, dwell


def estimate_seconds(text: str, vel: float, acc: float, loop: int, dt: float = 0.008) -> float:
    """Rough wall-clock time of one recorded run, from the trapezoidal profile.

    Sums ``dynamics.trapezoidal`` over the script's moves at this vel/acc and adds
    the sleeps. It ignores the move the robot makes to reach the first pose (it
    starts from wherever it is), so the real run is a little longer.
    """
    travel, dwell = script_moves(text)
    if not travel:
        return float("nan")
    per_cycle = sum(len(trapezoidal(d, vel, acc, dt)) * dt for d in travel) + dwell
    return per_cycle * max(loop or 1, 1)


# --- collection ---------------------------------------------------------------

def run_name(script: str, vel: float, acc: float) -> str:
    """Stable identifier for one (script, vel, acc) run, used as file and tag."""
    stem = os.path.splitext(os.path.basename(script))[0]
    return f"{stem}_v{vel:g}_a{acc:g}"


def trim_stale(out_csv: str, vel: float, acc: float) -> int:
    """Drop leading rows recorded before the program set the vel/acc registers.

    The recorder opens the RTDE stream before the program starts, and an output
    float register keeps whatever the last run left in it, so the first samples of
    a run can carry the *previous* run's vel/acc. That is a handful of stationary
    rows, but they are rows the gap model would read as "this speed produced this
    current". Since the commanded values are known exactly, drop every row before
    the registers first agree with them. Returns the number of rows dropped.
    """
    df = pd.read_csv(out_csv)
    ok = np.flatnonzero((df[VEL_COL].to_numpy() == vel) & (df[ACC_COL].to_numpy() == acc))
    if len(ok) == 0:
        return 0                                  # registers never matched: leave it alone
    n = int(ok[0])
    if n:
        df.iloc[n:].to_csv(out_csv, index=False)
    return n


def collect_one(robot_ip: str, script: str, vel: float, acc: float, out_csv: str,
                loop: int = None, hz: float = 125.0, port: int = send.SCRIPT_PORT):
    """Run one script at one (vel, acc) on the robot and record it to ``out_csv``.

    Rewrites the script's ``vel``/``acc`` lines in memory (no temp file), wraps it
    with ``send.wrap_program`` and records through ``send.run_and_record``, so this
    takes exactly the same path to the controller as ``send.py --script``. The
    ``vel``/``acc`` output float registers carry the commanded rad/s and rad/s^2
    into the CSV, which is what makes the sweep visible to the gap model.
    """
    text = set_param(set_param(load_script(script), "vel", vel), "acc", acc)
    program = send.wrap_program(text, loop)
    recipe = build_recipe(parse_registers(["1", "vel", "2", "acc"]) + [(send.DONE_REG, "_done")])
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    n, stop = send.run_and_record(robot_ip, program, out_csv, hz, recipe, port)
    if n:
        n -= trim_stale(out_csv, vel, acc)
    return n, stop


def verify(out_csv: str, vel: float) -> str:
    """One-line check that the run actually moved at the speed it was told to.

    Compares the fastest commanded joint speed in the recording against the
    commanded ``vel``. A run at the joint limit reports ``CLAMPED``: the controller
    ignored ``vel``, and that row is another copy of the max-speed data the
    existing recordings already have.
    """
    df = pd.read_csv(out_csv, usecols=[f"target_qd{j}" for j in range(6)])
    peak = float(np.abs(df.to_numpy()).max())
    flag = "CLAMPED" if peak > MAX_VEL - 1e-3 else "ok"
    return f"peak |target_qd| {peak:.3f} rad/s (asked {vel:g}) [{flag}]"


def pool(csvs: list[tuple[str, str]], out: str):
    """Concatenate recorded runs into one CSV, tagging each row with its run.

    ``csvs`` is ``[(run_name, path)]``. The tag goes in ``utils.SCRIPT_COL``, which
    ``common.segments`` uses to split the pooled file into per-run blocks before it
    looks at ``script_control_line`` -- without it, two runs of the same script
    would blend into one segment at the seam, since their line numbers match.
    """
    frames = []
    for name, path in csvs:
        raw = pd.read_csv(path)
        frames.append(pd.concat(
            [raw, pd.Series(name, index=raw.index, name=SCRIPT_COL)], axis=1))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out, index=False)
    print(f"pooled {len(frames)} runs, {len(df)} rows -> {out}")
    return df


def main():
    ap = argparse.ArgumentParser(
        description="Collect a vel/acc sweep over several motions and record each run.")
    ap.add_argument("--robot-ip", default=os.environ.get("UR_HOST", "127.0.0.1"),
                    help="UR controller IP (default: $UR_HOST or 127.0.0.1)")
    ap.add_argument("--scripts", nargs="+", default=list(DEFAULT_SCRIPTS),
                    help="URScript motions to sweep (default: the four coverage scripts)")
    ap.add_argument("--vel", nargs="+", type=float, default=list(DEFAULT_VEL),
                    help=f"movej speeds in rad/s (default: {' '.join(map(str, DEFAULT_VEL))})")
    ap.add_argument("--acc", nargs="+", type=float, default=list(DEFAULT_ACC),
                    help=f"movej accelerations in rad/s^2 (default: {' '.join(map(str, DEFAULT_ACC))})")
    ap.add_argument("--loop", type=int, default=1,
                    help="repeat each motion N times per run (default 1)")
    ap.add_argument("--out-dir", default="data/sweep", help="where the run CSVs go")
    ap.add_argument("--pool", default=None,
                    help="also write one pooled CSV (default: <out-dir>/pooled.csv)")
    ap.add_argument("--hz", type=float, default=125.0, help="sample rate (default 125)")
    ap.add_argument("--port", type=int, default=send.SCRIPT_PORT,
                    help="script interface port (30002 secondary, 30001 primary)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-record runs whose CSV already exists (default: skip them)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the time estimate, record nothing")
    args = ap.parse_args()
    pool_path = args.pool or os.path.join(args.out_dir, "pooled.csv")

    # Clamped runs are the bug this script exists to avoid, so refuse them up front.
    over_v = [v for v in args.vel if v > MAX_VEL]
    over_a = [a for a in args.acc if a > MAX_ACC]
    if over_v or over_a:
        raise SystemExit(
            f"vel {over_v} / acc {over_a} exceed the joint limits "
            f"({MAX_VEL:.3f} rad/s, {MAX_ACC:g} rad/s^2). URScript takes rad/s, not "
            "deg/s, and the controller clamps: those runs would all record at the "
            "same speed, which is exactly what the existing data already has.")

    plan = [(s, v, a) for s in args.scripts for v in args.vel for a in args.acc]
    texts = {s: load_script(s) for s in args.scripts}
    total = sum(estimate_seconds(texts[s], v, a, args.loop) for s, v, a in plan)
    print(f"{len(plan)} runs = {len(args.scripts)} scripts x {len(args.vel)} vel "
          f"x {len(args.acc)} acc, loop {args.loop}")
    print(f"estimated motion time {total / 60:.1f} min "
          f"(+ ~{len(plan) * SETTLE_S / 60:.1f} min of settle pauses)")

    if args.dry_run:
        for s, v, a in plan:
            est = estimate_seconds(texts[s], v, a, args.loop)
            print(f"  {run_name(s, v, a):40s} vel {v:4.2f} acc {a:4.2f}  ~{est:5.1f}s")
        return

    done, t0 = [], time.monotonic()
    for k, (s, v, a) in enumerate(plan, 1):
        name = run_name(s, v, a)
        out_csv = os.path.join(args.out_dir, name + ".csv")
        if os.path.exists(out_csv) and not args.overwrite:
            print(f"[{k}/{len(plan)}] {name}: exists, skipping")
            done.append((name, out_csv))
            continue
        print(f"[{k}/{len(plan)}] {name}: vel {v:g} rad/s  acc {a:g} rad/s^2")
        n, stop = collect_one(args.robot_ip, s, v, a, out_csv, args.loop,
                              args.hz, args.port)
        print(f"\n  {n} samples -> {out_csv}  ({stop})")
        if n:
            print("  " + verify(out_csv, v))
            done.append((name, out_csv))
        time.sleep(SETTLE_S)

    print(f"\ncollected {len(done)}/{len(plan)} runs in {(time.monotonic() - t0) / 60:.1f} min")
    if done:
        pool(done, pool_path)
        print(f"  train on it: python train_distillation_model.py --csvs {pool_path} "
              "--out models/distill.pkl")


if __name__ == "__main__":
    main()
