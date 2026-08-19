"""Record the full data-collection matrix: 11 trajectories x 3 velocities x 2 reps.

Eleven trajectories (T01-T11), three velocities (slow/medium/fast, rad/s --
movej's native units, see utils.VEL_COL), two repetitions each = 66
recordings. T09/T10 are held out: they are never sent to
train_distillation_model.py or train_rla.py, only used later for a final,
honest generalization check (like run.py's held-out script, but decided
before any training instead of after).

    python collect_data.py --list                          # preview the 66 runs, no robot needed
    python collect_data.py --robot-ip 127.0.0.1             # record all 66 against URSim
    python collect_data.py --robot-ip 10.54.5.147           # record against real hardware
    python collect_data.py --robot-ip 10.54.5.147 --only T09 T10   # just the held-out pair

Every run is logged to ``data/manifest.csv`` (trajectory, held_out, velocity
label + rad/s value, rep, output path, sample count, stop reason) so the dataset
is reproducible and the held-out split is explicit and auditable, not just "the
files I remembered not to touch".

Safety: recording against anything other than 127.0.0.1 (i.e. real hardware) is
NOT run automatically. It prints a pre-flight checklist and a connectivity/
Remote-Control probe, then requires ``--yes-i-am-supervising`` (a human at the
robot, hand near the e-stop) before moving anything. Nothing here can see your
cell's physical layout: T05-T11 are new trajectories this session wrote, not
carried over from previously-run scripts, so re-check clearance yourself before
the first real-hardware run of each -- especially T07/T10/T11, which move
wrist3 (a new axis versus the original four scripts), and T11, the only
trainable script where wrist1/wrist2/wrist3 all move at once.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

from record import build_recipe, open_stream
from send import record_run
from utils import load_script, set_param

SCRIPTS_DIR = "scripts"
GENERATED_DIR = os.path.join(SCRIPTS_DIR, "_generated")
DATA_DIR = "data"
HELDOUT_DIR = os.path.join(DATA_DIR, "heldout")
MANIFEST = os.path.join(DATA_DIR, "manifest.csv")

# id, script path, held out (never train on it)
TRAJECTORIES = [
    ("T01", "scripts/horizontal_swing.script", False),
    ("T02", "scripts/shoulder_swing.script", False),
    ("T03", "scripts/triangle.script", False),
    ("T04", "scripts/vertical_swing.script", False),
    ("T05", "scripts/T05_elbow_pump.script", False),
    ("T06", "scripts/T06_wrist_pitch_roll.script", False),
    ("T07", "scripts/T07_wrist3_spin.script", False),
    ("T08", "scripts/T08_reach_extend.script", False),
    ("T09", "scripts/T09_four_point_star.script", True),
    ("T10", "scripts/T10_all_joints_combo.script", True),
    ("T11", "scripts/T11_wrist_combo.script", False),
]

# label, vel (rad/s), acc (rad/s^2). Comfortably below dynamics.MAX_JOINT_SPEED
# (pi rad/s) and MAX_JOINT_ACC (4*pi rad/s^2), and below what most collaborative
# cells configure as a reduced-speed safety limit -- but VERIFY that against your
# own robot's Safety Configuration before running "fast" for real.
VELOCITIES = [
    ("slow", 0.3, 0.6),
    ("medium", 0.6, 1.2),
    ("fast", 1.0, 2.0),
]

REPS = (1, 2)

MANIFEST_HEADER = ["trajectory", "held_out", "script", "velocity_label", "vel_rad_s",
                   "acc_rad_s2", "rep", "robot_ip", "out_path", "n_samples",
                   "stop_reason", "timestamp"]


def plan():
    """The 60 (or fewer, if --only filters) planned runs, in a fixed order."""
    rows = []
    for tid, script, held_out in TRAJECTORIES:
        for label, vel, acc in VELOCITIES:
            for rep in REPS:
                rows.append({
                    "trajectory": tid, "script": script, "held_out": held_out,
                    "velocity_label": label, "vel": vel, "acc": acc, "rep": rep,
                })
    return rows


def generated_script_path(tid: str, label: str) -> str:
    return os.path.join(GENERATED_DIR, f"{tid}_{label}.script")


def write_generated_script(tid: str, script: str, label: str, vel: float, acc: float) -> str:
    """Write ``script`` with vel/acc substituted; returns the path (reused across reps)."""
    os.makedirs(GENERATED_DIR, exist_ok=True)
    text = set_param(set_param(load_script(script), "vel", vel), "acc", acc)
    path = generated_script_path(tid, label)
    with open(path, "w") as f:
        f.write(text)
    return path


def out_path(tid: str, held_out: bool, label: str, rep: int) -> str:
    d = HELDOUT_DIR if held_out else DATA_DIR
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{tid}_{label}_r{rep}.csv")


def append_manifest(row: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    new = not os.path.exists(MANIFEST)
    with open(MANIFEST, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_HEADER)
        if new:
            w.writeheader()
        w.writerow(row)


# --- safety preflight ---------------------------------------------------------

CHECKLIST = """
Pre-flight checklist for {host} (not 127.0.0.1 -- real hardware):

  [ ] Robot is in REMOTE CONTROL mode on the teach pendant (not Local).
  [ ] Someone is physically present with a hand near the e-stop for the whole run.
  [ ] Workspace is clear for the full range of motion of EVERY trajectory below,
      including T05-T11 (new this session, not previously validated on your cell).
  [ ] T06, T10 and T11 move wrist1/wrist2, T07, T10 and T11 move wrist3 -- check
      tool/cable clearance for those axes. T11 is the only trainable script
      that moves all three wrists at once (T10 also does, but is held out).
  [ ] T08, T09 (W3) and T10 straighten the elbow to reach farther than any other
      script -- verify the extended pose doesn't clip the workspace boundary,
      table, or fixtures, and isn't uncomfortably close to a singularity for
      your specific mount.
  [ ] The robot's configured Safety Configuration speed limit is compatible with
      the "fast" velocity (1.0 rad/s ~= 57 deg/s) -- if the safety limit is lower,
      "fast" will just clamp to it and you'll get slow/fast collapsing together
      again, the same failure mode this whole exercise is trying to fix.
  [ ] Enough disk space: ~66 recordings at 125 Hz, each a few seconds, is small
      (single-digit MB total) -- not the hundreds of MB the old test-*.csv were,
      since those looped each script for a long time.

Trajectories in this run: {traj_list}
"""


ROBOT_MODE_RUNNING = 7  # powered on, brakes released, ready to move (see the RTDE guide)
ROBOT_MODE_NAMES = {-1: "NO_CONTROLLER", 0: "DISCONNECTED", 1: "CONFIRM_SAFETY",
                    2: "BOOTING", 3: "POWER_OFF", 4: "POWER_ON", 5: "IDLE",
                    6: "BACKDRIVE", 7: "RUNNING", 8: "UPDATING_FIRMWARE"}


def check_robot_mode(host: str) -> None:
    """Fail fast with a clear message if the robot isn't powered on and ready.

    Both URSim and a real robot boot powered off; movej silently gets ignored
    (the program "never starts") until you power on + release the brakes. This
    catches that in ~1s instead of after 60 x 5s per-run timeouts.
    """
    recipe = build_recipe([])
    try:
        s = open_stream(host, 125.0, recipe)
    except Exception as exc:
        raise SystemExit(f"could not open RTDE stream to {host}: {exc}")
    try:
        from record import read_sample
        smp = read_sample(s, recipe)
    finally:
        s.close()
    mode = smp["robot_mode"]
    if mode != ROBOT_MODE_RUNNING:
        name = ROBOT_MODE_NAMES.get(mode, str(mode))
        raise SystemExit(
            f"Robot at {host} is not powered on (robot_mode={mode} {name}, need "
            f"{ROBOT_MODE_RUNNING} RUNNING). Open the PolyScope X UI "
            f"(http://localhost for URSim, or the robot's own IP) and power on + "
            f"release the brakes, then re-run.")
    print(f"  {host}: robot_mode RUNNING, ready.")


def preflight(host: str, rows: list[dict], yes: bool) -> None:
    if host not in ("127.0.0.1", "localhost"):
        traj_list = ", ".join(sorted({r["trajectory"] for r in rows}))
        print(CHECKLIST.format(host=host, traj_list=traj_list))
        if not yes:
            raise SystemExit(
                "Refusing to move real hardware without --yes-i-am-supervising "
                "(pass it once you've actually gone through the checklist above).")
    print("probing controller...")
    check_robot_mode(host)


# --- main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-ip", default="127.0.0.1",
                    help="UR controller IP (default 127.0.0.1, i.e. local URSim)")
    ap.add_argument("--only", nargs="+", default=None, metavar="Txx",
                    help="only these trajectory ids, e.g. --only T09 T10")
    ap.add_argument("--cycles", type=int, default=3,
                    help="send.py --loop for each recording, i.e. movement "
                         "cycles per CSV (default 3; separate from --reps, "
                         "which are independent recording sessions)")
    ap.add_argument("--hz", type=float, default=125.0, help="RTDE sample rate")
    ap.add_argument("--list", action="store_true",
                    help="print the planned runs and exit, no robot needed")
    ap.add_argument("--yes-i-am-supervising", dest="yes", action="store_true",
                    help="required to record against anything but 127.0.0.1")
    args = ap.parse_args()

    rows = plan()
    if args.only:
        wanted = {t.upper() for t in args.only}
        rows = [r for r in rows if r["trajectory"] in wanted]
        if not rows:
            raise SystemExit(f"--only matched nothing (have: "
                             f"{', '.join(t for t, _, _ in TRAJECTORIES)})")

    if args.list:
        for r in rows:
            tag = "HELD OUT" if next(h for t, _, h in TRAJECTORIES if t == r["trajectory"]) else ""
            print(f"{r['trajectory']:4s} {r['velocity_label']:6s} rep{r['rep']}  "
                  f"vel={r['vel']:.2f} acc={r['acc']:.2f}  {tag}")
        print(f"\n{len(rows)} runs planned.")
        return

    preflight(args.robot_ip, rows, args.yes)

    held_out_by_id = {t: h for t, _, h in TRAJECTORIES}

    print(f"recording {len(rows)} runs to {DATA_DIR}/ (held-out trajectories to "
          f"{HELDOUT_DIR}/) against {args.robot_ip}\n")
    for i, r in enumerate(rows, 1):
        tid, script, held_out = r["trajectory"], r["script"], held_out_by_id[r["trajectory"]]
        gen = write_generated_script(tid, script, r["velocity_label"], r["vel"], r["acc"])
        out = out_path(tid, held_out, r["velocity_label"], r["rep"])
        print(f"[{i}/{len(rows)}] {tid} {r['velocity_label']} rep{r['rep']} "
              f"(vel={r['vel']:.2f} acc={r['acc']:.2f} rad/s) -> {out}")
        n, stop = record_run(args.robot_ip, gen, out, hz=args.hz, loop=args.cycles)
        print(f"  {n} samples" + (f"  ({stop})" if stop else ""))
        append_manifest({
            "trajectory": tid, "held_out": held_out, "script": script,
            "velocity_label": r["velocity_label"], "vel_rad_s": r["vel"],
            "acc_rad_s2": r["acc"], "rep": r["rep"], "robot_ip": args.robot_ip,
            "out_path": out, "n_samples": n, "stop_reason": stop or "",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        if stop and "never started" in stop:
            raise SystemExit(
                "\nProgram never started -- robot is not in Remote Control mode "
                "(or the IP/port is wrong). Fix that and re-run; already-recorded "
                "runs are logged in the manifest, so you won't have to redo them "
                "if you re-run the same --only selection minus what's done.")

    print(f"\ndone. {len(rows)} runs recorded, manifest at {MANIFEST}")


if __name__ == "__main__":
    main()
