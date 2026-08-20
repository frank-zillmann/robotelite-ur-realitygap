"""Run a Baseline/Optimized script pair back to back on a real robot -- the
actual hardware A/B test, one command.

    python export_manual_scripts.py --agent models/agent_ppo_movej.zip \\
        --recording data/ur5e/T08_slow_r1.csv --model models/distill-ur5e-v2.pkl \\
        --robot UR5e --out-dir manual_scripts
    python run_ab_test.py --folder manual_scripts/T08_slow_r1 --robot-ip 192.168.1.100

Looks for Baseline.script and Optimized.script inside --folder (generate them
first with export_manual_scripts.py --style movej), runs each in turn via
send.py's record_run -- the plain movej/wrap_program path, which has been
reliable in every test today, unlike the servoj-streaming path this used to
go through. Results land in results/hardware_tests/<folder name>/, never next
to the source recording (data/ur5e/ is globbed by
train_distillation_model.py's load_recordings(), so a test result landing
there would silently become training data on the next retrain).
"""
from __future__ import annotations

import argparse
import os

from collect_data import check_robot_mode
from send import record_run


def _run(label: str, host: str, script: str, out: str, loop: int | None):
    """record_run, but fails loudly instead of silently "succeeding" with 0
    samples -- send.py returns a stop_reason string rather than raising, so
    nothing stops a script that only prints it."""
    print(f"running {label} ({script}) -> {out}")
    n, stop = record_run(host, script, out, loop=loop)
    print(f"  {n} samples" + (f"  ({stop})" if stop else ""))
    if stop and "never started" in stop:
        raise SystemExit(
            f"\n{label} never ran -- robot at {host} is not in Remote Control mode "
            f"(or the IP/port is wrong). Fix that on the teach pendant and re-run; "
            f"nothing moved yet, so there is nothing to redo except this.")


def main():
    ap = argparse.ArgumentParser(
        description="Run a Baseline/Optimized script pair back to back on real hardware.")
    ap.add_argument("--folder", required=True,
                    help="folder with Baseline.script + Optimized.script "
                         "(see export_manual_scripts.py)")
    ap.add_argument("--robot-ip", required=True, help="robot/URSim IP, in Remote Control mode")
    ap.add_argument("--loop", type=int, default=None, help="repeat each script N times")
    args = ap.parse_args()

    base_script = os.path.join(args.folder, "Baseline.script")
    opt_script = os.path.join(args.folder, "Optimized.script")
    missing = [p for p in (base_script, opt_script) if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing {missing} -- generate them first with "
                         f"export_manual_scripts.py --style movej --out-dir "
                         f"{os.path.dirname(args.folder) or '.'}")

    print("probing controller...")
    check_robot_mode(args.robot_ip)   # fails fast if not powered on, before doing any work

    name = os.path.basename(os.path.normpath(args.folder))
    out_dir = os.path.join("results", "hardware_tests", name)
    os.makedirs(out_dir, exist_ok=True)

    _run("baseline", args.robot_ip, base_script,
        os.path.join(out_dir, "baseline_result.csv"), args.loop)
    _run("optimized", args.robot_ip, opt_script,
        os.path.join(out_dir, "optimized_result.csv"), args.loop)

    print(f"\ndone. Results in {out_dir}/\n"
         f"  python evaluate.py --model models/distill-ur5e-v2.pkl --data {out_dir}")


if __name__ == "__main__":
    main()
