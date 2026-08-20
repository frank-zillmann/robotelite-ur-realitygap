"""Run every baseline/optimized pair on the robot and report what changed.

    python run_tests.py --robot-ip 192.168.1.100

One command: it checks the arm is actually able to run programs, then for each pair
runs the baseline motion and the optimized one, records both, and prints the measured
cycle time and tracking error side by side.

Three things it does that matter, learned the hard way:

- **It waits for the controller between programs.** A UR will not start a program while
  the previous one is still playing, and it does not say so -- it drops the new one and
  ``send.py`` reports "program never started (is the robot in Remote Control mode?)",
  which sends you to the pendant to check something that was never wrong.
- **It checks a run actually finished.** A program the controller drops halfway still
  moves the arm, so "did it move" is not a test. Recordings are compared against the
  length they should have been.
- **It defaults to ``.script``, not ``.path``.** A retimed motion is the same geometry
  at a different speed, so it is a handful of ``movej`` lines rather than a thousand
  streamed setpoints -- no chunking, and the arm approaches the start pose at a
  commanded speed instead of being told to jump there.

Stops on the first failure rather than continuing, so a fault is not buried under
twenty more runs.
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

import contextlib
import io

import record
import send

MODE = {0: "DISCONNECTED", 1: "CONFIRM_SAFETY", 2: "BOOTING", 3: "POWER_OFF",
        4: "POWER_ON", 5: "IDLE", 6: "BACKDRIVE", 7: "RUNNING", 8: "UPDATING_FIRMWARE"}
SETTLE_S = 1.0          # quiet time required before the next program is sent


class State:
    """One RTDE connection, reused and reopened if the controller drops it.

    A sweep used to open a fresh stream for every readiness check -- 49 connections
    across twelve pairs. A controller allows only so many RTDE clients, and a socket
    it has not finished releasing still counts, so a long run could start being
    refused part way through and take the whole sweep down with it. One connection,
    held open, removes that entirely.

    Reconnecting is still worth handling: a controller reboot or a network blip drops
    the stream, and losing an hour of runs to a momentary refusal is not acceptable.
    """

    def __init__(self, host: str, hz: float = 125.0):
        self.host, self.hz = host, hz
        self.recipe = record.build_recipe([])
        self.s = None

    def _stream(self):
        if self.s is None:
            self.s = record.open_stream(self.host, self.hz, self.recipe)
        return self.s

    def sample(self, tries: int = 5):
        for i in range(tries):
            try:
                return record.read_sample(self._stream(), self.recipe)
            except (OSError, ValueError, ConnectionError) as e:
                self.close()
                if i == tries - 1:
                    raise
                print(f"  RTDE dropped ({type(e).__name__}), reconnecting "
                      f"{i + 1}/{tries - 1}...")
                time.sleep(2.0 * (i + 1))
        raise RuntimeError("unreachable")

    def close(self):
        if self.s is not None:
            try:
                self.s.close()
            except Exception:
                pass
            self.s = None

    def mode(self) -> int:
        return int(self.sample()["robot_mode"])

    def wait_idle(self, timeout: float = 20.0) -> bool:
        """Block until no program is running, and none has been for a moment.

        Polling the real state rather than sleeping a guessed interval: a UR drops a
        program sent while another is playing, without saying so, and the error it
        produces then blames Remote Control.
        """
        quiet, t0 = None, time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if int(self.sample()["runtime_state"]) == send.RUNTIME_PLAYING:
                quiet = None
            else:
                quiet = quiet or time.perf_counter()
                if time.perf_counter() - quiet >= SETTLE_S:
                    return True
        return False


def summarize(csv: str, expect_s: float) -> dict:
    """Measured duration, motion and tracking error of one recording."""
    d = pd.read_csv(csv)
    q = d[[f"actual_q{j}" for j in range(6)]].to_numpy()
    t = d[[f"target_q{j}" for j in range(6)]].to_numpy()
    err = np.abs(q - t)
    dur = float(d["t"].iloc[-1])
    return {"seconds": dur, "travel": float(np.abs(np.diff(q, axis=0)).sum()),
            "err_mean": float(err.mean()), "err_max": float(err.max()),
            "short": dur < 0.7 * expect_s}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--robot-ip", default="127.0.0.1")
    ap.add_argument("--hz", type=float, default=125.0,
                    help="RTDE sample rate for the readiness-poll connection; lower "
                         "this if you see repeated 'RTDE dropped ... try a lower "
                         "--hz' messages (the controller drops a client that can't "
                         "keep up, e.g. over a slow/high-latency link)")
    ap.add_argument("--dir", default="optimized", help="folder of pairs")
    ap.add_argument("--form", choices=["script", "path"], default="script")
    ap.add_argument("--loop", type=int, default=5, help="cycles per run")
    ap.add_argument("--out-dir", default="results", help="where recordings go")
    ap.add_argument("--only", nargs="+", default=None, help="specific pairs by name")
    ap.add_argument("--stop-on-fail", action="store_true",
                    help="stop at the first bad run. Off by default: a sweep is long "
                         "and usually unattended, and one bad run is a data point, "
                         "not a reason to discard the remaining twenty.")
    args = ap.parse_args()

    state = State(args.robot_ip, hz=args.hz)
    try:
        mode = state.mode()
    except OSError as e:
        raise SystemExit(
            f"cannot reach the robot's RTDE port at {args.robot_ip}:30004 "
            f"({type(e).__name__}).\n"
            f"  Check the address, that the controller is on, and that nothing else\n"
            f"  already holds an RTDE connection -- a controller allows only a few.")
    print(f"robot at {args.robot_ip}: mode {mode} ({MODE.get(mode, '?')})")
    if mode != 7:
        raise SystemExit(
            f"  the arm is {MODE.get(mode, '?')}, not RUNNING -- power it on and release\n"
            f"  the brakes. Run `python hw_check.py --robot-ip {args.robot_ip}` if it\n"
            f"  looks powered but programs still will not start.")

    ext = args.form
    stems = args.only or sorted({os.path.basename(p).rsplit(".", 2)[0]
                                 for p in glob.glob(f"{args.dir}/*.baseline.{ext}")})
    if not stems:
        raise SystemExit(f"no *.baseline.{ext} files in {args.dir}/")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{len(stems)} pairs as .{ext}, {args.loop} cycles each -> {args.out_dir}/\n")
    print(f"{'pair':16s} {'lane':9s} {'seconds':>8s} {'travel':>8s} "
          f"{'err mean':>9s} {'err max':>8s} {'ok':>4s}")

    tally = os.path.join(args.out_dir, "results.csv")
    if not os.path.exists(tally):
        with open(tally, "w") as f:
            f.write("pair,lane,seconds,travel,err_mean,err_max,ok\n")

    rows, failed = [], []
    for stem in stems:
        got = {}
        for lane in ("baseline", "ppo"):
            src = f"{args.dir}/{stem}.{lane}.{ext}"
            if not os.path.exists(src):
                continue
            out = f"{args.out_dir}/{stem}.{lane}.csv"
            # Every run is isolated. A robot that faults, a controller that drops the
            # connection, a recording that ends mid-write -- each is one bad run to be
            # noted and moved past, not a reason to lose the rest of the sweep.
            try:
                if not state.wait_idle():
                    raise RuntimeError("controller never went idle; a program is still "
                                       "running -- stop it on the pendant")
                kw = dict(host=args.robot_ip, out=out, loop=args.loop, hz=args.hz)
                expect = None
                # send.py streams progress to stdout; it would land in the middle of
                # the results table, so it goes to the log file instead.
                noise = io.StringIO()
                with contextlib.redirect_stdout(noise):
                    if ext == "path":
                        expect = len(send.load_path(src)) * 0.008 * args.loop
                        send.record_path(path=src, **kw)
                    else:
                        send.record_run(script=src, **kw)
                open(os.path.join(args.out_dir, "send.log"), "a").write(noise.getvalue())
                s = summarize(out, expect or 0.0)
                ok = s["travel"] > 1e-3 and not (expect and s["short"])
            except Exception as e:
                print(f"{stem if lane == 'baseline' else '':16s} {lane:9s} "
                      f"{'-':>7s}  {'-':>7s}  {'-':>8s} {'-':>7s}  {'NO':>4s}"
                      f"   {type(e).__name__}: {str(e)[:60]}")
                failed.append(f"{stem}.{lane} ({type(e).__name__})")
                state.close()                     # force a fresh connection next time
                if args.stop_on_fail:
                    break
                continue
            got[lane] = s
            print(f"{stem if lane == 'baseline' else '':16s} {lane:9s} "
                  f"{s['seconds']:7.2f}s {s['travel']:7.2f}r {s['err_mean']*1000:8.4f}m "
                  f"{s['err_max']*1000:7.3f}m {'yes' if ok else 'NO':>4s}")
            # Appended per run, not held to the end: a sweep is long, and a crash on
            # run 20 must not throw away the nineteen that worked.
            with open(tally, "a") as f:
                f.write(f"{stem},{lane},{s['seconds']:.3f},{s['travel']:.4f},"
                        f"{s['err_mean']:.8f},{s['err_max']:.8f},{int(ok)}\n")
            if not ok:
                failed.append(f"{stem}.{lane} (ran short)")
                if args.stop_on_fail:
                    break
        if len(got) == 2:
            rows.append((stem, got["baseline"], got["ppo"]))

    print()
    if not rows:
        raise SystemExit("no complete pairs ran")
    cut = np.array([(1 - p["seconds"] / b["seconds"]) * 100 for _, b, p in rows])
    real = [(b, p) for _, b, p in rows if b["err_mean"] > 1e-9]
    ratio = np.array([p["err_mean"] / b["err_mean"] for b, p in real]) if real else None
    print(f"  {len(rows)} complete pairs out of {len(stems)}, {len(failed)} failed run(s)")
    for f in failed:
        print(f"    - {f}")
    print(f"  cycle time   : {np.median(cut):+.1f}% (median), range "
          f"{cut.min():+.1f}% to {cut.max():+.1f}%")
    if ratio is None:
        print("  tracking error: not measurable -- every baseline run recorded exactly "
              "zero\n                  error, which is a simulator telling you its "
              "measured angle IS\n                  its commanded one. Only a real arm "
              "answers that half.")
    else:
        print(f"  tracking error: x{np.median(ratio):.2f} (median), range "
              f"x{ratio.min():.2f} to x{ratio.max():.2f}")
    print(f"  every run is in {tally}, appended as it finished")
    print(f"\n  per-pair detail:  python analysis.py --csv {args.out_dir}/"
          f"{rows[0][0]}.baseline.csv {args.out_dir}/{rows[0][0]}.ppo.csv "
          f"--model models/distill-ur5e.pkl")


if __name__ == "__main__":
    main()
