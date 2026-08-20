"""Find out why a program will not start on the robot, rather than guessing.

    python hw_check.py --robot-ip 192.168.1.100

``send.py`` reports "program never started (is the robot in Remote Control mode?)"
whenever ``runtime_state`` fails to reach PLAYING within 5 s. That is one condition
with several possible causes, and the message names only the most common one. If some
programs run and a later one does not, Remote Control is demonstrably fine and the
message is misleading.

This separates them. It sends a program that moves nothing -- it flips the done
register, waits, and ends -- and watches the state transitions:

- it never reaches PLAYING          -> the controller is refusing programs:
                                       Remote Control off, wrong IP/port, or the arm
                                       is not powered and brakes released
- it plays and finishes             -> the controller accepts programs, so a failure
                                       on a *path* is about that path: its size, or
                                       sending the next one before this one ended
- it plays and then aborts          -> a runtime fault; check the pendant's log

Nothing here commands motion, so it is safe to run with the arm live.
"""
from __future__ import annotations

import argparse
import socket
import time

import record
import send

MODE = {0: "DISCONNECTED", 1: "CONFIRM_SAFETY", 2: "BOOTING", 3: "POWER_OFF",
        4: "POWER_ON", 5: "IDLE", 6: "BACKDRIVE", 7: "RUNNING", 8: "UPDATING_FIRMWARE"}
RUNTIME = {0: "STOPPING", 1: "STOPPED", 2: "PLAYING", 3: "PAUSING", 4: "PAUSED",
           5: "RESUMING"}
PROBE = ("def prog():\n"
         "  write_output_float_register(3, 0)\n"
         "  sleep(0.5)\n"
         "  write_output_float_register(3, 1)\n"
         "end\n")


def watch(host: str, seconds: float, hz: float = 125.0, send_after: str = None):
    """Stream state for ``seconds``, optionally sending a program once streaming."""
    recipe = record.build_recipe([(send.DONE_REG, "done")])
    s = record.open_stream(host, hz, recipe)
    seen, t0, sent = [], time.perf_counter(), False
    try:
        while time.perf_counter() - t0 < seconds:
            smp = record.read_sample(s, recipe)
            row = (int(smp["robot_mode"]), int(smp["runtime_state"]),
                   float(smp[send.DONE_FIELD]))
            if not seen or seen[-1][1:] != row:
                seen.append((time.perf_counter() - t0, *row))
            if send_after and not sent and time.perf_counter() - t0 > 0.5:
                send.send_program(host, send_after)
                sent = True
    finally:
        s.close()
    return seen


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--robot-ip", required=True)
    ap.add_argument("--port", type=int, default=send.SCRIPT_PORT)
    args = ap.parse_args()

    print(f"robot at {args.robot_ip}:{args.port}\n")
    for p in (args.port, 30004):
        c = socket.socket(); c.settimeout(3)
        try:
            c.connect((args.robot_ip, p)); print(f"  port {p:5d}: open")
        except Exception as e:
            print(f"  port {p:5d}: {type(e).__name__} -- nothing is listening")
            raise SystemExit("  the address or port is wrong; nothing else can work")
        finally:
            c.close()

    print("\nstate before sending anything:")
    for t, m, r, f in watch(args.robot_ip, 2.0):
        print(f"  {t:5.2f}s  robot_mode={m} ({MODE.get(m, '?')})  "
              f"runtime={r} ({RUNTIME.get(r, '?')})  done_reg={f:.0f}")

    print("\nsending a program that moves nothing...")
    seen = watch(args.robot_ip, 8.0, send_after=PROBE)
    for t, m, r, f in seen:
        print(f"  {t:5.2f}s  robot_mode={m} ({MODE.get(m, '?')})  "
              f"runtime={r} ({RUNTIME.get(r, '?')})  done_reg={f:.0f}")

    played = any(r == send.RUNTIME_PLAYING for _, _, r, _ in seen)
    ended = any(f >= 0.5 for _, _, _, f in seen[1:])
    mode = seen[-1][1]
    print()
    if mode != 7:
        print(f"  VERDICT: the arm is {MODE.get(mode, '?')}, not RUNNING. Power it on and\n"
              f"  release the brakes; nothing will execute until then.")
    elif not played:
        print("  VERDICT: the controller never started a program it cannot refuse for\n"
              "  size or timing reasons, so it is refusing programs outright.\n"
              "  That is Remote Control being off (or a firewall on 30002).")
    elif played and ended:
        print("  VERDICT: the controller accepts and completes programs. Remote Control\n"
              "  is ON, and the message you saw on the real path is misleading.\n"
              "  Look instead at that path's size, or at whether the previous chunk had\n"
              "  finished -- a UR will not start a program while one is still running.")
    else:
        print("  VERDICT: the program started but never signalled completion. Check the\n"
              "  pendant log; something aborted it mid-run.")


if __name__ == "__main__":
    main()
