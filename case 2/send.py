"""Send a URScript file to a UR controller (or URSim) and record the run.

Opens the RTDE stream, pushes the script to the secondary client interface
(port 30002) so the controller runs it, logs every sample to a CSV, and stops
when the program finishes.

    python send.py scripts/shoulder_swing.script --robot-ip 127.0.0.1

--loop N repeats the motion N times then ends; without it the script runs once:

    python send.py scripts/shoulder_swing.script --loop 10


A ``.path`` argument streams a servoj path (a CSV of joint setpoints, optionally
with a per-row servoj time as a 7th column) instead of running a script, via
one of two required ``--engine`` choices (the output CSV is named after it,
e.g. ``shoulder_swing.batch.csv``, so the two never overwrite each other):

    python send.py scripts/shoulder_swing.path --engine batch     # URSim
    python send.py scripts/shoulder_swing.path --engine stream    # real hardware

"batch" embeds the whole path as one program, same as a ``.script`` run --
simple, but a real controller silently drops any program over ~30 KB of text.
"stream" streams it live via the ``ur_rtde`` package instead (only imported
for this engine); see ``record_path``.

Requires Remote Control mode on the robot/URSim, powered on with brakes
released, and no active protective/safety stop; otherwise the controller
accepts the connection but does not run anything.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import socket

from record import build_recipe, open_stream, parse_registers, record_stream
from utils import DT, load_script

SCRIPT_PORT = 30002       # UR secondary client interface: accepts URScript programs
DASHBOARD_PORT = 29999    # UR dashboard server: plain-text status queries
RUNTIME_PLAYING = 2       # RTDE runtime_state while a program is running
START_TIMEOUT_S = 5.0     # how long to wait for the program to start playing
DONE_REG = 3              # output float register we flip to 1 when the program ends
DONE_FIELD = f"output_double_register_{DONE_REG}"
SERVO_LOOKAHEAD = 0.1     # servoj lookahead_time (s): smooths the streamed path
SERVO_GAIN = 300          # servoj gain: how hard it tracks each setpoint
STREAM_HZ = 500.0         # record_path's servoJ tick rate (e-Series native cycle)


def dashboard_query(host: str, cmd: str, port: int = DASHBOARD_PORT) -> str:
    """Send one line to the dashboard server (29999) and return its reply."""
    with socket.create_connection((host, port), timeout=5) as s:
        s.recv(4096)                       # welcome banner
        s.sendall((cmd + "\n").encode())
        return s.recv(4096).decode(errors="replace").strip()


def preflight(host: str) -> None:
    """Print the dashboard states that gate whether a sent program actually runs:
    power/brakes (robotmode), protective stops (safetymode), and Remote Control.
    A script sent to 30002 is accepted and silently dropped if any of these
    isn't right, so a timed-out start is otherwise indistinguishable between them.
    """
    for cmd in ("robotmode", "safetymode", "is in remote control"):
        try:
            print(f"  {cmd}: {dashboard_query(host, cmd)}")
        except OSError as exc:
            print(f"note: dashboard server unreachable ({exc}), skipping preflight check")
            return


def _is_assignment(line: str) -> bool:
    """True for a plain variable assignment `name = ...` (not a call `foo(...)`)."""
    return re.match(r"\s*[A-Za-z_]\w*\s*=", line) is not None


# A `$ N "..."` line is a PolyScope(X) source-listing annotation, not URScript;
# it must be stripped before the text is sent to the controller.
_MARKER = re.compile(r"^\s*\$\s+\d")


def _is_structured(text: str) -> bool:
    """True if the text is a real program, not a flat list of statements.

    The flat lesson scripts are only top-level assignments plus motion calls
    (`movej`/`sleep`/register writes). Anything with block structure --
    `def`/`sec`/`thread`, a `global` declaration, a block header ending in `:`,
    an `end`, or a PolyScope `$` marker -- is an exported program and must not
    be hoisted/looped (that reorders `global`s past their use and flattens the
    blocks). Comments are ignored.
    """
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if (_MARKER.match(ln) or s.endswith(":")
                or s == "end" or s.startswith("end ")
                or re.match(r"(def|sec|thread|global)\b", s)):
            return True
    return False


def _wrap_structured(text: str, loop: int | None) -> str:
    """Wrap an already-structured program without reordering it.

    Strips `$` markers, preserves the original block structure and indentation,
    and only adds the done-flag when we own the `def prog(): ... end` wrapper.
    Hoisting and the repeat loop are intentionally skipped: a structured program
    carries its own `global`s and control flow, which the flat-script path would
    corrupt. ``loop`` is not applied here (the program controls its own repeats).
    """
    if loop is not None:
        print("note: --loop ignored for a structured/exported program "
              "(it controls its own repeats); stop with Ctrl-C.")
    body = [ln for ln in text.splitlines() if not _MARKER.match(ln)]
    stripped = "\n".join(body).lstrip()
    if stripped.startswith("def ") or stripped.startswith("sec "):
        # Already a complete program: run as-is. No done-flag (we do not own its
        # `end`), so auto-stop falls back to Ctrl-C / the runtime dropping.
        print("note: complete program sent as-is; no auto-stop, use Ctrl-C.")
        return stripped + ("" if stripped.endswith("\n") else "\n")
    IND = "  "
    out = ["def prog():", IND + f"write_output_float_register({DONE_REG}, 0)"]
    out += [(IND + ln if ln.strip() else "") for ln in body]
    out += [IND + f"write_output_float_register({DONE_REG}, 1)", "end"]
    return "\n".join(out) + "\n"


def wrap_program(text: str, loop: int | None = None) -> str:
    """Wrap the script's statements in a `def prog(): ... end` program.

    The UR script interface runs a program, not loose top-level statements. Text
    already starting with `def` is returned unchanged (and does not get the loop
    or auto-stop below). A structured/exported program (functions, `global`s,
    control-flow blocks, `$` markers) is wrapped without reordering by
    ``_wrap_structured``; only the flat lesson scripts get the hoist-and-loop.

    Variable assignments are hoisted to initialize once, before any loop:
    URScript will not accept a variable first assigned inside a loop, and every
    variable must be initialized before use. The remaining (motion) statements go
    in the loop. ``loop`` repeats the motion that many times; ``None`` runs it
    once. URScript has no ``range``, so the repeat is a while-counter.

    A script on the secondary interface keeps ``runtime_state`` at PLAYING even
    after it returns, so "finished" cannot be read from it. Instead the program
    flips float register DONE_REG to 1 on its last line and the recorder stops
    on it.
    """
    if text.lstrip().startswith("def "):
        return text
    if _is_structured(text):
        return _wrap_structured(text, loop)

    init, motion, seen_motion = [], [], False
    for ln in text.splitlines():
        s = ln.strip()
        if _is_assignment(s):
            init.append(s)                        # initialize once, before the loop
        elif s == "" or s.startswith("#"):
            (motion if seen_motion else init).append(s)
        else:
            seen_motion = True
            motion.append(s)                      # movej / sleep / register writes

    # Each block body is indented under its `def`/`while` header.
    IND = "  "
    out = ["def prog():", IND + f"write_output_float_register({DONE_REG}, 0)"]
    out += [(IND + l if l else "") for l in init]
    if loop is None:
        out += [(IND + l if l else "") for l in motion]
    else:
        out += [IND + "loop_count = 0", IND + f"while (loop_count < {loop}):"]
        out += [(2 * IND + l if l else "") for l in motion]
        out += [2 * IND + "loop_count = loop_count + 1", IND + "end"]
    out += [IND + f"write_output_float_register({DONE_REG}, 1)", "end"]
    return "\n".join(out) + "\n"


def wrap_path(rows, dt: float, loop: int | None = None) -> str:
    """Wrap a list of joint setpoints into a servoj-streaming program.

    Each row is six joint values, optionally followed by a per-row servoj time
    (a 7th column); rows without it use ``dt``. Embeds every setpoint as a
    literal ``servoj()`` line -- fine on URSim, but a real controller silently
    drops any program over roughly 30 KB of text (see ``record_path``'s
    "batch" vs "stream" engines).
    """
    IND = "  "
    body = []
    for r in rows:
        q, dt_r = r[:6], (r[6] if len(r) > 6 else dt)
        body.append(f"servoj([{', '.join(f'{v:.6f}' for v in q)}], 0, 0, "
                    f"{dt_r:.6f}, {SERVO_LOOKAHEAD}, {SERVO_GAIN})")
    out = ["def prog():", IND + f"write_output_float_register({DONE_REG}, 0)"]
    if loop is None:
        out += [IND + b for b in body]
    else:
        out += [IND + "loop_count = 0", IND + f"while (loop_count < {loop}):"]
        out += [2 * IND + b for b in body]
        out += [2 * IND + "loop_count = loop_count + 1", IND + "end"]
    out += [IND + f"write_output_float_register({DONE_REG}, 1)", "end"]
    return "\n".join(out) + "\n"


def load_path(path: str) -> list:
    """Read a path CSV of joint setpoints (six columns q0..q5) into a list of rows."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.reader(f):
            if not r:
                continue
            try:
                rows.append([float(x) for x in r])
            except ValueError:
                continue                   # skip a header line
    return rows


def send_program(host: str, program: str, port: int = SCRIPT_PORT):
    """Push a ready-made URScript program to the controller, which runs it."""
    if not program.endswith("\n"):
        program += "\n"                    # controller runs on the trailing newline
    with socket.create_connection((host, port), timeout=5) as s:
        s.sendall(program.encode())


def _done_check():
    """Build a per-sample stop test for the sent program's done-flag register.

    Returns a closure ``check(smp, t)`` for ``record.record_stream`` that returns
    a stop reason once the program ends:
      - program starts (runtime_state = PLAYING) and the wrapper resets the flag
        to 0 (armed = motion running under our program),
      - wrapper flips the flag to 1 on its last line -> finished,
      - runtime leaves PLAYING after it started -> aborted,
      - never starts within START_TIMEOUT_S -> not in Remote Control mode.
    """
    state = {"started": False, "armed": False}

    def check(smp, t):
        playing = smp["runtime_state"] == RUNTIME_PLAYING
        flag = smp[DONE_FIELD]
        if playing:
            state["started"] = True
        if state["started"] and flag < 0.5:        # our program reset it: motion running
            state["armed"] = True
        if state["armed"] and flag >= 0.5:         # prog flipped it on its last line
            return "program finished"
        if state["started"] and not playing:       # runtime dropped: script aborted
            return "program stopped (aborted? check URSim Log Messages)"
        if not state["started"] and t > START_TIMEOUT_S:
            return "program never started (is the robot in Remote Control mode?)"
        return None

    return check


def run_and_record(host, program, out, hz, recipe, port):
    """Send a ready-made program, record RTDE to `out`, stop when it finishes.

    Returns (n_samples, reason). Uses ``record.record_stream`` for the RTDE
    read/write loop, sending the program first and passing a done-flag stop
    test. Ctrl-C or a dropped stream also stop.
    """
    stream = open_stream(host, hz, recipe)
    send_program(host, program, port)      # stream open first, so we catch the start
    print(f"sent + recording {host}:{port} -> {out}  (Ctrl-C to stop)")
    try:
        return record_stream(stream, out, recipe, stop_check=_done_check())
    finally:
        stream.close()


def record_run(host, script, out, hz=125.0, registers=("1", "vel", "2", "acc"),
               loop=None, port=SCRIPT_PORT):
    """Run a URScript file on the robot once (or ``loop`` times) and record it.

    Loads the script, wraps it as a program, builds the RTDE recipe (the
    requested registers plus the done-flag), and records to ``out``.
    """
    preflight(host)
    regs = parse_registers(list(registers)) + [(DONE_REG, "_done")]
    recipe = build_recipe(regs)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    program = wrap_program(load_script(script), loop)
    return run_and_record(host, program, out, hz, recipe, port)


_UR_RTDE_INDEX = re.compile(r"_(\d+)$")  # ur_rtde names channels target_q_0; we use target_q0


def _reshape_recording(raw: str, out: str) -> int:
    """Turn ur_rtde's own recording CSV into record.py's column schema, so
    analysis.py can load it like any other recording: target_q_0 -> target_q0,
    and its absolute controller ``timestamp`` -> ``t`` seconds since the first
    row. Returns the row count.
    """
    with open(raw, newline="") as f:
        header, *data = csv.reader(f)
    cols = ["t" if c == "timestamp" else _UR_RTDE_INDEX.sub(r"\1", c) for c in header]
    ti = header.index("timestamp")
    t0 = float(data[0][ti]) if data else 0.0
    for row in data:
        row[ti] = f"{float(row[ti]) - t0:.6f}"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(data)
    os.remove(raw)
    return len(data)


def _record_path_batch(host, rows, out, dt, hz, loop, port):
    """Embed the whole path as literal servoj() text in one program, sent like
    a .script run. Simple, no extra dependency, but a real controller silently
    drops any program over ~30 KB of text; fine on URSim.
    """
    recipe = build_recipe([(DONE_REG, "_done")])
    program = wrap_path(rows, dt, loop)
    return run_and_record(host, program, out, hz, recipe, port)


def _record_path_stream(host, rows, out, dt):
    """Stream the path live: RTDEControlInterface for servoJ,
    RTDEReceiveInterface for recording (the pairing its own examples use). A
    hand-rolled real-time protocol over raw sockets can't reliably hit an 8 ms
    deadline from Python and is not an alternative (see README's Known gaps).
    A safety-limited moveJ reaches the path's start first: servoJ only tracks
    small per-tick corrections, not an arbitrary jump.

    RTDEControlInterface paces ``waitPeriod`` to its own fixed ``frequency``,
    not to the ``t`` passed to servoJ, so a row's own ``dt`` can't be handed to
    servoJ directly -- the loop ticks at the fixed STREAM_HZ instead, and each
    row's servoJ call repeats for as many ticks as its own ``dt`` needs
    (rounded to the nearest tick). Every ``.path`` this pipeline writes is on
    the uniform DT grid (optimize.py resamples onto it before writing), so
    that round-trip is exact in practice, one tick per row; the rounding only
    matters for a ``dt`` that isn't a whole multiple of STREAM_HZ's period.
    """
    import rtde_control    # heavy optional dependency: only needed for this engine
    import rtde_receive

    period = 1.0 / STREAM_HZ
    c = rtde_control.RTDEControlInterface(host, frequency=STREAM_HZ)
    r = rtde_receive.RTDEReceiveInterface(host)
    raw = out + ".raw"
    r.startFileRecording(raw)
    print(f"streaming {host} -> {out}  ({len(rows)} rows)")
    stop = "path finished"
    try:
        c.moveJ(rows[0][:6])    # a safety-limited move to the path's start, same
                                 # as exercise01/ur_servoJ.py -- servoJ handles only
                                 # small per-tick corrections, not an initial jump
        for row in rows:
            q, row_dt = row[:6], (row[6] if len(row) > 6 else dt)
            for _ in range(max(1, round(row_dt / period))):
                t_start = c.initPeriod()
                c.servoJ(q, 0.0, 0.0, period, SERVO_LOOKAHEAD, SERVO_GAIN)
                c.waitPeriod(t_start)
    except KeyboardInterrupt:
        stop = "Ctrl-C"
    finally:
        c.servoStop()
        c.stopScript()
        r.stopFileRecording()
    return _reshape_recording(raw, out), stop


def record_path(host, path, out, engine, dt=DT, hz=125.0, loop=None, port=SCRIPT_PORT):
    """Stream a servoj path file on the robot once (or ``loop`` times) and record it.

    ``engine`` is "batch" (embed the whole path as one program; see
    ``_record_path_batch`` -- fine on URSim, but a real controller silently
    drops any program over ~30 KB) or "stream" (stream live via the ur_rtde
    library; see ``_record_path_stream`` -- needed on real hardware).
    """
    preflight(host)
    rows = load_path(path)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if engine == "batch":
        return _record_path_batch(host, rows, out, dt, hz, loop, port)
    return _record_path_stream(host, rows * (loop or 1), out, dt)


def main():
    ap = argparse.ArgumentParser(description="Send a URScript to a UR robot and record it.")
    ap.add_argument("--robot-ip", default="127.0.0.1",
                    help="UR controller IP (default 127.0.0.1, i.e. local URSim)")
    ap.add_argument("source", help="a .script to run, or a .path to stream")
    ap.add_argument("--dt", type=float, default=DT,
                    help="servoj time per row, when the path has no dt column")
    ap.add_argument("--out", default=None,
                    help="output CSV (default: the source name, plus .<engine> for "
                         "a .path, with .csv)")
    ap.add_argument("--hz", type=float, default=125.0, help="sample rate (default 125)")
    ap.add_argument("--float-register", nargs="+", metavar="IDX NAME",
                    default=["1", "vel", "2", "acc"],
                    help="log output float registers, e.g. 1 vel 2 acc")
    ap.add_argument("--port", type=int, default=SCRIPT_PORT,
                    help="script interface port (30002 secondary, 30001 primary)")
    ap.add_argument("--loop", type=int, default=None,
                    help="repeat N times then stop (default: run once)")
    ap.add_argument("--engine", choices=["batch", "stream"],
                    help="how to stream a .path: batch (URSim) or stream (real hardware)")
    args = ap.parse_args()
    stem = args.source.rsplit(".", 1)[0]

    if args.source.endswith(".path"):        # stream a servoj path
        if args.engine is None:
            raise SystemExit("--engine batch|stream is required for a .path")
        out = args.out or f"{stem}.{args.engine}.csv"
        n, stop = record_path(args.robot_ip, args.source, out, args.engine,
                              args.dt, args.hz, args.loop, args.port)
    else:                                    # run a URScript
        out = args.out or f"{stem}.csv"
        n, stop = record_run(args.robot_ip, args.source, out, args.hz,
                             args.float_register, args.loop, args.port)
    print(f"\nwrote {n} samples to {out}" + (f"  ({stop})" if stop else ""))


if __name__ == "__main__":
    main()
