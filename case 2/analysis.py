"""Interactive plot of a recorded UR run.

Loads a CSV from ``record.py`` and plots up to three sources of one quantity:

    target   what the controller commanded  (target_* columns)
    actual   what the robot measured        (actual_* columns)
    script   what the URScript alone implies, rebuilt through this repo's own
             models (``--script``, see ``script_plan``)

    python analysis.py --csv data/test-4.csv --script data/test-4.script --quantity "angle q" --joint 1

``--joint`` picks the component, which is a joint (0..5 = base..wrist3) for the
per-joint quantities and a Cartesian axis (0..5 = x, y, z, rx, ry, rz) for the
TCP ones, since those are poses in space and have nothing to do with joints.

``Recording`` is the shared CSV data loader used across the pipeline
(``train_distillation_model.py``, ``train_rla.py``); it wraps a run's CSV as
numpy arrays.
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from utils import (ACC_COL, JOINT_NAMES, N_JOINTS, SCL_COL, SCRIPT_COL, TIME_COL,
                   VEL_COL, load_script)

TCP_AXES = ("x", "y", "z", "rx", "ry", "rz")
WAYPOINT_TOL = 0.05       # rad: script waypoint vs the one actually reached

# Quantities the viewer can show: label -> (target base, actual base, unit, component names).
QUANTITIES = {
    "current":   ("target_current", "actual_current", "A", JOINT_NAMES),
    "angle q":   ("target_q", "actual_q", "rad", JOINT_NAMES),
    "speed qd":  ("target_qd", "actual_qd", "rad/s", JOINT_NAMES),
    "accel qdd": ("target_qdd", "actual_qdd", "rad/s^2", JOINT_NAMES),
    "moment":    ("target_moment", "actual_current_as_torque", "Nm", JOINT_NAMES),
    "TCP pose":  ("target_TCP_pose", "actual_TCP_pose", "m, rad", TCP_AXES),
    "TCP speed": ("target_TCP_speed", "actual_TCP_speed", "m/s, rad/s", TCP_AXES),
}


class Recording:
    """One recorded run loaded from a ``record.py`` CSV, as numpy arrays."""

    def __init__(self, path: str, df=None):
        # Loads ``path``, or wraps an already-loaded table if ``df`` is given
        # (e.g. one a preprocess.Preprocess transformed).
        if df is None:
            df = pd.read_csv(path)
        self.path = path
        self.df = df             # raw table; callers can slice rows by index
        self.t = df[TIME_COL].to_numpy(dtype=float)
        self.target_q = np.column_stack([df[f"target_q{j}"] for j in range(6)])
        self.actual_q = np.column_stack([df[f"actual_q{j}"] for j in range(6)])
        self.target_qd = np.column_stack([df[f"target_qd{j}"] for j in range(6)])
        self.target_current = np.column_stack([df[f"target_current{j}"] for j in range(6)])
        self.actual_current = np.column_stack([df[f"actual_current{j}"] for j in range(6)])
        # Commanded movej parameters, if record.py logged them (raw URScript
        # values, e.g. 100). None when the run has no vel/acc registers.
        self.vel_cmd = df[VEL_COL].to_numpy(dtype=float) if VEL_COL in df else None
        self.acc_cmd = df[ACC_COL].to_numpy(dtype=float) if ACC_COL in df else None
        # URScript line currently executing (RTDE): nonzero while a movej runs, 0
        # during a sleep/dwell. The value is script-specific and not comparable
        # across scripts, but a change marks a move boundary. Zeros if not logged.
        self.scl = (df[SCL_COL].to_numpy() if SCL_COL in df
                    else np.zeros(len(self.t), dtype=int))
        # Source script of each row, tagged when several runs are pooled into one
        # file, so segmentation never spans two scripts. One value if not tagged.
        self.script = (df[SCRIPT_COL].to_numpy() if SCRIPT_COL in df
                       else np.zeros(len(self.t), dtype=int))

    @property
    def dt(self) -> float:
        """Median sample period (s)."""
        return float(np.median(np.diff(self.t)))

    def channel(self, base: str):
        """A per-joint channel as ``(n, N_JOINTS)``, or ``None`` if not recorded.

        ``actual_qdd`` is not an RTDE field, so it is differentiated from
        ``actual_qd`` on demand; everything else is read straight from the CSV.
        """
        if base is None:
            return None
        cols = [f"{base}{j}" for j in range(N_JOINTS)]
        if all(c in self.df for c in cols):
            return self.df[cols].to_numpy(dtype=float)
        if base == "actual_qdd":
            qd = self.channel("actual_qd")
            return None if qd is None else np.gradient(qd, self.dt, axis=0)
        return None


# --- the script as a third data source ---------------------------------------

_VEC = r"\[\s*([-+\d.eE,\s]+?)\s*\]"


def _waypoints(text: str) -> list:
    """Joint waypoint of every ``movej``, in order; ``None`` where unparsable.

    Covers the three forms here: a literal ``movej([...])``, a named pose
    (``POSE_A = [...]``, scripts/*.script), and the teach-pendant
    ``movej(get_inverse_kin(..., qnear=P.q))`` whose joints are the ``q=[...]``
    of ``global P = struct(...)`` (data/test-*.script).
    """
    vec = lambda s: np.array([float(v) for v in s.split(",")])
    named = {}
    for m in re.finditer(rf"(?:global\s+)?(\w+)\s*=\s*{_VEC}", text):
        if len(v := vec(m.group(2))) == N_JOINTS:
            named[m.group(1)] = v
    for m in re.finditer(rf"(?:global\s+)?(\w+)\s*=\s*struct\([^\n]*?q\s*=\s*{_VEC}", text):
        if len(v := vec(m.group(2))) == N_JOINTS:
            named[f"{m.group(1)}.q"] = v

    out = []
    for m in re.finditer(r"movej\s*\(([^\n]*)", text):
        arg = m.group(1)
        if lit := re.match(rf"\s*{_VEC}", arg):
            out.append(vec(lit.group(1)))
        elif (nm := re.match(r"\s*([A-Za-z_]\w*)\s*[,)]", arg)) and nm.group(1) in named:
            out.append(named[nm.group(1)])
        elif (qn := re.search(r"qnear\s*=\s*([A-Za-z_]\w*)\.q", arg)) \
                and f"{qn.group(1)}.q" in named:
            out.append(named[f"{qn.group(1)}.q"])
        else:
            out.append(None)
    return out


def _limits(rec: Recording):
    """Per-joint speed and acceleration ceilings, measured from the run.

    ``movej``'s ``v=``/``a=`` are rad/s and rad/s^2, and the controller clamps
    them to its own per-joint ceiling (scaled by the speed slider). In every
    supplied run the requested values sit far above that ceiling, so the script's
    numbers say nothing about the motion: ``vel=100, acc=100`` would be a 0.36 s
    move where the real one takes 2.98 s. And the ceiling is not in the script --
    the same request gives 59 deg/s in test-4 and 120 deg/s in test-6, because
    the speed slider is set on the pendant. So it has to be measured.
    """
    a = np.gradient(rec.target_qd, rec.dt, axis=0)
    return np.abs(rec.target_qd).max(axis=0), np.abs(a).max(axis=0)


def script_plan(rec: Recording, script_path: str, payload: float = 0.0,
                fit_duration: bool = False) -> dict:
    """What the URScript alone implies, on the recording's clock.

    A third source next to ``target_*`` and ``actual_*``: the distance to
    ``target_*`` is ``dynamics.py``'s error, and to ``actual_*`` that plus the
    reality gap. Each ``movej`` waypoint is parsed from the script, its speed
    profile rebuilt with ``dynamics.trapezoidal`` and its current with
    ``dynamics.UR10eDynamics``. All joints of a movej start and stop together, so
    the profile is the slowest joint's. Between moves the plan holds the waypoint.

    ``script_control_line`` gives the sample where each move began (via
    ``common.segments``), and every rebuilt move is laid down there. Its
    *duration* then comes from ``trapezoidal``, which matches the swing runs to
    under 1% but is out by ~25% on test-6/7, where the controller eases
    acceleration in and out rather than switching it. That is a real gap in
    ``dynamics.py``, left visible on purpose; ``fit_duration=True`` stretches
    each move onto its recorded window instead, to compare shape without it.

    Returns ``{base: (n, N_JOINTS)}``, or ``{}`` if no ``movej`` could be parsed.
    """
    from common import segments
    from dynamics import GRID, MAX_JOINT_ACC, MAX_JOINT_SPEED, UR10eDynamics, trapezoidal

    way = _waypoints(load_script(script_path))
    segs = segments(rec)
    if not segs or not any(w is not None for w in way):
        return {}

    dt = rec.dt
    v_lim, a_lim = _limits(rec)
    v_lim = np.where(v_lim > 1e-6, v_lim, MAX_JOINT_SPEED)
    a_lim = np.where(a_lim > 1e-6, a_lim, MAX_JOINT_ACC)

    # Which movej is which: script_control_line is the running movej's line
    # number, so the distinct values sorted are the movejs in file order -- right
    # even when nested loops run them out of order (data/test-6-7.script). With
    # no such column, guess a cyclic offset and score it against the recording.
    lines = sorted({int(rec.scl[s.i0]) for s in segs if rec.scl[s.i0]})
    if len(lines) == len(way):
        index = {ln: i for i, ln in enumerate(lines)}
        pick = lambda k, seg: index.get(int(rec.scl[seg.i0]))
    else:
        off = min(range(len(way)),
                  key=lambda o: sum(float(np.abs(w - rec.target_q[s.i1]).max())
                                    for k, s in enumerate(segs)
                                    if (w := way[(k + o) % len(way)]) is not None))
        pick = lambda k, seg: (k + off) % len(way)

    n = len(rec.t)
    q = np.repeat(rec.target_q[segs[0].i0][None], n, axis=0)
    prog = np.ones(n)                     # progress along the current move, for the cache
    blocks, pos = [], rec.target_q[segs[0].i0].copy()
    for k, seg in enumerate(segs):
        mid = pick(k, seg)                    # which movej of the script this is
        w = way[mid] if mid is not None else None
        # `movej(get_inverse_kin(..., qnear=P.q))` names a pose; qnear is only the
        # seed for the controller's IK, and for some waypoints the solution it
        # picks is far from it. Where the script disagrees with what was actually
        # reached, trust the recording.
        bad = w is None or np.abs(w - rec.target_q[seg.i1]).max() > WAYPOINT_TOL
        dest = rec.target_q[seg.i1] if bad else w
        vel = seg.vel or np.inf                # requested; the ceiling usually wins
        acc = seg.acc or np.inf
        travel = dest - pos
        # One shared profile per movej, so it is the slowest joint's.
        s = max((trapezoidal(abs(travel[j]), min(vel, v_lim[j]), min(acc, a_lim[j]), dt)
                 for j in range(N_JOINTS) if abs(travel[j]) > 1e-4),
                key=len, default=np.array([0.0, 1.0]))
        if fit_duration and seg.i1 > seg.i0:   # re-time onto the recorded window
            s = np.interp(np.linspace(0.0, 1.0, seg.i1 - seg.i0),
                          np.linspace(0.0, 1.0, len(s)), s)
        nxt = segs[k + 1].i0 if k + 1 < len(segs) else n  # where the next movej starts
        stop = min(seg.i0 + len(s), nxt)                  # motion, clipped to the slot
        q[seg.i0:stop] = pos + s[:stop - seg.i0, None] * travel
        prog[seg.i0:stop] = s[:stop - seg.i0]
        q[stop:nxt] = q[stop - 1] if stop > seg.i0 else pos   # hold through the sleep
        blocks.append((seg.i0 if k else 0, nxt, mid, pos, dest))
        # Carry the pose the plan reached, not the waypoint: a move clipped short
        # would otherwise tear a step into the trace.
        pos = q[max(nxt - 1, 0)].copy()

    qd = np.gradient(q, dt, axis=0)
    qdd = np.gradient(qd, dt, axis=0)
    dyn = UR10eDynamics(rec, payload=payload)
    cur = np.zeros_like(q)
    for a, b, mid, p0, p1 in blocks:
        # Pose cache keyed by which movej, not by occurrence: a looped script
        # retraces the same geometry, so this prepares once per movej.
        if mid not in dyn._cache:
            dyn.prepare(mid, p0 + np.linspace(0.0, 1.0, GRID)[:, None] * (p1 - p0))
        cur[a:b] = dyn.current(q[a:b], qd[a:b], qdd[a:b], s=prog[a:b], key=mid)
    return {"target_q": q, "target_qd": qd, "target_qdd": qdd,
            "target_current": cur, "target_moment": cur * dyn.kt}


# --- viewer -------------------------------------------------------------------

def view(rec: Recording, plan: dict = None, quantity: str = "current", joint: int = 1):
    """Plotly figure comparing target / actual / script for one channel."""
    import plotly.graph_objects as go

    t_base, a_base, unit, comps = QUANTITIES[quantity]
    fig = go.Figure()
    for name, block, dash in (("target", rec.channel(t_base), "solid"),
                              ("actual", rec.channel(a_base), "solid"),
                              ("script", (plan or {}).get(t_base), "dash")):
        if block is not None:
            fig.add_scattergl(x=rec.t, y=block[:, joint], name=name,
                              line=dict(dash=dash, width=1.5))
    fig.update_layout(title=f"{rec.path} - {quantity} - {comps[joint]}",
                      xaxis_title="time [s]", yaxis_title=f"{quantity} [{unit}]",
                      hovermode="x unified", template="plotly_white")
    return fig


def main():
    ap = argparse.ArgumentParser(description="Interactive plot of a recorded UR run.")
    ap.add_argument("--csv", default="data/test-4.csv", help="recorded run CSV")
    ap.add_argument("--script", default=None,
                    help="URScript of the run; adds the rebuilt plan as a third trace")
    ap.add_argument("--quantity", choices=list(QUANTITIES), default="current",
                    help="which channel to plot")
    ap.add_argument("--joint", type=int, default=1, choices=range(N_JOINTS),
                    help="component 0..5: a joint (base..wrist3), or a Cartesian "
                         "axis (x, y, z, rx, ry, rz) for the TCP quantities")
    ap.add_argument("--payload", type=float, default=0.0,
                    help="tool mass at the flange [kg], for the rebuilt plan")
    ap.add_argument("--fit-duration", action="store_true",
                    help="re-time each rebuilt move to end when the recorded one "
                         "did, hiding dynamics.py's duration error")
    args = ap.parse_args()

    rec = Recording(args.csv)
    plan = (script_plan(rec, args.script, args.payload, args.fit_duration)
            if args.script else {})
    view(rec, plan, args.quantity, args.joint).show()


if __name__ == "__main__":
    main()
