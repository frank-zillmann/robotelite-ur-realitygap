"""Shared helpers for the case 2 scripts: constants, robot physics, scripts.

- constants: joint count/names and the CSV column names.
- `Robot`: numpy-only kinematics (FK, Jacobian) and the controller's speed and
  acceleration ceilings, for a UR10e or a UR5e.
- `load_script` / `set_param`: read a `.script`, retune its `vel`/`acc`.
"""
from __future__ import annotations

import re

import numpy as np

# --- constants ---------------------------------------------------------------
N_JOINTS = 6                                             # base, shoulder, elbow, wrist1, wrist2, wrist3
JOINT_NAMES = ("base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3")

# One sample period for everything: what record.py asks the RTDE stream for, what
# the model is trained on, and what a path is written at. 8 ms divides both control
# cycles UR ships (2 ms on e-Series, 8 ms on CB3), so it is a rate every box can
# actually deliver rather than one it has to round.
DT = 0.008                                               # s (125 Hz)

TIME_COL = "t"                                           # seconds since recording started
# The optimized motion parameters, from the URScript `movej(..., a=acc, v=vel)`,
# logged by record.py through these output float registers (raw script values,
# e.g. 100).
VEL_COL = "vel"                                          # output_double_register_1
ACC_COL = "acc"                                          # output_double_register_2
SCL_COL = "script_control_line"                          # URScript line running now
SCRIPT_COL = "script"                                    # source script of each row


# --- recording column helpers ------------------------------------------------
# The CSV stores each per-joint channel as six columns, e.g. actual_current0..5.
# These read/write a whole channel as one (n, N_JOINTS) block, addressed by the
# base name.

def joint_cols(base: str) -> list[str]:
    """Column names for one per-joint channel, e.g. ``actual_current0..5``."""
    return [f"{base}{j}" for j in range(N_JOINTS)]


def get_block(df, base: str) -> "np.ndarray":
    """Read a per-joint channel from a DataFrame as an ``(n, N_JOINTS)`` array."""
    return df[joint_cols(base)].to_numpy(dtype=float)


# --- UR kinematics and limits (numpy only) -----------------------------------

class Robot:
    """One UR arm: kinematics, and the ceilings its controller enforces.

        r = Robot("UR5e")
        r.fk(q)                  # 4x4 base -> flange pose
        r.jacobian(q)            # 6x6 geometric Jacobian, [v; w] = J(q) @ qd
        r.tcp_speed(q, dt)       # tool speed (m/s) along a trajectory
        r.v_joint, r.a_joint, r.v_tcp
    """

    ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])
    A_JOINT = np.array([25.0, 65.0, 60.0, 45.0, 35.0, 35.0])      # rad/s^2
    V_TCP = 1.35                                                  # m/s
    # The speed ceilings are checked by differencing q, which overshoots by about a
    # percent at the corners of a profile, so they get that much headroom. Without
    # it the controller's own paths score as violations and the optimizer only ever
    # slows down. The recordings sit exactly on the spec: 2.094 and 3.142 rad/s.
    MARGIN = 1.02
    MODELS = {
        "UR10e": dict(a=[0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0],
                      d=[0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655],
                      v_deg=[120, 120, 180, 180, 180, 180]),
        "UR5e": dict(a=[0.0, -0.425, -0.3922, 0.0, 0.0, 0.0],
                     d=[0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996],
                     v_deg=[180, 180, 180, 180, 180, 180]),
    }

    def __init__(self, model: str = "UR10e"):
        if model not in self.MODELS:
            raise ValueError(f"unknown robot {model!r}, have {list(self.MODELS)}")
        p = self.MODELS[model]
        self.model = model
        self.a, self.d = np.array(p["a"]), np.array(p["d"])
        self.v_joint = np.deg2rad(p["v_deg"]) * self.MARGIN
        self.a_joint, self.v_tcp = self.A_JOINT, self.V_TCP * self.MARGIN

    @staticmethod
    def _dh(theta: float, a: float, d: float, alpha: float) -> np.ndarray:
        """Standard DH homogeneous transform from one frame to the next."""
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([[ct, -st * ca, st * sa, a * ct],
                         [st, ct * ca, -ct * sa, a * st],
                         [0.0, sa, ca, d],
                         [0.0, 0.0, 0.0, 1.0]])

    def _frames(self, q) -> list[np.ndarray]:
        """Cumulative base->frame transforms T[0..6]; T[0] is the base."""
        q = np.asarray(q, dtype=float)
        frames = [np.eye(4)]
        for i in range(N_JOINTS):
            frames.append(frames[-1] @ self._dh(q[i], self.a[i], self.d[i], self.ALPHA[i]))
        return frames

    def fk(self, q) -> np.ndarray:
        """Base->flange (TCP) pose as a 4x4 homogeneous transform."""
        return self._frames(q)[N_JOINTS]

    def jacobian(self, q) -> np.ndarray:
        """Geometric Jacobian (6x6) of the flange in the base frame."""
        frames = self._frames(q)
        point = frames[N_JOINTS][:3, 3]
        J = np.zeros((6, 6))
        for j in range(N_JOINTS):
            z, p = frames[j][:3, 2], frames[j][:3, 3]   # joint axis, and a point on it
            J[:3, j], J[3:, j] = np.cross(z, point - p), z
        return J

    def jacobians(self, q) -> np.ndarray:
        """Linear part of the Jacobian at every pose of a trajectory, (n, 3, 6)."""
        return np.array([self.jacobian(p)[:3] for p in np.asarray(q, float)])

    def tcp_speed(self, q, dt: float = DT) -> np.ndarray:
        """Tool speed (m/s) along a commanded trajectory."""
        qd = np.gradient(np.asarray(q, float), dt, axis=0)
        return np.linalg.norm(np.einsum("nij,nj->ni", self.jacobians(q), qd), axis=1)


# --- URScript helpers --------------------------------------------------------
# Matches `vel = 100` / `acc = 100` (int or float), with an optional `global`
# prefix that is preserved on replacement. Plain (non-global) assignments let
# send.py wrap the motion in a repeat loop, since URScript rejects a `global`
# declaration inside a loop.
def load_script(path: str) -> str:
    """Read a URScript file to text."""
    with open(path) as f:
        return f.read()


def set_param(text: str, name: str, value: float) -> str:
    """Replace a `<name> = <number>` line in URScript text, e.g. `vel`/`acc`.

    Used by collect_data.py to run one motion at several speeds.
    """
    pattern = r"((?:global\s+)?" + name + r"\s*=\s*)([0-9]+(?:\.[0-9]+)?)"
    new, n = re.subn(pattern, lambda m: f"{m.group(1)}{value:g}", text)
    if not n:
        raise ValueError(f"no `{name} = ...` line in the script")
    return new
