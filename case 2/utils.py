"""Shared helpers for the case 2 scripts: constants, robot physics, scripts.

- constants: joint count/names and the CSV column names.
- `UR10e`: numpy-only kinematics (FK, Jacobian), plus the speed and acceleration
  ceilings the controller enforces.
- URScript helpers: load a `.script`, read/replace its `vel`/`acc` parameters.
"""
from __future__ import annotations

import re

import numpy as np

# --- constants ---------------------------------------------------------------
N_JOINTS = 6                                             # base, shoulder, elbow, wrist1, wrist2, wrist3
JOINT_NAMES = ("base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3")

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


def set_block(df, base: str, block) -> None:
    """Overwrite a per-joint channel in a DataFrame with an ``(n, N_JOINTS)`` array."""
    df[joint_cols(base)] = np.asarray(block, dtype=float)


def frame_dt(df) -> float:
    """Median sample period (s) of a recording DataFrame (from its ``t`` column)."""
    return float(np.median(np.diff(df[TIME_COL].to_numpy(dtype=float))))


# --- UR10e kinematics (numpy only) -------------------------------------------
# UR10e parameters, from Universal Robots "DH Parameters for calculations of
# kinematics" (universal-robots.com). Standard (classic) DH.
#   joint i rotates by q[i] about z of the previous frame.
_A = np.array([0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0])          # link length a [m]
_D = np.array([0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655])    # link offset d [m]
_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])  # twist [rad]


class UR10e:
    """UR10e kinematics. Extend by overriding the parameter arrays.

        ur = UR10e()
        q  = [0, -1.57, 1.57, -1.57, -1.57, 0]
        ur.fk(q)                            # 4x4 base -> flange pose
        ur.jacobian(q)                      # 6x6 geometric Jacobian (base frame)

    Kinematics only: motion.py needs the Jacobian to convert the controller's
    Cartesian speed cap into joint terms. Nothing models torque any more.
    """

    # --- kinematics -----------------------------------------------------------

    @staticmethod
    def _dh(theta: float, a: float, d: float, alpha: float) -> np.ndarray:
        """Standard DH homogeneous transform from one frame to the next."""
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ])

    def _frames(self, q) -> list[np.ndarray]:
        """Cumulative base->frame transforms T[0..6]; T[0] is the base (identity)."""
        q = np.asarray(q, dtype=float)
        frames = [np.eye(4)]
        for i in range(6):
            frames.append(frames[-1] @ self._dh(q[i], _A[i], _D[i], _ALPHA[i]))
        return frames

    def fk(self, q) -> np.ndarray:
        """Base->flange (TCP) pose as a 4x4 homogeneous transform."""
        return self._frames(q)[6]

    def tcp_position(self, q) -> np.ndarray:
        """Flange (TCP) position in the base frame, shape (3,)."""
        return self.fk(q)[:3, 3]

    def _point_jacobian(self, frames: list[np.ndarray], point: np.ndarray,
                        up_to: int) -> np.ndarray:
        """Linear + angular Jacobian (6x6) of a point rigidly on link ``up_to``.

        Only the first ``up_to`` joints move the point; later columns are zero.
        Columns use the classic revolute-joint form with axis z of the previous
        frame.
        """
        J = np.zeros((6, 6))
        for j in range(up_to):
            z = frames[j][:3, 2]           # joint j axis (z of previous frame)
            p = frames[j][:3, 3]           # origin of previous frame
            J[:3, j] = np.cross(z, point - p)
            J[3:, j] = z
        return J

    def jacobian(self, q) -> np.ndarray:
        """Geometric Jacobian (6x6) of the flange in the base frame.

        Rows 0..2 map joint rates to TCP linear velocity, rows 3..5 to angular
        velocity: ``[v; w] = J(q) @ qd``.
        """
        frames = self._frames(q)
        return self._point_jacobian(frames, frames[6][:3, 3], up_to=6)


# --- URScript helpers --------------------------------------------------------
# Matches `vel = 100` / `acc = 100` (int or float), with an optional `global`
# prefix that is preserved on replacement. Plain (non-global) assignments let
# send.py wrap the motion in a repeat loop, since URScript rejects a `global`
# declaration inside a loop.
_PARAM = r"((?:global\s+)?{name}\s*=\s*)([0-9]+(?:\.[0-9]+)?)"


def load_script(path: str) -> str:
    """Read a URScript file to text."""
    with open(path) as f:
        return f.read()


def get_param(text: str, name: str) -> float:
    """Read a `<name> = <number>` value from URScript text."""
    m = re.search(_PARAM.format(name=name), text)
    if not m:
        raise ValueError(f"no `{name} = ...` line in the script")
    return float(m.group(2))


def set_param(text: str, name: str, value: float) -> str:
    """Return the script with `<name>` set to ``value`` (rounded int)."""
    return re.sub(_PARAM.format(name=name), rf"\g<1>{int(round(value))}", text)


# What the controller will actually run. Speeds are the UR10e spec sheet (base and
# shoulder 120 deg/s, the rest 180) and the tool-speed cap the recordings sit on;
# UR publishes no joint acceleration, so that one is the most the controller was
# ever seen to command in data/. MARGIN is headroom: these get checked by
# differentiating target_q, which reads a few percent above the controller's own
# target_qd channel.
MARGIN = 1.05
V_JOINT = np.deg2rad([120, 120, 180, 180, 180, 180]) * MARGIN   # rad/s
A_JOINT = np.array([25.0, 65.0, 60.0, 45.0, 35.0, 35.0])        # rad/s^2
V_TCP = 1.35 * MARGIN                                           # m/s


def tcp_speed(q, dt: float) -> np.ndarray:
    """Tool speed (m/s) along a commanded trajectory."""
    ur = UR10e()
    qd = np.gradient(np.asarray(q, float), dt, axis=0)
    return np.array([np.linalg.norm((ur.jacobian(a) @ b)[:3]) for a, b in zip(q, qd)])
