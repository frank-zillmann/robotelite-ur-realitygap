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

_G = 9.80665  # gravity [m/s^2]


class Robot:
    """One UR arm: kinematics, dynamics, and the ceilings its controller enforces.

        r = Robot("UR5e")
        r.fk(q)                  # 4x4 base -> flange pose
        r.jacobian(q)            # 6x6 geometric Jacobian, [v; w] = J(q) @ qd
        r.tcp_speed(q, dt)       # tool speed (m/s) along a trajectory
        r.gravity(q)             # (6,) gravity torque per joint, Nm
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
    # Inertial parameters (mass, per-link COM, per-link inertia about its COM, all
    # in that link's own DH frame) from Universal_Robots_ROS2_Description
    # (config/ur10e, config/ur5e physical_parameters.yaml), mapped into the same
    # DH frames as ``a``/``d``/``ALPHA`` -- upstream international-summer-school
    # course commit "Use a UR5e as the case 2 robot" (github.com/ureskr/
    # international-summer-school-robotics-TER-UR, branch case-2-ur5e).
    MODELS = {
        "UR10e": dict(
            a=[0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0],
            d=[0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655],
            v_deg=[120, 120, 180, 180, 180, 180],
            mass=[7.369, 13.051, 3.989, 2.1, 1.98, 0.615],
            com=[[0.021, 0.000, 0.027], [0.380, 0.000, 0.158], [0.240, 0.000, 0.068],
                [0.000, 0.007, 0.018], [0.000, 0.007, 0.018], [0.000, 0.000, -0.026]],
            inertia=[
                [[0.0341, 0.0000, -0.0043], [0.0000, 0.0353, 0.0001], [-0.0043, 0.0001, 0.0216]],
                [[0.0281, 0.0001, -0.0156], [0.0001, 0.7707, 0.0000], [-0.0156, 0.0000, 0.7694]],
                [[0.0101, 0.0001, 0.0092], [0.0001, 0.3093, 0.0000], [0.0092, 0.0000, 0.3065]],
                [[0.0030, 0.0000, 0.0000], [0.0000, 0.0022, -0.0002], [0.0000, -0.0002, 0.0026]],
                [[0.0030, 0.0000, 0.0000], [0.0000, 0.0022, -0.0002], [0.0000, -0.0002, 0.0026]],
                [[0.0000, 0.0000, 0.0000], [0.0000, 0.0004, 0.0000], [0.0000, 0.0000, 0.0003]]]),
        "UR5e": dict(
            a=[0.0, -0.425, -0.3922, 0.0, 0.0, 0.0],
            d=[0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996],
            v_deg=[180, 180, 180, 180, 180, 180],
            mass=[3.761, 8.058, 2.846, 1.37, 1.3, 0.365],
            com=[[0.0000, -0.02561, 0.001930], [0.2125, 0.00000, 0.113360],
                [0.1500, 0.00000, 0.026500], [0.0000, -0.00180, 0.016340],
                [0.0000, 0.00180, 0.016340], [0.0000, 0.00000, -0.001159]],
            inertia=[
                [[0.00700210, 0.00000073, -0.00001053], [0.00000073, 0.00648091, 0.00049994],
                 [-0.00001053, 0.00049994, 0.00657286]],
                [[0.01505885, -0.00005400, 0.00000563], [-0.00005400, 0.33388086, -0.00000181],
                 [0.00000563, -0.00000181, 0.33247207]],
                [[0.00399632, -0.00001365, 0.00137272], [-0.00001365, 0.07879254, -0.00000660],
                 [0.00137272, -0.00000660, 0.07848510]],
                [[0.00165491, -0.00000282, -0.00000438], [-0.00000282, 0.00135962, 0.00010157],
                 [-0.00000438, 0.00010157, 0.00126279]],
                [[0.00135617, -0.00000274, 0.00000444], [-0.00000274, 0.00127827, -0.00005048],
                 [0.00000444, -0.00005048, 0.00096614]],
                [[0.00018694, 0.00000006, -0.00000017], [0.00000006, 0.00018908, -0.00000092],
                 [-0.00000017, -0.00000092, 0.00025756]]]),
    }

    def __init__(self, model: str = "UR10e"):
        if model not in self.MODELS:
            raise ValueError(f"unknown robot {model!r}, have {list(self.MODELS)}")
        p = self.MODELS[model]
        self.model = model
        self.a, self.d = np.array(p["a"]), np.array(p["d"])
        self.v_joint = np.deg2rad(p["v_deg"]) * self.MARGIN
        self.a_joint, self.v_tcp = self.A_JOINT, self.V_TCP * self.MARGIN
        self.mass = np.array(p["mass"])
        self.com = np.array(p["com"])
        self.inertia = np.array(p["inertia"])

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

    def _point_jacobian(self, frames: list, point: np.ndarray, up_to: int) -> np.ndarray:
        """Linear + angular Jacobian (6x6) of a point rigidly on link ``up_to``.

        Only the first ``up_to`` joints move the point; later columns are zero.
        Shared by ``jacobian`` (the flange) and ``_link_terms`` (each link's COM).
        """
        J = np.zeros((6, 6))
        for j in range(up_to):
            z, p = frames[j][:3, 2], frames[j][:3, 3]   # joint axis, and a point on it
            J[:3, j], J[3:, j] = np.cross(z, point - p), z
        return J

    def jacobian(self, q) -> np.ndarray:
        """Geometric Jacobian (6x6) of the flange in the base frame."""
        frames = self._frames(q)
        return self._point_jacobian(frames, frames[N_JOINTS][:3, 3], N_JOINTS)

    def _link_terms(self, q):
        """Per-link COM Jacobians, for the gravity torque sum.

        Returns ``[(mass, Jv), ...]``, one per link: ``Jv`` is the 3xN_JOINTS
        linear Jacobian of that link's centre of mass.
        """
        frames = self._frames(q)
        terms = []
        for i in range(N_JOINTS):
            R = frames[i + 1][:3, :3]                          # base <- link frame
            com = frames[i + 1] @ np.append(self.com[i], 1.0)  # COM in base frame
            Jv = self._point_jacobian(frames, com[:3], i + 1)[:3]
            terms.append((self.mass[i], Jv))
        return terms

    def gravity(self, q) -> np.ndarray:
        """Gravity torque per joint (N_JOINTS,), Nm: the torque to hold pose ``q``
        against gravity with no motion at all (the ``g(q)`` term of the arm's
        rigid-body dynamics, ``M(q) qdd + C(q,qd) qd + g(q) = tau``)."""
        g_vec = np.array([0.0, 0.0, -_G])
        tau = np.zeros(N_JOINTS)
        for mass, Jv in self._link_terms(q):
            tau -= mass * (Jv.T @ g_vec)
        return tau

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
