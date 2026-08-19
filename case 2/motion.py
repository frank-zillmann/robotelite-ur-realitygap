"""The spline the optimizer bends a path with, and the envelope it has to stay in.

    q = bspline(u, control_points)         # differentiable in both arguments

What the controller itself commands is not modelled here -- convert.py records it
from the controller instead. What is still needed is where the limits are, measured
from the recordings in ``data/``: the ceiling is usually not a joint limit but the
Cartesian one, the tool running at exactly 1.3500 m/s through the middle of every
recorded move.
"""
from __future__ import annotations

import numpy as np
import torch

from utils import N_JOINTS, UR10e

# Identified from data/test-*.csv, peak per lead joint (rad/s, rad/s^2).
V_JOINT = np.array([1.885, 2.094, 2.765, 3.142, 3.142, 3.142])
A_JOINT = np.array([18.48, 15.00, 32.63, 40.90, 40.90, 40.90])
V_TCP = 1.359             # m/s, the tool-speed cap the recordings sit on
_UR = UR10e()


def bspline(u, points):
    """Uniform cubic B-spline ``q(u)`` for ``u`` in [0,1], control ``points`` (m, 6).

    Differentiable in both arguments, which is why the basis is spelled out here
    rather than taken from scipy: the optimizer moves the points *and* the time
    scale, so ``u`` itself carries gradient.

    Repeating an end point three times pins the curve to it with zero velocity and
    acceleration, which is how ``clamp`` builds the ends.
    """
    m = points.shape[0]
    x = u.clamp(0.0, 1.0) * (m - 3)
    k = x.floor().clamp(0, m - 4).long()
    s = (x - k)[:, None]
    b = torch.stack([(1 - s) ** 3, 3 * s ** 3 - 6 * s ** 2 + 4,
                     -3 * s ** 3 + 3 * s ** 2 + 3 * s + 1, s ** 3]) / 6.0
    return sum(b[i] * points[k + i] for i in range(4))


def clamp(q0, q1, interior):
    """Control points that start at ``q0`` and end at ``q1``, both at a standstill."""
    ends = lambda q: torch.as_tensor(q, dtype=torch.float32).expand(3, N_JOINTS)
    return torch.cat([ends(q0), interior, ends(q1)])


def tcp_speed(q, dt: float) -> np.ndarray:
    """Tool speed (m/s) of a commanded trajectory, to check it against ``V_TCP``."""
    qd = np.gradient(np.asarray(q, float), dt, axis=0)
    return np.array([np.linalg.norm((_UR.jacobian(qi) @ qdi)[:3]) for qi, qdi in zip(q, qd)])
