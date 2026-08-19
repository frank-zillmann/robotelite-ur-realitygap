"""Commanded joint trajectories: the controller's ``movej``, and a spline to bend it into.

Both produce the same thing -- joint angles on a fixed time grid -- because that is
all the distilled model reads (it takes ``target_q`` and differentiates it itself).

    q = movej(q0, q1, dt)                  # what the UR controller commands
    q = bspline(u, control_points)         # a free trajectory, differentiable in both

``movej`` is an identification of the real controller, not its code, which is
closed. Measured from the recordings in ``data/`` (see ``LIMITS``):

- All joints run one shared path parameter s(t) in [0,1] along the straight
  joint-space line: qd_j/dq_j is the same for every moving joint to within 7e-5.
- s(t) is jerk limited (an S-curve): acceleration ramps in at a fixed rate, holds,
  ramps out.
- The ceiling is usually not a joint limit but the Cartesian one: the tool runs at
  exactly 1.3500 m/s through the middle of every recorded move.
"""
from __future__ import annotations

import numpy as np
import torch

from utils import N_JOINTS, UR10e

# Identified from data/test-*.csv, peak per lead joint (rad/s, rad/s^2).
V_JOINT = np.array([1.885, 2.094, 2.765, 3.142, 3.142, 3.142])
A_JOINT = np.array([18.48, 15.00, 32.63, 40.90, 40.90, 40.90])
# The Cartesian caps read 1.350 m/s and 14.63 m/s^2 off the recorded ``target_qd``
# channel, but ``target_q`` advances ~2% faster than that channel says (the two are
# not consistent at 128 Hz). Everything downstream differentiates ``target_q``, so
# these are calibrated against it, which also lands the move duration exactly.
V_TCP = 1.359             # m/s
A_TCP = 14.84             # m/s^2
SMOOTH = 0.060            # s, the box the trapezoid's corners are rounded with
_UR = UR10e()


def limits(q0, q1):
    """Per-joint speed and acceleration ceilings for a move along ``q0 -> q1``.

    The Cartesian caps are what usually bind, so they are converted into joint
    terms with the tool speed per unit of path, taken at its worst pose. Both
    ``movej`` and the optimizer stay inside this envelope, which is what makes
    their durations comparable.
    """
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = q1 - q0
    cart = max(np.linalg.norm((_UR.jacobian(q0 + u * d) @ d)[:3])
               for u in np.linspace(0.0, 1.0, 16))
    # Tool speed per unit of path, turned into each joint's share of it. A joint
    # that does not travel gets no Cartesian share, so it keeps its own limit.
    share = np.where(np.abs(d) > 1e-9, np.abs(d) / max(cart, 1e-9), np.inf)
    return np.minimum(V_JOINT, share * V_TCP), np.minimum(A_JOINT, share * A_TCP)


def movej(q0, q1, dt: float, v: float = np.inf, a: float = np.inf) -> np.ndarray:
    """Commanded angles ``(n, N_JOINTS)`` for ``movej(q1, a, v)`` starting at ``q0``.

    A trapezoidal speed profile along the straight joint-space line, smoothed by a
    box filter ``SMOOTH`` seconds wide. That is what the recordings show: the corners
    are rounded (the acceleration ramps in over a fixed time, whatever its height)
    but the move still takes the trapezoid's duration, not the longer one a
    jerk-limited replan would need.

    The speed ceiling is the tightest of the requested ``v``, the per-joint limits
    and the Cartesian one, and likewise for the acceleration.
    """
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    v = v if v and v > 0 else np.inf
    a = a if a and a > 0 else np.inf
    d = q1 - q0
    if np.abs(d).max() < 1e-9:
        return q0[None].copy()

    # In units of the path parameter s, which runs 0 -> 1.
    reach = np.abs(d).clip(1e-12)
    v_lim, a_lim = limits(q0, q1)
    sd = min(v / np.abs(d).max(), (v_lim / reach).min())
    sdd = min(a / np.abs(d).max(), (a_lim / reach).min())
    sd = min(sd, np.sqrt(sdd))                # too short to ever reach the ceiling

    T = 1.0 / sd + sd / sdd                   # trapezoid: cruise + one ramp
    t = np.arange(int(round(T / dt)) + 1) * dt
    speed = np.clip(np.minimum(sdd * t, sdd * (T - t)), 0.0, sd)
    # An odd box keeps the smoothing centred, so the corners round without the move
    # getting longer. A move shorter than the box gets a box that fits.
    nb = min(2 * round(SMOOTH / dt / 2) + 1, 2 * (len(t) // 4) + 1)
    speed = np.convolve(speed, np.ones(nb) / nb, mode="same")
    s = np.cumsum(speed)
    return q0 + (s / s[-1])[:, None] * d


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
