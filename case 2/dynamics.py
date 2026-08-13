"""Commanded torque and current for a candidate motion, without re-running URSim.

For a move (fixed joint geometry) and a candidate speed profile, produces the
commanded trajectory (q, qd, qdd) and the commanded joint current, from the UR10e
inverse dynamics:

    tau     = M(q) qdd + g(q)          (utils.UR10e; Coriolis dropped by default)
    current = tau / Kt                 (Kt fit from the recording: moment / current)

Units: the ``vel``/``acc`` registers (and the URScript numbers) are deg/s and
deg/s^2; joint speeds in the recording are rad/s. The trajectory is built in rad,
with the speed clamped to the joint limit (URScript clamps a movej speed above the
limit).

``Dynamics`` is an interface; subclass it for a different torque model (friction,
Coriolis, identified inertial parameters, a learned model). The default is fast
because the pose-dependent terms are precomputed along each move's geometry once
(they do not depend on the speed profile).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from utils import (ACC_COL, N_JOINTS, TIME_COL, VEL_COL, UR10e, get_block,
                   joint_cols)

DEG2RAD = np.pi / 180.0
MAX_JOINT_SPEED = np.pi          # rad/s: URScript clamps a movej speed to this
MAX_JOINT_ACC = 4.0 * np.pi      # rad/s^2: a generous joint acceleration ceiling
MAX_JOINT_JERK = 20.0 * np.pi    # rad/s^3: generous jerk ceiling (constant-jerk ramp time acc/jerk ~0.2s at MAX_JOINT_ACC)
GRID = 50                        # samples along a move's geometry for the M,g cache


def fit_kt(rec) -> np.ndarray:
    """Per-joint torque-to-current constant ``Kt`` from a recording.

    ``target_moment = Kt * target_current`` per joint (motor and gearing). Fit the
    slope through the origin from the recorded pair, so torque can be converted to
    current in the same units the recording uses. Joints with no current in the
    run fall back to 1.0 (they carry no information to fit).
    """
    moment = get_block(rec.df, "target_moment")
    current = get_block(rec.df, "target_current")
    num = (moment * current).sum(axis=0)
    den = (current * current).sum(axis=0)
    return np.where(den > 1e-9, num / np.where(den > 1e-9, den, 1.0), 1.0)


def trapezoidal(distance: float, vel: float, acc: float, dt: float) -> np.ndarray:
    """Progress ``s(t)`` in [0,1] for a joint move of ``distance`` (rad).

    Standard trapezoidal profile at peak speed ``vel`` (rad/s) and acceleration
    ``acc`` (rad/s^2), sampled at ``dt`` (s); triangular if it never reaches
    ``vel``. All joints of a movej share this one profile (they start and stop
    together), so ``q(t) = start + s(t) * (dest - start)``. Kept alongside
    ``s_curve`` (same signature) so the two speed-profile strategies can be
    compared directly, e.g. via ``GapEnv(..., profile=trapezoidal)``.
    """
    if distance <= 1e-9 or vel <= 0 or acc <= 0:
        return np.array([0.0, 1.0])                  # degenerate: no motion
    t_acc = vel / acc
    d_acc = 0.5 * acc * t_acc ** 2
    if 2 * d_acc >= distance:                        # triangular: never reaches vel
        t_acc = np.sqrt(distance / acc)
        d_acc = 0.5 * distance
        t_flat = 0.0
    else:
        t_flat = (distance - 2 * d_acc) / vel
    total = 2 * t_acc + t_flat
    t = np.arange(0.0, total + dt, dt)
    d = np.where(t < t_acc, 0.5 * acc * t ** 2,
                 np.where(t < t_acc + t_flat, d_acc + vel * (t - t_acc),
                          distance - 0.5 * acc * np.clip(total - t, 0, None) ** 2))
    return np.clip(d, 0.0, distance) / distance


def _accel_phase(v_peak: float, acc: float, jerk: float):
    """Shape of a 0 -> ``v_peak`` jerk-limited ramp: jerk up to ``a_peak``, hold
    it for ``t_a``, jerk back to 0, arriving at ``v_peak`` exactly.

    ``a_peak`` is ``acc``, or reduced to ``sqrt(v_peak * jerk)`` if ``v_peak``
    is too small for the ramp to ever reach ``acc`` (a jerk-limited triangle,
    ``t_a`` = 0). Returns ``(a_peak, t_j, t_a, duration, distance)``.
    """
    v_peak = max(v_peak, 0.0)
    a_peak = min(acc, np.sqrt(v_peak * jerk)) if v_peak > 0 else 0.0
    t_j = a_peak / jerk if jerk > 0 else 0.0
    t_a = max((v_peak - a_peak * t_j) / a_peak, 0.0) if a_peak > 1e-12 else 0.0
    v1 = 0.5 * a_peak * t_j                           # velocity at the end of the jerk-up phase
    d1 = a_peak * t_j ** 2 / 6.0                       # distance at the end of the jerk-up phase
    d2 = v1 * t_a + 0.5 * a_peak * t_a ** 2            # distance added by the constant-accel phase
    v2 = v1 + a_peak * t_a                             # velocity at the end of the constant-accel phase
    d3 = v2 * t_j + a_peak * t_j ** 2 / 3.0            # distance added by the jerk-down phase
    return a_peak, t_j, t_a, 2 * t_j + t_a, d1 + d2 + d3


def _ramp_profile(t: np.ndarray, a_peak: float, t_j: float, t_a: float) -> np.ndarray:
    """Distance covered by time ``t`` (array, clipped to the ramp's own
    duration) into a 0 -> ``v_peak`` jerk-limited ramp shaped by ``_accel_phase``.
    """
    jerk = a_peak / t_j if t_j > 0 else 0.0
    v1 = 0.5 * a_peak * t_j
    d1 = a_peak * t_j ** 2 / 6.0
    d2 = d1 + v1 * t_a + 0.5 * a_peak * t_a ** 2
    v2 = v1 + a_peak * t_a
    u2 = np.clip(t - t_j, 0.0, t_a)
    u3 = np.clip(t - t_j - t_a, 0.0, t_j)
    q1 = jerk * np.clip(t, 0.0, t_j) ** 3 / 6.0
    q2 = d1 + v1 * u2 + 0.5 * a_peak * u2 ** 2
    q3 = d2 + v2 * u3 + 0.5 * a_peak * u3 ** 2 - jerk * u3 ** 3 / 6.0
    return np.where(t < t_j, q1, np.where(t < t_j + t_a, q2, q3))


def s_curve(distance: float, vel: float, acc: float, dt: float,
            jerk: float = MAX_JOINT_JERK) -> np.ndarray:
    """Progress ``s(t)`` in [0,1] for a joint move of ``distance`` (rad),
    jerk-limited.

    Same accel-cruise-decel shape as a trapezoidal profile, but the accel/decel
    ramps are jerk-limited (jerk up, hold ``acc``, jerk down) instead of an
    instantaneous acceleration step, so acceleration is continuous -- closer to
    what a real UR controller does (see ``analysis.py``'s duration mismatch on
    the swing scripts, ~25% under a plain trapezoid). Sampled at ``dt`` (s).
    All joints of a movej share this one profile, so
    ``q(t) = start + s(t) * (dest - start)``.

    Degrades exactly like a trapezoidal profile: a jerk-limited triangle (no
    constant-acceleration plateau) if ``vel`` is never reached under ``acc``,
    and -- one derivative smoother -- a reduced peak speed (found by bisection,
    since a ramp's distance grows monotonically with its peak speed) if even a
    single ramp up and back down would overshoot ``distance``.
    """
    if distance <= 1e-9 or vel <= 0 or acc <= 0 or jerk <= 0:
        return np.array([0.0, 1.0])                  # degenerate: no motion
    a_peak, t_j, t_a, t_ramp, d_ramp = _accel_phase(vel, acc, jerk)
    if 2 * d_ramp > distance:                         # a single up+down ramp overshoots
        lo, hi = 0.0, vel
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if 2 * _accel_phase(mid, acc, jerk)[4] < distance else (lo, mid)
        vel = 0.5 * (lo + hi)
        a_peak, t_j, t_a, t_ramp, d_ramp = _accel_phase(vel, acc, jerk)
    t_cruise = (distance - 2 * d_ramp) / vel if vel > 0 else 0.0
    total = 2 * t_ramp + t_cruise
    t = np.arange(0.0, total + dt, dt)
    up = _ramp_profile(np.clip(t, 0.0, t_ramp), a_peak, t_j, t_a)
    u3 = np.clip(t - t_ramp - t_cruise, 0.0, t_ramp)
    down = distance - _ramp_profile(t_ramp - u3, a_peak, t_j, t_a)
    d = np.where(t < t_ramp, up,
                 np.where(t < t_ramp + t_cruise,
                          d_ramp + vel * np.clip(t - t_ramp, 0.0, t_cruise),
                          down))
    return np.clip(d, 0.0, distance) / distance


class Dynamics(ABC):
    """Interface: commanded joint current for a candidate motion.

    ``current(q, qd, qdd)`` is the one thing a torque model must provide. The rest
    (differentiation, assembling a recording-like frame) is handled for you, so a
    student overrides just the physics.
    """

    @abstractmethod
    def current(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray,
                s: np.ndarray = None, key=None) -> np.ndarray:
        """Commanded joint current, ``(n, N_JOINTS)``, for a trajectory.

        ``q``/``qd``/``qdd`` are ``(n, N_JOINTS)`` in rad. ``s`` is each row's
        progress along the move geometry in [0,1]; ``key`` identifies the move.
        Both let an implementation reuse pose-dependent terms cached per move.
        """

    def frame(self, q: np.ndarray, dt: float, vel_deg: float, acc_deg: float,
              s: np.ndarray = None, key=None) -> pd.DataFrame:
        """Assemble a commanded-trajectory DataFrame for a candidate ``q(t)``.

        Differentiates ``q`` for ``qd``/``qdd``, asks ``current`` for the commanded
        current, and lays it out with the same columns a URSim recording has (the
        ``actual_*`` columns are zeros for the distill model to overwrite). ``dt``
        is the sample period; ``vel_deg``/``acc_deg`` are stored as the register
        columns (still deg/s). ``s`` is each row's progress along the move geometry
        in [0,1] (how the cached pose terms are looked up).
        """
        q = np.asarray(q, dtype=float)
        qd = np.gradient(q, dt, axis=0)
        qdd = np.gradient(qd, dt, axis=0)
        cur = self.current(q, qd, qdd, s=s, key=key)
        n = len(q)
        out = {TIME_COL: np.arange(n) * dt}
        for j in range(N_JOINTS):
            out[f"target_q{j}"] = q[:, j]
            out[f"target_qd{j}"] = qd[:, j]
            out[f"target_current{j}"] = cur[:, j]
            out[f"actual_current{j}"] = 0.0          # distill overwrites these
        out[VEL_COL] = float(vel_deg)
        out[ACC_COL] = float(acc_deg)
        return pd.DataFrame(out)


class UR10eDynamics(Dynamics):
    """Default torque model: ``tau = M(q) qdd + g(q)`` via ``utils.UR10e``.

    Fast because the pose-dependent terms ``M(q)`` and ``g(q)`` are precomputed on
    a grid along each move's geometry (they do not depend on the speed profile) and
    interpolated per candidate. Coriolis is dropped (small here, and its finite
    difference is expensive); add it in a subclass if you need it.
    """

    def __init__(self, rec, payload: float = 0.0, grid: int = GRID):
        self.ur = UR10e(payload=payload)
        self.kt = fit_kt(rec)
        self.grid = grid
        self._cache = {}          # key -> (s_grid, M_grid, g_grid)

    def prepare(self, key, q_geo: np.ndarray):
        """Precompute ``M(q)``, ``g(q)`` along a move's geometry ``q_geo`` (G, 6).

        Call once per move. ``q_geo`` is the joint path the move traces (a straight
        line for movej, the recorded curve for a re-timed path), sampled uniformly.
        Every candidate of that move reuses the result.
        """
        q_geo = np.asarray(q_geo, dtype=float)
        idx = np.linspace(0, len(q_geo) - 1, min(self.grid, len(q_geo))).round().astype(int)
        s_grid = idx / max(len(q_geo) - 1, 1)
        M_grid = np.array([self.ur.mass_matrix(q_geo[i]) for i in idx])
        g_grid = np.array([self.ur.gravity(q_geo[i]) for i in idx])
        self._cache[key] = (s_grid, M_grid, g_grid, q_geo)
        return self

    def current(self, q, qd, qdd, s=None, key=None) -> np.ndarray:
        q, qd, qdd = np.asarray(q), np.asarray(qd), np.asarray(qdd)
        if key is not None and key in self._cache and s is not None:
            # Interpolate the cached pose terms at each row's progress along the move.
            s_grid, M_grid, g_grid, _ = self._cache[key]
            s = np.asarray(s, dtype=float)
            M = np.empty((len(q), N_JOINTS, N_JOINTS))
            for a in range(N_JOINTS):
                for b in range(N_JOINTS):
                    M[:, a, b] = np.interp(s, s_grid, M_grid[:, a, b])
            g = np.column_stack([np.interp(s, s_grid, g_grid[:, a]) for a in range(N_JOINTS)])
        else:
            # No cache: compute per row (correct but slow; fine for a few candidates).
            M = np.array([self.ur.mass_matrix(row) for row in q])
            g = np.array([self.ur.gravity(row) for row in q])
        tau = np.einsum("nij,nj->ni", M, qdd) + g
        return tau / self.kt


def default_dynamics(rec) -> Dynamics:
    """The dynamics model the pipeline uses. Swap the return to change it globally."""
    return UR10eDynamics(rec)
