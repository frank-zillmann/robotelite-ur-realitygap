"""Bend a movej into a better commanded trajectory by differentiating through the model.

The whole chain -- spline control points -> commanded angles -> model inputs ->
predicted actual angles -> score -- is one torch graph, so the score's gradient
reaches the control points directly. No agent, no rollouts:

    q(t)      = bspline(t / T, points)              free, except pinned ends
    features  = sin/cos q, qd, qdd, qddd            differentiated from q
    actual_q  = target_q + DistillModel(features)   the learned reality gap
    loss      = |actual_q - target_q| + k*sd  +  w * T

The first term is the tracking error the robot would really show (the model's own
uncertainty added on top, so the optimizer cannot win by wandering into
trajectories the model has never seen); the second is what a fast move costs.
``T`` is optimized alongside the shape, so nothing forces a duration.

    python optimize.py --script scripts/triangle.script --model models/distill.pkl
"""
from __future__ import annotations

import argparse
import csv
import numpy as np
import torch

from analysis import Recording, _waypoints
from common import X_CLIP, features
from motion import V_TCP, bspline, clamp, limits, movej, tcp_speed
from train_distillation_model import DistillModel
from utils import N_JOINTS, load_script

SETTLE = 1.0          # s of holding the goal that the score sees: where the ring is
LIMIT_W = 1000.0      # weight on exceeding a joint's speed/acceleration ceiling


def trajectory(points, T, dt: float, settle: float = SETTLE):
    """Commanded angles on the model's time grid, the goal held for ``settle`` s.

    ``u = t/T`` is clamped to 1 by ``bspline``, so the held rows are exactly the
    goal and carry no gradient -- but the ring-down into them does.
    """
    n = int((float(T) + settle) / dt) + 1
    return bspline(torch.arange(n, dtype=torch.float32) * dt / T, points)


def error(model, q, dt: float, k: float):
    """Risk-averse ``|actual_q - q|`` per row (rad): the gap the model predicts,
    widened by ``k`` standard deviations of its own uncertainty."""
    mx, sx, my, sy = (torch.as_tensor(s, dtype=torch.float32) for s in model.stats)
    x = ((features(q, dt, model.pad) - mx) / sx).clamp(-X_CLIP, X_CLIP)
    mu, lv = model.forward(x.T[None])
    gap = mu.mean(0)[0].T * sy + my
    var = (torch.exp(lv).mean(0) + mu.var(0, unbiased=False))[0].T * sy ** 2
    return gap.abs() + k * var.sqrt()


def penalty(q, dt: float, v_lim, a_lim):
    """Mean overshoot of the move's speed and acceleration ceilings, as a fraction.

    0 inside the limits, so a feasible move pays nothing. The ceilings come from
    ``motion.limits``, i.e. they already carry the Cartesian caps, which keeps the
    result executable rather than something the controller would slow down.
    """
    qd = (q[2:] - q[:-2]) / (2 * dt)
    qdd = (q[2:] - 2 * q[1:-1] + q[:-2]) / dt ** 2
    over = lambda z, lim: torch.relu(z.abs() / torch.as_tensor(lim, dtype=torch.float32) - 1)
    return over(qd, v_lim).mean() + over(qdd, a_lim).mean()


def detour(q, q0, q1):
    """How far the path wanders off the straight line, relative to its length."""
    d = torch.as_tensor(np.asarray(q1 - q0, np.float32))
    off = q - torch.as_tensor(np.asarray(q0, np.float32))
    return ((off - (off @ d)[:, None] * d / (d @ d)) ** 2).sum(1).mean() / (d @ d)


def plan(model, q0, q1, dt: float, n_interior: int = 8, steps: int = 300,
         k: float = 1.0, alpha: float = 1.0, straight: float = 1000.0,
         lr: float = 0.02, quiet: bool = False):
    """Optimize one move from ``q0`` to ``q1``. Returns ``(q, T, before, after)``.

    ``before`` is the controller's own ``movej`` scored by the same model, so the
    two numbers are comparable. ``alpha`` sets the trade: error and duration both
    enter relative to that baseline, so ``alpha`` is how much a percent of cycle
    time is worth in percent of tracking error (0 = accuracy at any duration).
    """
    v_lim, a_lim = limits(q0, q1)
    base = movej(q0, q1, dt)                          # the controller's own move
    T = torch.tensor(float(len(base) * dt))
    with torch.no_grad():
        held = np.vstack([base, np.repeat(base[-1:], int(SETTLE / dt), axis=0)])
        before = float(error(model, held, dt, k).mean())
    # Start on the straight line, evenly spaced: a smooth ease from q0 to q1, close
    # to the movej it replaces but without its corners.
    u = torch.linspace(0.0, 1.0, n_interior + 2)[1:-1, None]
    ends = lambda q: torch.as_tensor(q, dtype=torch.float32)
    inner = (ends(q0) + u * (ends(q1) - ends(q0))).requires_grad_(True)
    log_T = T.log().clone().requires_grad_(True)
    opt = torch.optim.Adam([inner, log_T], lr=lr)
    T0 = float(T)
    for i in range(steps):
        T = log_T.exp()
        q = trajectory(clamp(q0, q1, inner), T, dt)
        err = error(model, q, dt, k).mean()
        # All three terms are relative to the move being replaced, so ``alpha`` is
        # simply how many percent of error a percent of cycle time is worth.
        pen = penalty(q, dt, v_lim, a_lim)
        loss = err / before + alpha * T / T0 + LIMIT_W * pen + straight * detour(q, q0, q1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if not quiet and i % 50 == 0:
            print(f"    step {i:4d}  err {float(err) * 1000:7.3f} mrad  T {float(T):5.3f}s  "
                  f"over limits {float(pen) * 100:5.1f}%")
    with torch.no_grad():
        T = log_T.exp()
        q = trajectory(clamp(q0, q1, inner), T, dt)
        after = float(error(model, q, dt, k).mean())
    return q.detach().numpy(), float(T), before, after


def main():
    ap = argparse.ArgumentParser(description="Optimize a script's moves against the model.")
    ap.add_argument("--script", default="scripts/triangle.script", help="URScript to optimize")
    ap.add_argument("--model", default="models/distill.pkl", help="distilled model pickle")
    ap.add_argument("--out", default=None, help="path CSV (default: <script>.path)")
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="how much a percent of cycle time is worth in percent of "
                         "tracking error; above ~0.1 the movej it replaces wins, "
                         "because a trapezoid is time-optimal and a spline is smooth")
    ap.add_argument("--k", type=float, default=1.0,
                    help="standard deviations of model uncertainty added to the error")
    ap.add_argument("--straight", type=float, default=1000.0,
                    help="how hard the path is held to the straight line between the "
                         "waypoints; lower lets it detour to track better")
    ap.add_argument("--steps", type=int, default=300, help="optimizer steps per move")
    ap.add_argument("--points", type=int, default=8, help="free control points per move")
    args = ap.parse_args()

    model = DistillModel.load(args.model)
    for p in model.parameters():
        p.requires_grad_(False)               # only the trajectory is optimized
    dt = model.train_dt
    way = [w for w in _waypoints(load_script(args.script)) if w is not None]
    if len(way) < 2:
        raise SystemExit(f"{args.script}: need at least two movej waypoints")

    rows, tb, ta, eb, ea = [], 0.0, 0.0, 0.0, 0.0
    for i, (q0, q1) in enumerate(zip(way, way[1:])):
        print(f"  move {i + 1}/{len(way) - 1}")
        q, T, before, after = plan(model, q0, q1, dt, args.points, args.steps,
                                   args.k, args.alpha, args.straight)
        base_T = len(movej(q0, q1, dt)) * dt
        keep = q[:int(T / dt) + 1]                    # drop the settle hold
        rows += [[*r, dt] for r in keep]
        tb, ta = tb + base_T, ta + T
        eb, ea = eb + before, ea + after
        v = tcp_speed(keep, dt).max()
        # Both terms relative to the movej, so the baseline always scores 1 + alpha.
        rel = after / before + args.alpha * T / base_T
        print(f"    error {before * 1000:.3f} -> {after * 1000:.3f} mrad   "
              f"time {base_T:.3f} -> {T:.3f} s   score {rel:.3f} vs {1 + args.alpha:.3f} "
              f"{'better' if rel < 1 + args.alpha else 'WORSE than the movej'}"
              f"   tool {v:.3f} m/s{'  ** over the cap' if v > V_TCP else ''}")

    n = len(way) - 1
    print(f"\n  total  error {eb / n * 1000:.3f} -> {ea / n * 1000:.3f} mrad   "
          f"cycle {tb:.3f} -> {ta:.3f} s")
    out = args.out or args.script.rsplit(".", 1)[0] + ".path"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"q{j}" for j in range(N_JOINTS)] + ["dt"])
        w.writerows([f"{v:.6f}" for v in r] for r in rows)
    print(f"wrote {out} ({len(rows)} setpoints)\n  run it: python send.py --path {out} --out optimized.csv")


if __name__ == "__main__":
    main()
