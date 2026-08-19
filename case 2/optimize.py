"""Optimize a recorded path by differentiating a score through the distilled model.

convert.py gives the trajectory the controller really commands. This bends it:

    t, u, T  = phase(theta)                     each slice of the path gets its own time
    q(t)     = path(u) + bspline(u, offset)     the reference plus a free offset
    actual   = q + DistillModel(features(q))    the learned reality gap
    loss     = |actual - q| + K*sd + ALPHA*T + limits + drift

All of it is torch, so one ``backward()`` moves the shape of the trajectory, where
its time goes, and how long it takes. Both parameters start at zero, which
reproduces the recorded path exactly -- so the optimizer can leave it alone if that
is already best, and every number is reported against it.

Steps go to ``runs/optimize/<stamp>``: ``loss/total`` (without the limit penalty,
which would swamp it), ``loss/error`` in rad, ``loss/time`` in s, and
``loss/limits``, the overshoot of the ceilings, 0 meaning the controller can run it.

    python optimize.py --path scripts/triangle.path --model models/distill.pkl
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from common import X_CLIP, features
from convert import write_path
from send import load_path
from train_distillation_model import DistillModel
from utils import A_JOINT, N_JOINTS, UR10e, V_JOINT, V_TCP, tcp_speed

ALPHA = 1.0        # a second of cycle time costs as much as this much error, relatively
K = 1.0            # standard deviations of the model's own uncertainty added to the error
STAY = 100.0       # how hard the path is held to the recorded one
LIMIT = 1000.0     # how hard the speed, acceleration and tool-speed ceilings are held
KNOTS, POINTS, STEPS, LR = 24, 16, 600, 0.01


def bspline(u, points):
    """Uniform cubic B-spline ``q(u)`` for ``u`` in [0,1], control ``points`` (m, 6).

    Differentiable in both, which is why the basis is spelled out rather than taken
    from scipy: ``u`` itself carries gradient. Repeating an end point three times
    pins the curve to it with zero velocity and acceleration.
    """
    m = points.shape[0]
    x = u.clamp(0.0, 1.0) * (m - 3)
    k = x.floor().clamp(0, m - 4).long()
    s = (x - k)[:, None]
    b = torch.stack([(1 - s) ** 3, 3 * s ** 3 - 6 * s ** 2 + 4,
                     -3 * s ** 3 + 3 * s ** 2 + 3 * s + 1, s ** 3]) / 6.0
    return sum(b[i] * points[k + i] for i in range(4))


def zeroed(interior):
    """Control points for an offset that starts and ends at nothing."""
    return torch.cat([torch.zeros(3, N_JOINTS), interior, torch.zeros(3, N_JOINTS)])


def resample(q, dt_from: float, dt_to: float):
    """The same motion on a different sample grid: same duration, new spacing."""
    t = np.arange(len(q)) * dt_from
    grid = np.arange(0.0, t[-1] + 1e-9, dt_to)
    return np.column_stack([np.interp(grid, t, c) for c in np.asarray(q).T])


def sample(ref, u):
    """Linear interpolation of ``ref`` (m, c) at phases ``u`` in [0,1]."""
    x = u.clamp(0.0, 1.0) * (len(ref) - 1)
    i = x.floor().long().clamp(0, len(ref) - 2)
    f = (x - i)[:, None]
    return ref[i] * (1 - f) + ref[i + 1] * f


def phase(theta, T0: float, dt: float):
    """``(t, u, T)``: the time grid, the phase along the path, and the cycle time.

    Slice k of the path gets ``exp(theta_k)`` times the time it takes in the
    reference, so all-zeros replays it unchanged. The cycle time is what the slices
    add up to rather than a parameter of its own, which is what lets a pause be cut
    without touching the moves: one global duration would speed those up too and run
    straight into the tool-speed cap, and the optimizer could never get started.
    """
    tk = torch.cat([torch.zeros(1), torch.cumsum((T0 / len(theta)) * theta.exp(), 0)])
    t = torch.arange(int(tk[-1].item() / dt) + 1, dtype=torch.float32) * dt
    uk = torch.linspace(0.0, 1.0, len(theta) + 1)
    i = torch.searchsorted(tk.detach().contiguous(), t.contiguous()).clamp(1, len(theta))
    u = uk[i - 1] + (t - tk[i - 1]) / (tk[i] - tk[i - 1]).clamp_min(1e-6) / len(theta)
    return t, u.clamp(0.0, 1.0), tk[-1]


def error(model, q, dt: float):
    """Per-row ``|actual_q - q|`` (rad), widened by ``K`` sd of the model's own
    uncertainty so that trajectories it has never seen are not free."""
    mx, sx, my, sy = (torch.as_tensor(s, dtype=torch.float32) for s in model.stats)
    x = ((features(q, dt, model.pad) - mx) / sx).clamp(-X_CLIP, X_CLIP)
    mu, lv = model.forward(x.T[None])
    gap = (mu.mean(0)[0].T * sy + my).abs()
    var = (torch.exp(lv).mean(0) + mu.var(0, unbiased=False))[0].T * sy ** 2
    return gap + K * var.sqrt()


def penalty(q, dt: float, jac):
    """Overshoot of the speed, acceleration and tool-speed ceilings, as a fraction.

    0 for a path the controller can run as written. Explicit rather than left to the
    model: outside the envelope the model extrapolates, and a confident wrong answer
    there costs nothing. Mean and worst row together, so one bad row is worth fixing.
    """
    qd = (q[2:] - q[:-2]) / (2 * dt)
    qdd = (q[2:] - 2 * q[1:-1] + q[:-2]) / dt ** 2
    over = lambda z, lim: (lambda e: e.mean() + e.max())(
        torch.relu(z.abs() / torch.as_tensor(lim, dtype=torch.float32) - 1))
    tool = torch.linalg.norm((jac[1:-1] @ qd[..., None])[..., 0], dim=1)
    return over(qd, V_JOINT) + over(qdd, A_JOINT) + over(tool, V_TCP)


def plan(model, q_ref, dt: float, run: str = None):
    """Optimize a whole path. Returns the best ``(q, T, error)`` seen, and the
    reference's own error for comparison."""
    ref = torch.as_tensor(np.asarray(q_ref, np.float32))
    T0 = (len(ref) - 1) * dt
    # Where the reference stands still it is holding a waypoint, which is the task
    # itself; the path is held ten times harder there than along a move.
    hold = 1.0 + 9.0 * torch.as_tensor(
        (np.abs(np.gradient(q_ref, dt, axis=0)).max(1) < 0.01).astype(np.float32))
    ur = UR10e()
    with torch.no_grad():
        before = float(error(model, ref, dt).mean())

    offset = torch.zeros(POINTS, N_JOINTS, requires_grad=True)
    theta = torch.zeros(KNOTS, requires_grad=True)
    opt = torch.optim.Adam([offset, theta], lr=LR)
    log = SummaryWriter(run) if run else None
    best = None
    for i in range(STEPS + 1):
        t, u, T = phase(theta, T0, dt)
        q = sample(ref, u) + bspline(u, zeroed(offset))
        err = error(model, q, dt).mean()
        # The Jacobian is refreshed each step from the detached poses, so the tool
        # speed is right even when the offset grows.
        pen = penalty(q, dt, torch.as_tensor(
            np.array([ur.jacobian(p)[:3] for p in q.detach().numpy()], np.float32)))
        off = q - sample(ref, u)
        drift = ((off ** 2).mean(1) * sample(hold[:, None], u)[:, 0]).mean()
        total = err / before + ALPHA * T / T0 + STAY * drift
        loss = total + LIMIT * pen
        if best is None or float(loss.detach()) < best[0]:
            best = (float(loss.detach()), q.detach().numpy(), float(T.detach()),
                    float(err.detach()), float(off.abs().max().detach()))
        if log:
            for key, v in (("loss/total", total), ("loss/error", err),
                           ("loss/time", T), ("loss/limits", pen)):
                log.add_scalar(key, float(v.detach()), i)
        if i < STEPS:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # The limit penalty is stiff: touching a ceiling produces a huge
            # gradient, and without this Adam overshoots and never settles.
            torch.nn.utils.clip_grad_norm_([offset, theta], 1.0)
            opt.step()
        if i % 100 == 0:
            print(f"    step {i:4d}  loss {float(total.detach()):6.3f}  err "
                  f"{float(err.detach()) * 1000:6.3f} mrad  T {float(T.detach()):5.3f}s  "
                  f"over limits {float(pen.detach()) * 100:5.2f}%")
    if log:
        log.close()
    return best[1], best[2], best[3], best[4], before


def main():
    ap = argparse.ArgumentParser(description="Optimize a recorded path against the model.")
    ap.add_argument("--path", default="scripts/triangle.path", help="path CSV from convert.py")
    ap.add_argument("--model", default="models/distill.pkl", help="distilled model pickle")
    ap.add_argument("--out", default=None, help="default: <path>.optimized.path")
    args = ap.parse_args()

    model = DistillModel.load(args.model)
    for p in model.parameters():
        p.requires_grad_(False)              # only the trajectory is optimized
    dt = model.train_dt
    rows = np.array(load_path(args.path), float)
    q_ref = resample(rows[:, :N_JOINTS], rows[0, 6] if rows.shape[1] > 6 else dt, dt)
    run = f"runs/optimize/{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"  {len(rows)} setpoints -> {len(q_ref)} at {dt * 1000:.2f} ms "
          f"({(len(q_ref) - 1) * dt:.2f} s); logging to {run}")

    q, T, after, off, before = plan(model, q_ref, dt, run)
    T0 = (len(q_ref) - 1) * dt
    score, base = after / before + ALPHA * T / T0, 1 + ALPHA
    v = tcp_speed(q, dt).max()
    print(f"\n  error {before * 1000:.3f} -> {after * 1000:.3f} mrad   cycle {T0:.3f} -> {T:.3f} s"
          f"\n  score {score:.3f} vs {base:.3f} for the recorded path: "
          f"{'better' if score < base else 'NOT an improvement'}"
          f"\n  up to {off * 1000:.1f} mrad off it, peak tool speed {v:.3f} m/s"
          f"{'  ** over the cap' if v > V_TCP else ''}")

    out = args.out or args.path.rsplit(".", 1)[0] + ".optimized.path"
    write_path(out, q, dt)
    print(f"wrote {out} ({len(q)} setpoints)\n"
          f"  run it: python send.py --path {out} --loop 5 --out optimized.csv")


if __name__ == "__main__":
    main()
