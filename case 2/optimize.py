"""Optimize a recorded path by differentiating a score through the distilled model.

The path from convert.py is what the controller really commands. This bends it:

    u(t)      = phase(t, T, rate)                   monotone, retimes the cycle
    q(t)      = path(u) + bspline(u, offset)        the reference plus a free offset
    features  = sin/cos q, qd, qdd, qddd            differentiated from q
    actual_q  = q + DistillModel(features)          the learned reality gap
    loss      = |actual - q| + k*sd  +  a*T  +  limits  +  offset

Everything after the parameters is torch, so one ``backward()`` moves the shape of
the trajectory, where its time goes, and how long it takes. The parameters start at
zero, which reproduces the recorded path exactly -- so the optimizer can leave it
alone if it is already the best thing to do, and every number is reported against it.

    python optimize.py --path scripts/triangle.path --model models/distill.pkl
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from common import X_CLIP, features
from convert import write_path
from motion import A_JOINT, V_JOINT, V_TCP, bspline, clamp, tcp_speed
from send import load_path
from train_distillation_model import DistillModel
from utils import N_JOINTS, UR10e

LIMIT_W = 1000.0      # weight on exceeding a speed, acceleration or tool-speed limit


def resample(q, dt_from: float, dt_to: float):
    """The same motion on a different sample grid: same duration, new spacing."""
    t = np.arange(len(q)) * dt_from
    grid = np.arange(0.0, t[-1] + 1e-9, dt_to)
    return np.column_stack([np.interp(grid, t, c) for c in np.asarray(q).T])


def sample(ref, u):
    """Linear interpolation of ``ref`` (m, c) at phases ``u`` in [0,1]. Differentiable."""
    x = u.clamp(0.0, 1.0) * (len(ref) - 1)
    i = x.floor().long().clamp(0, len(ref) - 2)
    f = (x - i)[:, None]
    return ref[i] * (1 - f) + ref[i + 1] * f


def phase(t, T, rate):
    """Monotone time -> phase in [0,1].

    ``rate`` gives each equal slice of the path a share of the cycle, so all-zeros
    replays the reference at its own pace and a negative entry makes that stretch
    (a pause, say) shorter. Differentiable in both ``T`` and ``rate``.
    """
    w = torch.softmax(rate, 0)
    tk = torch.cat([torch.zeros(1), torch.cumsum(w, 0)]) * T
    uk = torch.linspace(0.0, 1.0, len(w) + 1)
    i = torch.searchsorted(tk.detach().contiguous(), t.contiguous()).clamp(1, len(w))
    return (uk[i - 1] + (t - tk[i - 1]) / (tk[i] - tk[i - 1]).clamp_min(1e-6)
            * (uk[i] - uk[i - 1])).clamp(0.0, 1.0)


def error(model, q, dt: float, k: float):
    """Risk-averse ``|actual_q - q|`` per row (rad): the gap the model predicts,
    widened by ``k`` standard deviations of its own uncertainty."""
    mx, sx, my, sy = (torch.as_tensor(s, dtype=torch.float32) for s in model.stats)
    x = ((features(q, dt, model.pad) - mx) / sx).clamp(-X_CLIP, X_CLIP)
    mu, lv = model.forward(x.T[None])
    gap = mu.mean(0)[0].T * sy + my
    var = (torch.exp(lv).mean(0) + mu.var(0, unbiased=False))[0].T * sy ** 2
    return gap.abs() + k * var.sqrt()


def penalty(q, dt: float, jac):
    """Mean overshoot of the speed, acceleration and tool-speed ceilings, as a fraction.

    0 for a trajectory the controller can execute as written, so a feasible path
    pays nothing. Explicit rather than left to the model: outside the envelope the
    model is extrapolating, and a confident wrong answer there is worth nothing.
    ``jac`` is the tool Jacobian at the current trajectory, refreshed every step from
    the detached poses: the value it gives is then right even when the offset grows,
    while the gradient still flows through the speed alone.
    """
    qd = (q[2:] - q[:-2]) / (2 * dt)
    qdd = (q[2:] - 2 * q[1:-1] + q[:-2]) / dt ** 2
    over = lambda z, lim: torch.relu(z.abs() / torch.as_tensor(lim, dtype=torch.float32) - 1)
    tool = torch.linalg.norm((jac[1:-1] @ qd[..., None])[..., 0], dim=1)
    return over(qd, V_JOINT).mean() + over(qdd, A_JOINT).mean() + over(tool, V_TCP).mean()


def plan(model, q_ref, dt: float, knots: int = 8, points: int = 16, steps: int = 300,
         k: float = 1.0, alpha: float = 1.0, stay: float = 100.0, lr: float = 0.01,
         quiet: bool = False):
    """Optimize a whole path. Returns ``(q, T, before, after)``.

    ``before`` is the recorded path scored by the same model, and the parameters
    start where they reproduce it, so ``after < before`` means a real improvement
    rather than a different starting point.
    """
    ref = torch.as_tensor(np.asarray(q_ref, np.float32))
    T0 = (len(ref) - 1) * dt
    # Where the reference stands still it is holding a waypoint, which is the task
    # itself; the path is held ten times harder there than along a move.
    hold = 1.0 + 9.0 * torch.as_tensor(
        (np.abs(np.gradient(q_ref, dt, axis=0)).max(1) < 0.01).astype(np.float32))
    ur = UR10e()
    jacobians = lambda z: torch.as_tensor(np.array([ur.jacobian(p)[:3] for p in z], np.float32))
    with torch.no_grad():
        before = float(error(model, ref, dt, k).mean())

    offset = torch.zeros(points, N_JOINTS, requires_grad=True)
    rate = torch.zeros(knots, requires_grad=True)
    log_T = torch.tensor(float(T0)).log().requires_grad_(True)
    opt = torch.optim.Adam([offset, rate, log_T], lr=lr)
    best = None
    for i in range(steps + 1):
        T = log_T.exp()
        t = torch.arange(int(float(T) / dt) + 1, dtype=torch.float32) * dt
        u = phase(t, T, rate)
        q = sample(ref, u) + bspline(u, clamp(np.zeros(N_JOINTS), np.zeros(N_JOINTS), offset))
        err = error(model, q, dt, k).mean()
        pen = penalty(q, dt, jacobians(q.detach().numpy()))
        # Everything is relative to the recorded path, which therefore scores
        # 1 + alpha: ``alpha`` is what a percent of cycle time is worth in percent
        # of tracking error, and ``stay`` how much straying from the path costs.
        drift = (((q - sample(ref, u)) ** 2).mean(1) * sample(hold[:, None], u)[:, 0]).mean()
        loss = err / before + alpha * T / T0 + LIMIT_W * pen + stay * drift
        if best is None or float(loss) < best[0]:
            with torch.no_grad():          # Adam wanders; keep the best iterate seen
                best = (float(loss), q.detach().numpy(), float(T), float(err),
                        float((q - sample(ref, u)).abs().max()))
        if i < steps:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if not quiet and i % 50 == 0:
            print(f"    step {i:4d}  loss {float(loss):6.3f}  err {float(err) * 1000:7.4f} mrad  "
                  f"T {float(T):5.3f}s  over limits {float(pen) * 100:5.2f}%  off path up to "
                  f"{float((q - sample(ref, u)).abs().max()) * 1000:6.2f} mrad")
    return best[1], best[2], before, best[3], best[4]


def main():
    ap = argparse.ArgumentParser(description="Optimize a recorded path against the model.")
    ap.add_argument("--path", default="scripts/triangle.path", help="path CSV from convert.py")
    ap.add_argument("--model", default="models/distill.pkl", help="distilled model pickle")
    ap.add_argument("--out", default=None, help="default: <path>.optimized.path")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="what a percent of cycle time is worth in percent of error")
    ap.add_argument("--k", type=float, default=1.0,
                    help="standard deviations of model uncertainty added to the error")
    ap.add_argument("--stay", type=float, default=100.0,
                    help="how hard the path is held to the recorded one, and its "
                         "waypoints ten times harder still")
    ap.add_argument("--steps", type=int, default=300, help="optimizer steps")
    ap.add_argument("--knots", type=int, default=8, help="free retiming knots")
    ap.add_argument("--points", type=int, default=16, help="free offset control points")
    args = ap.parse_args()

    model = DistillModel.load(args.model)
    for p in model.parameters():
        p.requires_grad_(False)              # only the trajectory is optimized
    rows = np.array(load_path(args.path), float)
    dt_ref = rows[0, 6] if rows.shape[1] > 6 else model.train_dt
    q_ref = resample(rows[:, :N_JOINTS], dt_ref, model.train_dt)
    print(f"  {len(rows)} setpoints at {dt_ref * 1000:.2f} ms -> {len(q_ref)} at "
          f"{model.train_dt * 1000:.2f} ms ({(len(q_ref) - 1) * model.train_dt:.2f} s)")

    q, T, before, after, off = plan(model, q_ref, model.train_dt, args.knots,
                                    args.points, args.steps, args.k, args.alpha,
                                    args.stay)
    T0 = (len(q_ref) - 1) * model.train_dt
    score, base = after / before + args.alpha * T / T0, 1 + args.alpha
    print(f"\n  error {before * 1000:.4f} -> {after * 1000:.4f} mrad   "
          f"cycle {T0:.3f} -> {T:.3f} s\n  score {score:.3f} vs {base:.3f} for the "
          f"recorded path: {'better' if score < base else 'NOT an improvement'}")
    v = tcp_speed(q, model.train_dt).max()
    print(f"  up to {off * 1000:.1f} mrad off the recorded path")
    print(f"  peak tool speed {v:.3f} m/s"
          f"{'  ** over the ' + str(V_TCP) + ' m/s cap' if v > V_TCP else ''}")

    out = args.out or args.path.rsplit(".", 1)[0] + ".optimized.path"
    write_path(out, q, model.train_dt)
    print(f"wrote {out} ({len(q)} setpoints)\n"
          f"  run it: python send.py --path {out} --out optimized.csv")


if __name__ == "__main__":
    main()
