"""Optimize a streamed joint path against the distilled reality-gap model.

Every recorded waypoint is kept; only its time interval is learned. The model is
always evaluated on its 8 ms training grid; linear interpolation turns the
variable-time path into that grid.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from convert import write_path
from send import load_path
from train_distillation_model import DistillModel
from utils import DT, N_JOINTS, Robot

STEPS = 2000
LR = 0.01
MIN_DT = 0.002
EXPLOITATION_EXPLORATION_FACTOR = 1.0
BARRIER_WEIGHT_START = 10.0
BARRIER_WEIGHT_END = 0.01
# A waypoint-to-waypoint step smaller than this isn't a distinct pose worth its own
# minimum dwell time -- it's below the model's own tracking-error scale (a few mrad),
# so it is free to shrink towards zero instead of floored at MIN_DT.
PAUSE_TOL = 1e-3


def inverse_softplus(x):
    return x + torch.log(-torch.expm1(-x))


def resample(q, dt):
    """Uniform-DT samples of piecewise-linear waypoints ``q`` and intervals ``dt``."""
    ends = torch.cat([torch.zeros(1), dt.cumsum(0)])
    t = torch.arange(max(3, int(torch.ceil(ends[-1].detach() / DT)) + 1)) * DT
    t = t.to(q).clamp_max(ends[-1])
    i = torch.searchsorted(ends.detach().contiguous(), t.contiguous(), right=True)
    i = i.clamp(1, len(q) - 1)
    f = ((t - ends[i - 1]) / (ends[i] - ends[i - 1]).clamp_min(1e-8))[:, None]
    return q[i - 1].lerp(q[i], f)


def barrier(value, limit, weight):
    """A true barrier: -log(1 - ratio), 0 at rest and +inf exactly at the limit,
    so a violation is never merely expensive. Clamped just short of 1 so a stray
    Adam step past the limit gives a huge but finite (not NaN) gradient -- the
    existing clip_grad_norm_ is what actually keeps that step small.
    """
    ratio = (value / torch.as_tensor(limit, dtype=value.dtype, device=value.device))
    return (weight * -torch.log1p(-ratio.clamp(max=1 - 1e-6))).mean()


def penalties(q, robot, weight):
    qd = (q[2:] - q[:-2]) / (2 * DT)
    qdd = (q[2:] - 2 * q[1:-1] + q[:-2]) / DT ** 2
    jac = torch.as_tensor(robot.jacobians(q.detach().cpu().numpy())[1:-1], dtype=q.dtype)
    tool = torch.linalg.vector_norm((jac @ qd[..., None])[..., 0], dim=1)
    return {
        "position": barrier(q.abs(), robot.q_joint, weight),
        "joint_speed": barrier(qd.abs(), robot.v_joint, weight),
        "joint_acceleration": barrier(qdd.abs(), robot.a_joint, weight),
        "tool_speed": barrier(tool, robot.v_tcp, weight),
    }


def measures(model, robot, q, weight=BARRIER_WEIGHT_START):
    gap, var = model.gap(q)
    gap_rmse = torch.sqrt((gap.square() + EXPLOITATION_EXPLORATION_FACTOR * var).mean())
    costs = penalties(q, robot, weight)
    penalty = sum(costs.values())
    return gap_rmse, penalty, costs


def objective(model, robot, q, cycle_time, cycle_time_per_gap_rmse, weight=BARRIER_WEIGHT_START):
    gap_rmse, penalty, costs = measures(model, robot, q, weight)
    cycle_time = torch.as_tensor(cycle_time, dtype=q.dtype, device=q.device)
    total = cycle_time + cycle_time_per_gap_rmse * gap_rmse + penalty
    return total, gap_rmse, penalty, costs


def optimize(model, q_ref, dt_ref, robot, run=None, start_dt=DT):
    q0 = torch.as_tensor(np.asarray(q_ref, np.float32))
    moving = (q0[1:] - q0[:-1]).abs().amax(1) > PAUSE_TOL
    min_dt = MIN_DT * moving
    base_dt = torch.as_tensor(dt_ref, dtype=torch.float32)
    raw_dt = torch.nn.Parameter(inverse_softplus((base_dt - min_dt).clamp_min(1e-5)))
    opt = torch.optim.Adam([raw_dt], lr=LR)
    log, best = (SummaryWriter(run) if run else None), None
    with torch.no_grad():
        baseline, _, _ = measures(model, robot, resample(q0, base_dt))
        cycle_time_per_gap_rmse = float((base_dt.sum() + start_dt) / baseline.clamp_min(1e-8))
    for step in range(STEPS + 1):
        # The barrier weight starts high (smooth, keeps the path well clear of every
        # limit) and decays to a thousandth of that by 80% of the run, well before
        # the end -- so the last stretch is essentially free to optimize cycle time
        # and gap alone. Never exactly 0: -log(1-ratio) must stay in the loss, or a
        # true violation would go unpunished. Reuses BARRIER_WEIGHT_START/END and
        # STEPS, no extra hyperparameter.
        weight = BARRIER_WEIGHT_START + step / (0.8 * STEPS) * (BARRIER_WEIGHT_END - BARRIER_WEIGHT_START) if step < 0.8 * STEPS else BARRIER_WEIGHT_END
        dt = F.softplus(raw_dt) + min_dt
        path = resample(q0, dt)
        total_loss, gap, penalty, costs = objective(
            model, robot, path, dt.sum() + start_dt, cycle_time_per_gap_rmse, weight)
        loss = total_loss - penalty       # the true objective alone, without the barrier
        if best is None or float(total_loss.detach()) < best["total"]:
            best = {"total": float(total_loss.detach()), "q": q0.numpy(), "dt": dt.detach().numpy(),
                    "gap": float(gap.detach()), "penalty": float(penalty.detach())}
        if log:
            values = {"loss/loss": loss, "loss/total_loss": total_loss, "loss/gap_rmse": gap,
                      "loss/cycle_time": dt.sum(), "loss/penalties": penalty,
                      **{f"penalty/{k}": v for k, v in costs.items()}}
            for key, value in values.items():
                log.add_scalar(key, float(value.detach()), step)
        if step < STEPS:
            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([raw_dt], 1.0)
            opt.step()
        if step % 100 == 0:
            print(f"    step {step:4d}  loss {float(loss.detach()):6.3f}  total_loss "
                  f"{float(total_loss.detach()):6.3f}  gap {float(gap.detach()) * 1000:6.3f} mrad  "
                  f"T {float(dt.sum().detach()):5.3f}s  penalty {float(penalty.detach()):.4f}")
    if log:
        log.close()
    return best


def executable(q, dt):
    """Drop only effectively-zero waits: URScript ``servoj`` time itself cannot be 0."""
    keep = ~((np.abs(q[1:] - q[:-1]).max(1) <= PAUSE_TOL) & (dt < MIN_DT))
    return np.vstack([q[:1], q[1:][keep]]), dt[keep]


def main():
    ap = argparse.ArgumentParser(description="Optimize a path against the distilled model.")
    ap.add_argument("--path", required=True, help="path CSV from convert.py")
    ap.add_argument("--model", required=True, help="distilled model pickle")
    ap.add_argument("--robot", required=True, choices=list(Robot.MODELS))
    ap.add_argument("--out", default=None, help="default: <path>.retime.path")
    args = ap.parse_args()

    rows = np.asarray(load_path(args.path), float)
    if len(rows) < 3 or rows.shape[1] < N_JOINTS:
        raise SystemExit("path needs at least three joint setpoints")
    q_ref = rows[:, :N_JOINTS]
    path_dt = rows[:, N_JOINTS] if rows.shape[1] > N_JOINTS else np.full(len(rows), DT)
    if np.any(path_dt <= 0):
        raise SystemExit("input path has a non-positive dt")
    start_dt, dt_ref = path_dt[0], path_dt[1:]
    model, robot = DistillModel.load(args.model), Robot(args.robot)
    if model.predicts() != ["actual_q"]:
        raise SystemExit("optimizer currently requires a model trained for actual_q")
    for p in model.parameters():
        p.requires_grad_(False)
    run = f"runs/optimize/{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"  {len(q_ref)} input setpoints, {path_dt.sum():.3f} s; logging to {run}")
    best = optimize(model, q_ref, dt_ref, robot, run, start_dt)
    q, dt = executable(best["q"], best["dt"])
    # Written on the model's own uniform DT grid, not the learned variable dt: a
    # servoJ streamer can only tick at one fixed rate (see send.py's ur_rtde engine),
    # and this is exactly the discretization the objective was scored on anyway.
    final = resample(torch.as_tensor(q, dtype=torch.float32),
                     torch.as_tensor(dt, dtype=torch.float32)).numpy()
    out = args.out or args.path.rsplit(".", 1)[0] + ".retime.path"
    write_path(out, final, DT)
    print(f"\n  gap RMSE {best['gap'] * 1000:.3f} mrad, penalty {best['penalty']:.4f}, cycle {path_dt.sum():.3f} -> "
          f"{start_dt + dt.sum():.3f} s\n  wrote {out}: {len(final)} commands")


if __name__ == "__main__":
    main()
