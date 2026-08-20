"""Optimize a streamed joint path against the distilled reality-gap model.

``retime`` keeps every recorded waypoint and changes only its interval. ``reshape``
uses a small, direct, piecewise-linear set of waypoints between the same endpoints.
The model is always evaluated on its 8 ms training grid; linear interpolation turns
either variable-time path into that grid.
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

STEPS, LR, MIN_DT, BARRIER_WIDTH = 600, 0.01, 0.002, 0.05
CONTROL_POINT_FACTOR = 0.02
EXPLOITATION_EXPLORATION_FACTOR, BARRIER = 1.0, 10.0


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


def barrier(value, limit):
    """A smooth, increasingly steep cost around each physical limit."""
    ratio = value / torch.as_tensor(limit, dtype=value.dtype, device=value.device)
    return (BARRIER_WIDTH * F.softplus((ratio - 1) / BARRIER_WIDTH)).square().mean()


def penalties(q, robot):
    qd = (q[2:] - q[:-2]) / (2 * DT)
    qdd = (q[2:] - 2 * q[1:-1] + q[:-2]) / DT ** 2
    jac = torch.as_tensor(robot.jacobians(q.detach().cpu().numpy())[1:-1], dtype=q.dtype)
    tool = torch.linalg.vector_norm((jac @ qd[..., None])[..., 0], dim=1)
    return {
        "position": barrier(q.abs(), robot.q_joint),
        "joint_speed": barrier(qd.abs(), robot.v_joint),
        "joint_acceleration": barrier(qdd.abs(), robot.a_joint),
        "tool_speed": barrier(tool, robot.v_tcp),
    }


def objective(model, robot, q, dt, cycle_time_per_gap_rmse):
    path = resample(q, dt)
    gap, var = model.gap(path)
    gap_rmse = torch.sqrt((gap.square() + EXPLOITATION_EXPLORATION_FACTOR * var).mean())
    costs = penalties(path, robot)
    penalty = sum(costs.values())
    total = dt.sum() + cycle_time_per_gap_rmse * gap_rmse + BARRIER * penalty
    return total, gap_rmse, penalty, costs


def nodes(reference, mode):
    if mode == "retime":
        return reference
    k = max(2, round(CONTROL_POINT_FACTOR * len(reference)))
    return reference[torch.linspace(0, len(reference) - 1, k).round().long()]


def snapshot(total, q, dt, gap, penalty):
    return {
        "total": float(total.detach()),
        "q": q.detach().numpy(),
        "dt": dt.detach().numpy(),
        "gap": float(gap.detach()),
        "penalty": float(penalty.detach()),
    }


def optimize(model, q_ref, dt_ref, robot, mode="retime", run=None):
    ref = torch.as_tensor(np.asarray(q_ref, np.float32))
    q0 = nodes(ref, mode)
    moving = (q0[1:] - q0[:-1]).abs().amax(1) > 1e-7
    min_dt = (torch.full_like(moving, MIN_DT, dtype=torch.float32)
              if mode == "reshape" else MIN_DT * moving)
    base_dt = (torch.as_tensor(dt_ref, dtype=torch.float32) if mode == "retime"
               else torch.full((len(q0) - 1,), float(np.sum(dt_ref)) / (len(q0) - 1)))
    raw_dt = torch.nn.Parameter(inverse_softplus((base_dt - min_dt).clamp_min(1e-5)))
    interior = torch.nn.Parameter(q0[1:-1].clone()) if mode == "reshape" else None
    params = [raw_dt] + ([] if interior is None else [interior])
    opt = torch.optim.Adam(params, lr=LR)
    log, best = (SummaryWriter(run) if run else None), None
    with torch.no_grad():
        base_path = resample(q0, F.softplus(raw_dt) + min_dt)
        base_gap, base_var = model.gap(base_path)
        baseline = torch.sqrt(
            (base_gap.square() + EXPLOITATION_EXPLORATION_FACTOR * base_var).mean()).clamp_min(1e-8)
        cycle_time_per_gap_rmse = float(base_dt.sum() / baseline)
    for step in range(STEPS + 1):
        q = q0 if interior is None else torch.cat([q0[:1], interior, q0[-1:]])
        dt = F.softplus(raw_dt) + min_dt
        total, gap, penalty, costs = objective(
            model, robot, q, dt, cycle_time_per_gap_rmse)
        if best is None or float(total.detach()) < best["total"]:
            best = snapshot(total, q, dt, gap, penalty)
        if log:
            values = {"loss/total": total, "loss/gap_rmse": gap,
                      "loss/cycle_time": dt.sum(), "loss/penalties": penalty,
                      **{f"penalty/{k}": v for k, v in costs.items()}}
            for key, value in values.items():
                log.add_scalar(key, float(value.detach()), step)
        if step < STEPS:
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        if step % 100 == 0:
            print(f"    step {step:4d}  loss {float(total.detach()):6.3f}  gap "
                  f"{float(gap.detach()) * 1000:6.3f} mrad  T {float(dt.sum().detach()):5.3f}s  "
                  f"penalty {float(penalty.detach()):.4f}")
    if log:
        log.close()
    return best


def executable(q, dt, mode):
    """Drop only effectively-zero waits: URScript ``servoj`` time itself cannot be 0."""
    keep = np.ones(len(dt), bool)
    if mode == "retime":
        keep = ~((np.abs(q[1:] - q[:-1]).max(1) < 1e-7) & (dt < MIN_DT))
    return np.vstack([q[:1], q[1:][keep]]), dt[keep]


def main():
    ap = argparse.ArgumentParser(description="Optimize a path against the distilled model.")
    ap.add_argument("--path", required=True, help="path CSV from convert.py")
    ap.add_argument("--model", required=True, help="distilled model pickle")
    ap.add_argument("--robot", required=True, choices=list(Robot.MODELS))
    ap.add_argument("--mode", choices=("retime", "reshape"), default="retime")
    ap.add_argument("--out", default=None, help="default: <path>.<mode>.path")
    args = ap.parse_args()

    rows = np.asarray(load_path(args.path), float)
    if len(rows) < 3 or rows.shape[1] < N_JOINTS:
        raise SystemExit("path needs at least three joint setpoints")
    q_ref = rows[:, :N_JOINTS]
    dt_ref = rows[1:, N_JOINTS] if rows.shape[1] > N_JOINTS else np.full(len(rows) - 1, DT)
    if np.any(dt_ref <= 0):
        raise SystemExit("input path has a non-positive dt")
    model, robot = DistillModel.load(args.model), Robot(args.robot)
    if model.predicts() != ["actual_q"]:
        raise SystemExit("optimizer currently requires a model trained for actual_q")
    for p in model.parameters():
        p.requires_grad_(False)
    run = f"runs/optimize/{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"  {args.mode}: {len(q_ref)} input setpoints, {dt_ref.sum():.3f} s; logging to {run}")
    best = optimize(model, q_ref, dt_ref, robot, args.mode, run)
    q, dt = executable(best["q"], best["dt"], args.mode)
    out = args.out or args.path.rsplit(".", 1)[0] + f".{args.mode}.path"
    write_path(out, q, np.r_[MIN_DT, dt])
    print(f"\n  gap RMSE {best['gap'] * 1000:.3f} mrad, penalty {best['penalty']:.4f}, cycle {dt_ref.sum():.3f} -> "
          f"{dt.sum():.3f} s\n  wrote {out}: {len(q)} commands")


if __name__ == "__main__":
    main()
