"""Optimize a streamed joint path against the distilled reality-gap model.

Every recorded waypoint is kept; only its time interval is learned. The model is
always evaluated on its 8 ms training grid; linear interpolation turns the
variable-time path into that grid.
"""
from __future__ import annotations

import argparse
import math
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


BARRIER_EDGE = 1e-6   # distance from the limit, as a fraction of it, where the log caps off


def barrier(value, limit, weight):
    """log(limit / slack), slack = limit - value: 0 at rest, +inf at the limit.
    (The plain ``-log(slack)`` a reader might reach for first isn't anchored at
    0 -- it carries a per-quantity ``-log(limit)`` offset that, scaled by the
    decaying ``weight``, would quietly bias which training step looks "best".)

    Past ``edge`` short of the limit, swaps in that point's own tangent line
    (same value and slope) instead of the raw log -- ``max`` always keeps
    whichever is valid, so the barrier stays finite with a live, restoring
    gradient even on a step that overshoots the limit.
    """
    limit = torch.as_tensor(limit, dtype=value.dtype, device=value.device)
    edge = BARRIER_EDGE * limit
    slack = limit - value
    log_part = torch.log(limit) - torch.log(slack.clamp_min(edge))
    tangent = -math.log(BARRIER_EDGE) + (edge - slack) / edge
    return (weight * torch.maximum(log_part, tangent)).mean()


def gap_rmse(model, q):
    """Predicted RMSE between actual and commanded q (aleatoric + epistemic)."""
    gap, var = model.gap(q)
    return torch.sqrt((gap.square() + EXPLOITATION_EXPLORATION_FACTOR * var).mean())


def penalties(model, robot, q, weight):
    """Gap RMSE, the summed barrier penalty, and the barrier terms it's made of."""
    qd = (q[2:] - q[:-2]) / (2 * DT)
    qdd = (q[2:] - 2 * q[1:-1] + q[:-2]) / DT ** 2
    jac = torch.as_tensor(robot.jacobians(q.detach().cpu().numpy())[1:-1], dtype=q.dtype)
    tool = torch.linalg.vector_norm((jac @ qd[..., None])[..., 0], dim=1)
    costs = {
        "joint_speed": barrier(qd.abs(), robot.v_joint, weight),
        "joint_acceleration": barrier(qdd.abs(), robot.a_joint, weight),
        "tool_speed": barrier(tool, robot.v_tcp, weight),
    }
    return gap_rmse(model, q), sum(costs.values()), costs


def loss(model, robot, q, cycle_time, cycle_time_per_gap_rmse, weight=BARRIER_WEIGHT_START):
    gap, penalty, costs = penalties(model, robot, q, weight)
    cycle_time = torch.as_tensor(cycle_time, dtype=q.dtype, device=q.device)
    total = cycle_time + cycle_time_per_gap_rmse * gap + penalty
    return total, gap, penalty, costs


def optimize(model, q_ref, dt_ref, robot, run=None, start_dt=DT):
    q0 = torch.as_tensor(np.asarray(q_ref, np.float32))
    moving = (q0[1:] - q0[:-1]).abs().amax(1) > PAUSE_TOL
    min_dt = MIN_DT * moving
    base_dt = torch.as_tensor(dt_ref, dtype=torch.float32)
    raw_dt = torch.nn.Parameter(inverse_softplus((base_dt - min_dt).clamp_min(1e-5)))
    opt = torch.optim.Adam([raw_dt], lr=LR)
    log = SummaryWriter(run) if run else None
    with torch.no_grad():
        baseline, _, _ = penalties(model, robot, resample(q0, base_dt), BARRIER_WEIGHT_START)
        cycle_time_per_gap_rmse = float((base_dt.sum() + start_dt) / baseline.clamp_min(1e-8))
    for step in range(STEPS + 1):
        # High early (stays well clear of every limit), decayed to 1/1000th by 80%
        # of the run so the tail is free to chase cycle time and gap alone. Never
        # exactly 0, or a true violation would go unpunished. No new hyperparameter:
        # reuses BARRIER_WEIGHT_START/END and STEPS.
        weight = BARRIER_WEIGHT_START + step / (0.8 * STEPS) * (BARRIER_WEIGHT_END - BARRIER_WEIGHT_START) if step < 0.8 * STEPS else BARRIER_WEIGHT_END
        dt = F.softplus(raw_dt) + min_dt
        path = resample(q0, dt)
        total_loss, gap, penalty, costs = loss(
            model, robot, path, dt.sum() + start_dt, cycle_time_per_gap_rmse, weight)
        task_loss = total_loss - penalty       # the true loss alone, without the barrier
        if log:
            values = {"loss/loss": task_loss, "loss/total_loss": total_loss, "loss/gap_rmse": gap,
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
            print(f"    step {step:4d}  loss {float(task_loss.detach()):6.3f}  total_loss "
                  f"{float(total_loss.detach()):6.3f}  gap {float(gap.detach()) * 1000:6.3f} mrad  "
                  f"T {float(dt.sum().detach()):5.3f}s  penalty {float(penalty.detach()):.4f}")
    if log:
        log.close()
    # The final step's state, not the best-seen one: total_loss bakes in the barrier,
    # which is a validity helper, not part of the real objective, so "best by total_loss"
    # would just reward whichever step's barrier weight happened to be smallest.
    return {"q": q0.numpy(), "dt": dt.detach().numpy(),
            "gap": float(gap.detach()), "penalty": float(penalty.detach())}


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
    result = optimize(model, q_ref, dt_ref, robot, run, start_dt)
    q, dt = executable(result["q"], result["dt"])
    # Resampled onto the fixed DT grid: a servoJ streamer only ticks at one rate,
    # and it's what the loss was scored on anyway.
    final = resample(torch.as_tensor(q, dtype=torch.float32),
                     torch.as_tensor(dt, dtype=torch.float32)).numpy()
    out = args.out or args.path.rsplit(".", 1)[0] + ".retime.path"
    write_path(out, final, DT)
    print(f"\n  gap RMSE {result['gap'] * 1000:.3f} mrad, penalty {result['penalty']:.4f}, cycle "
          f"{path_dt.sum():.3f} -> {start_dt + dt.sum():.3f} s\n  wrote {out}: {len(final)} commands")


if __name__ == "__main__":
    main()
