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

    python optimize.py --path scripts/triangle.path --model models/distill-ur5e.pkl \\
        --robot UR5e
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
from utils import DT, N_JOINTS, Robot

ALPHA = 2.0        # a second of cycle time costs as much as this much error, relatively
                   # Calibrated on the UR5e recordings, where the measured gap is
                   # 0.04-0.08 mrad mean and 0.84 mrad at worst -- tens of arcseconds,
                   # near the arm's own repeatability. At ALPHA=1 a percent of that is
                   # priced like a percent of cycle time, which nobody running the
                   # robot would agree to. At 2 the optimizer buys about half the cycle
                   # for a gap of 0.09 -> 0.12 mrad. Raise it further only with the
                   # model's uncertainty in view: 43% of the error term is already the
                   # model saying it does not know.
K = 1.0            # standard deviations of the model's own uncertainty added to the error
LIMIT = 30.0       # how hard the ceilings are held. Both lanes share this, and they
                   # want different things from it: a black-box search is indifferent
                   # (CEM returns the same solution, always feasible, anywhere from 30
                   # to 1000) while gradient descent degrades monotonically and finds
                   # nothing at all above ~100 -- `penalty`'s barrier has slope ~LIMIT
                   # against a useful gradient of order 1. 30 is where both still work
                   # and a one-row 10% excursion still costs 0.55, some 50x what the
                   # older mean-only penalty charged for it.
KNOTS, POINTS, STEPS, LR = 24, 16, 600, 0.01

# The reduced basis the RL lane acts in: one time scale per ``movable`` block plus
# a coarser offset spline. B_MAX is a fixed action width (the longest path here has
# 9 blocks); a path with fewer reads only its first n_blocks and the rest are dead.
# B_MAX is a fixed action width, so it also fixes the largest path the policy can
# describe: a path with more blocks than this cannot be decoded at all. 16 covers the
# 13 that a doubled cycle reaches, with headroom -- 12 did not, and a policy trained
# only on 2-7 block paths emitted garbage on the 8- and 9-block ones it met later.
B_MAX, RL_POINTS = 16, 8
T_GAIN = 2.5       # a block may be cut to exp(-T_GAIN) = 8% of its recorded duration
T_SHIFT = 1.0      # ...but only stretched to ~1.5x. See ``time_scale``.
A_GAIN = 0.02      # offset ceiling, as a fraction of that joint's travel


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


def movable(q_ref, dt: float = DT):
    """1 where the reference is moving, 0 where it stands still, per row.

    The offset is multiplied by this, so the waypoints the script holds are kept
    exactly while the moves between them are free. A structural constraint rather
    than another penalty term: nothing to weigh against the rest of the loss.
    """
    return torch.as_tensor(
        (np.abs(np.gradient(q_ref, dt, axis=0)).max(1) > 0.01).astype(np.float32))


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


def bounds(q_ref, dt: float = DT) -> np.ndarray:
    """Block boundary *times* of a path: ``(n_blocks + 1,)``, from ``movable``.

    A recorded path alternates moves and the pauses the script sleeps through, and
    ``movable`` already marks exactly that. Those flips are the boundaries worth
    retiming against: on the recordings here they give 4 to 9 blocks, each of which
    is wholly a move or wholly a pause, so a single number per block can delete a
    pause without touching the move beside it. ``phase``'s ``KNOTS`` uniform slices
    cannot -- a pause spans anywhere from 0.1 to 4.8 of them.

    The last boundary is pinned to ``(N - 1) * dt``, the path's own duration, so the
    block durations sum to exactly ``T0`` and a neutral action reproduces the
    reference row for row rather than one row long.
    """
    free = movable(q_ref, dt).numpy()
    cut = [0] + [i for i in range(1, len(free)) if free[i] != free[i - 1]] + [len(free)]
    t = np.asarray(cut, dtype=float) * dt
    t[-1] = (len(free) - 1) * dt
    return t


def time_scale(a):
    """How much longer a block takes, for an action coordinate in [-1, 1].

    Asymmetric, because the task is: reclaiming a pause needs to cut a block to a
    tenth of its length, while nothing is ever improved by running one ten times
    slower. A symmetric map prices those the same and makes exploration ruinous --
    ``a = +1`` under one cost ``ALPHA * 11``, so sampled actions were dominated by
    catastrophic slowdowns and a policy could not learn past them.

    ``tanh`` shifted by ``T_SHIFT`` saturates quickly upward and slowly downward:
    ``a = -1`` gives ``exp(-T_GAIN)``, about 8% of the recorded duration, while
    ``a = +1`` gives only ~1.5x. It stays smooth everywhere and is exactly 1 at
    ``a = 0``, which is what keeps the neutral action the recorded path.
    """
    b = T_SHIFT
    span = float(np.tanh(b) - np.tanh(b - 2.0))          # the value at a = -1
    return torch.exp(T_GAIN * (torch.tanh(2.0 * a + b) - float(np.tanh(b))) / span)


def phase_blocks(a_time, bnd, dt: float = DT):
    """``(t, u, T)`` like ``phase``, but each ``movable`` block gets its own scale.

    Block ``b`` takes ``exp(T_GAIN * tanh(2 * a_b))`` times its recorded duration.
    The ``tanh`` is what keeps the useful range in the *interior* of the action box:
    deleting a 0.8 s pause wants a multiplier near 0.02, which on a raw log scale
    sits on the wall, and SB3's PPO clips at the wall without correcting the
    log-prob, so the Gaussian's mean drifts out and ``log_std`` collapses. Here
    ``a = -0.5`` already buys 89% of the achievable cut and ``a = -0.75`` 96%, so
    the mean has no reason to leave the box.

    ``a_time = 0`` replays the reference exactly.
    """
    # In float64: the phase is a ratio of accumulated times, and in float32 the
    # round-off reaches ~1e-6 rad on the longest paths -- small, but it would keep
    # the neutral action from scoring exactly the recorded path, which is the one
    # identity everything downstream is calibrated on. Autograd is unaffected.
    bnd = torch.as_tensor(np.asarray(bnd, np.float64))
    dur = bnd[1:] - bnd[:-1]
    mult = time_scale(a_time[:len(dur)].double())
    tk = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(dur * mult, 0)])
    # round, not truncate: at the neutral action tk[-1] is T0 to floating-point, and
    # int() would drop the last row half the time.
    t = torch.arange(int(round(float(tk[-1]) / dt)) + 1, dtype=torch.float64) * dt
    i = torch.searchsorted(tk.detach().contiguous(), t.contiguous()).clamp(1, len(dur))
    frac = (t - tk[i - 1]) / (tk[i] - tk[i - 1]).clamp_min(1e-12)
    u = ((bnd[i - 1] + frac * dur[i - 1]) / bnd[-1]).clamp(0.0, 1.0)
    return t.float(), u.float(), tk[-1].float()


def decode(action, ref, bnd, travel, free, dt: float = DT):
    """``(q, T)`` for one action: the candidate path the agent proposes.

    The action is ``[time scales (B_MAX), offset control points (n * N_JOINTS)]``,
    every component in ``[-1, 1]``. Offsets are a fraction of how far that joint
    travels along the path, so one scale fits every trajectory; they run through the
    same ``bspline``/``movable`` machinery the gradient lane uses, which is what
    makes the two comparable.

    ``action = 0`` is the recorded path, exactly -- so a fresh policy starts
    feasible, at the baseline it has to beat, rather than behind a penalty cliff.
    """
    a = torch.as_tensor(np.asarray(action, np.float32))
    t, u, T = phase_blocks(a[:B_MAX], bnd, dt)
    q = sample(ref, u)
    n_points = (len(a) - B_MAX) // N_JOINTS
    if n_points:
        scale = A_GAIN * torch.as_tensor(np.asarray(travel, np.float32))
        off = scale * torch.tanh(2.0 * a[B_MAX:].view(n_points, N_JOINTS))
        q = q + sample(free, u) * bspline(u, zeroed(off))
    return q, T


def predict_gap(model, q):
    """``(gap, sd)`` per row and joint (rad): what the model says the robot will do.

    The gap is *signed*, which ``error`` throws away -- a search that wants to score
    the ring after a stop (how far it overshoots, how long it takes to settle, how
    many times it crosses back) needs the sign, and none of those are differentiable,
    so only the RL lane can use them.
    """
    mx, sx, my, sy = (torch.as_tensor(s, dtype=torch.float32) for s in model.stats)
    x = ((features(q, DT, model.pad) - mx) / sx).clamp(-X_CLIP, X_CLIP)
    mu, lv = model.forward(x.T[None])
    gap = mu.mean(0)[0].T * sy + my
    var = (torch.exp(lv).mean(0) + mu.var(0, unbiased=False))[0].T * sy ** 2
    return gap, var.sqrt()


def error(model, q, k: float = None):
    """Per-row ``|actual_q - q|`` (rad), widened by ``k`` sd of the model's own
    uncertainty so that trajectories it has never seen are not free.

    ``k`` defaults to ``K``. The RL lane raises it: a policy evaluates the model
    hundreds of thousands of times and will find whatever it is confidently wrong
    about, which 600 gradient steps never get the chance to.
    """
    gap, sd = predict_gap(model, q)
    return gap.abs() + (K if k is None else k) * sd


BAND_FRAC, BAND_FLOOR = 0.2, 2e-4     # settled means "inside a fifth of the baseline ring"


def ring(gap, still, dt: float, band: float) -> tuple[float, float]:
    """``(peak, settle)`` of the ring-down: how far it overshoots, how long for.

    Scored only where the path stands still, which is where a gap is unambiguously
    the arm still moving after the command stopped. ``peak`` is the worst joint's
    worst row there; ``settle`` is the time from each standstill's start until the
    gap stays inside ``band``, summed over them.

    Neither is differentiable -- one is a max over a data-dependent window, the other
    a threshold crossing -- which is exactly why they are here and not in ``loss``.
    """
    still = np.asarray(still, bool)
    worst = np.abs(np.asarray(gap, float)).max(1)
    idx = np.flatnonzero(still)
    if not len(idx):
        return 0.0, 0.0
    peak = settle = 0.0
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        w = worst[run]
        peak = max(peak, float(w.max()))
        over = np.flatnonzero(w > band)
        settle += float(over[-1] + 1) * dt if len(over) else 0.0
    return peak, settle


def penalty(q, robot: Robot):
    """Overshoot of the speed, acceleration and tool-speed ceilings, as a fraction.

    0 for a path the controller can run as written. Explicit rather than left to the
    model: outside the envelope the model extrapolates, and a confident wrong answer
    there costs nothing. The Jacobian is taken at the current poses, refreshed each
    step from the detached trajectory, so the tool speed stays right as the path
    moves; only the speed carries gradient.

    Each ceiling contributes its mean overshoot and a fourth-power norm of it. The
    mean says how much of the path is illegal but dilutes a brief excursion into
    noise -- a path running a tool-speed cap over for a tenth of a second scored 9e-5,
    which at any sane weight is worth buying. The norm answers how far outside it ever
    went, which is what decides whether the controller will run it.

    A plain ``amax`` would say that more directly and is wrong here: it carries
    gradient on exactly one row, so the gradient lane gets a large-magnitude spike
    pointing at a single sample and stops moving at all (measured: it found nothing on
    every path, at any weight). The fourth power keeps the emphasis on the worst rows
    while spreading gradient over all of them, and for an excursion confined to one
    row of a thousand it is still ~150x the mean.

    The ceilings are the controller's *spec* values, not the margined ones. That
    margin exists so a differenced recording is not misread as a violation; handing
    it to an optimizer just converts tolerance into 2% more speed, which the
    controller then clamps. Safe to hold to spec here because the recorded paths are
    nowhere near it -- they peak at 0.31 of the joint speed ceiling, 0.08 of the
    acceleration one and 0.82 of the tool-speed one -- so they still score 0.
    """
    qd = (q[2:] - q[:-2]) / (2 * DT)
    qdd = (q[2:] - 2 * q[1:-1] + q[:-2]) / DT ** 2
    jac = torch.as_tensor(robot.jacobians(q.detach().numpy())[1:-1], dtype=torch.float32)
    tool = torch.linalg.norm((jac @ qd[..., None])[..., 0], dim=1)

    eps = 1e-24

    def over(z, lim):
        e = torch.relu(z.abs() / torch.as_tensor(lim, dtype=torch.float32) - 1)
        # The epsilon keeps the fourth root differentiable at zero, where a feasible
        # path sits and where the bare root's gradient is undefined; subtracting it
        # back off keeps a feasible path scoring *exactly* zero, which the reward's
        # baseline depends on. The gradient survives both: the chain rule carries a
        # factor of e^3, which is what actually vanishes at zero.
        return e.mean() + (e.pow(4).mean() + eps).pow(0.25) - eps ** 0.25

    return (over(qd, robot.v_joint_spec) + over(qdd, robot.a_joint_spec)
            + over(tool, robot.v_tcp_spec))


def loss(model, robot, q, T, T0, before):
    """``(loss, error, limits)`` for a candidate path, everything relative to the
    recorded one, which therefore scores ``1 + ALPHA``."""
    err = error(model, q).mean()
    pen = penalty(q, robot)
    return err / before + ALPHA * T / T0 + LIMIT * pen, err, pen


def optimize(model, q_ref, robot: Robot, run: str = None):
    """Optimize a whole path. Returns the best ``(q, T, error, offset)`` seen and
    the recorded path's own error, which is what they are measured against."""
    ref = torch.as_tensor(np.asarray(q_ref, np.float32))
    free = movable(q_ref)[:, None]
    T0 = (len(ref) - 1) * DT
    with torch.no_grad():
        before = float(error(model, ref).mean())

    offset = torch.zeros(POINTS, N_JOINTS, requires_grad=True)
    theta = torch.zeros(KNOTS, requires_grad=True)
    opt = torch.optim.Adam([offset, theta], lr=LR)
    log = SummaryWriter(run) if run else None
    best = None
    for i in range(STEPS + 1):
        t, u, T = phase(theta, T0, DT)
        q = sample(ref, u) + sample(free, u) * bspline(u, zeroed(offset))
        total, err, pen = loss(model, robot, q, T, T0, before)
        if best is None or float(total.detach()) < best[0]:
            best = (float(total.detach()), q.detach().numpy(), float(T.detach()),
                    float(err.detach()), float((q - sample(ref, u)).abs().max().detach()))
        if log:
            for key, v in (("loss/total", total - LIMIT * pen), ("loss/error", err),
                           ("loss/time", T), ("loss/limits", pen)):
                log.add_scalar(key, float(v.detach()), i)
        if i < STEPS:
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_([offset, theta], 1.0)   # keep Adam settled
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
    ap.add_argument("--path", required=True, help="path CSV from convert.py")
    ap.add_argument("--model", required=True, help="distilled model pickle")
    ap.add_argument("--robot", required=True, choices=list(Robot.MODELS),
                    help="which arm's kinematics and limits to hold the path to")
    ap.add_argument("--out", default=None, help="default: <path>.optimized.path")
    args = ap.parse_args()

    robot = Robot(args.robot)
    model = DistillModel.load(args.model)
    for p in model.parameters():
        p.requires_grad_(False)              # only the trajectory is optimized
    rows = np.array(load_path(args.path), float)
    q_ref, T0 = rows[:, :N_JOINTS], (len(rows) - 1) * DT
    tensorboard_path = f"runs/optimize/{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"  {len(rows)} setpoints, {T0:.2f} s at {DT * 1000:.2f} ms; "
          f"logging to {tensorboard_path}")

    q, T, after, off, before = optimize(model, q_ref, robot, tensorboard_path)
    score, base = after / before + ALPHA * T / T0, 1 + ALPHA
    v = robot.tcp_speed(q).max()
    print(f"\n  error {before * 1000:.3f} -> {after * 1000:.3f} mrad   cycle {T0:.3f} -> {T:.3f} s"
          f"\n  score {score:.3f} vs {base:.3f} for the recorded path: "
          f"{'better' if score < base else 'NOT an improvement'}"
          f"\n  up to {off * 1000:.1f} mrad off it, peak tool speed {v:.3f} m/s"
          f"{'  ** over the cap' if v > robot.v_tcp else ''}")

    out = args.out or args.path.rsplit(".", 1)[0] + ".optimized.path"
    write_path(out, q, DT)
    print(f"wrote {out} ({len(q)} setpoints)\n"
          f"  run it: python send.py --path {out} --loop 5 --out optimized.csv")


if __name__ == "__main__":
    main()
