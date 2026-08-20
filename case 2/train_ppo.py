"""PPO policy that picks a real movej(vel, acc) for each recorded move, biased
to go as fast as the model says it can without the tracking error breaking down.

``optimize.py`` bends a whole *path* with gradient descent (a differentiable
bspline offset + per-slice time warp) -- a shape no real URScript line can
express directly. This instead picks the two numbers an actual ``movej(pose,
a=acc, v=vel)`` line takes, for one move at a time, by trial:

    obs     = [q0, q1, dist, gravity(q0), gravity(q1)]   one move (common.segments)
    action  = (vel, acc)              rad/s, rad/s^2 -- ``movej()`` clips these to
              the physical per-joint/Cartesian ceiling (``limits``) itself, so
              every action is executable, and the chosen numbers are literally
              what you'd write into the script's ``movej`` line.
    reward  = -(SPEED_WEIGHT * cycle_time / cycle_time(recording)
                + ERROR_PENALTY_WEIGHT * max(0, err / err(recording) - ERROR_CEIL)
                + LIMIT * penalty)

Framed as "go fast, subject to staying accurate" rather than a smooth trade:
cycle time is the only thing that costs anything until the predicted error
exceeds ``ERROR_CEIL`` (a fraction worse than the recorded move's own error),
past which it's punished hard. The intent is a policy that pushes speed right
up to where tracking starts to break down, not one that smoothly sells accuracy
for speed everywhere. ``err``/``penalty`` are ``optimize.py``'s own: the model's
predicted ``|actual_q - target_q|`` (widened by K std devs of its own
uncertainty) plus the soft overshoot penalty on the joint-speed/accel/tool-speed
ceilings (``Robot``) -- reused so the numbers read on the same scale as
``optimize.py``'s before/after.

``movej()``/``limits()`` recover what ``motion.py`` did before it was deleted in
the merge to the path-based architecture, generalized from the UR10e-only
original to any ``utils.Robot`` model.

One step per episode (a contextual bandit, not a multi-step trajectory): reset
picks a random move from the recordings, step scores the candidate and ends
immediately.

    python train_ppo.py --data data/ur5e --model models/distill-ur5e.pkl --robot UR5e --steps 20000
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from common import segments
from optimize import LIMIT, error, penalty
from train_distillation_model import DistillModel, load_recordings
from utils import DT, N_JOINTS, Robot

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Outer action bounds (rad/s, rad/s^2); ``movej()`` clips further to the tighter
# of the per-joint and Cartesian ceilings (``limits``, per robot model), so these
# only need to cover the fastest either UR10e or UR5e could ever be asked to go.
VEL_BOUNDS = (0.05, 3.3)
ACC_BOUNDS = (0.5, 70.0)

# The "go fast" objective: cycle time is the only continuous driver of cost;
# error only costs once it's ERROR_CEIL worse (relative) than the recorded
# move's own, at which point ERROR_PENALTY_WEIGHT punishes it hard -- a hinge,
# not a smooth trade, so the policy is pushed toward the fastest motion that
# still stays close to the recorded move's own accuracy, not toward selling
# accuracy for speed everywhere.
SPEED_WEIGHT = 1.0
ERROR_CEIL = 1.2
ERROR_PENALTY_WEIGHT = 10.0

# Gravity torque (utils.Robot.gravity, Nm) observed in the range of a few tens of
# Nm, versus q/dist in radians -- divide by this so it sits in a comparable range
# to the rest of the observation instead of dominating it by two orders of
# magnitude (no VecNormalize wrapper here, so scaling is on us).
GRAVITY_SCALE = 20.0

# ``penalty()`` (utils.Robot ceiling overshoot) is a rare, heavy-tailed term under
# random exploration. Clipped before scaling by LIMIT so an occasional bad action
# doesn't inject a reward outlier that dwarfs the rest of the cost and
# destabilizes PPO's advantage estimates -- see train_ppo's earlier calibration;
# ``movej()``'s own clipping to ``limits()`` makes big violations rarer here than
# under the old synthetic-shape action, but the safety net stays cheap to keep.
PEN_CLIP = 0.05

# Width of the box filter that rounds the trapezoid's corners: the acceleration
# ramps in over a fixed time, whatever its height, matching the real controller
# rather than snapping to a speed instantaneously.
SMOOTH = 0.060


def limits(q0, q1, robot: Robot):
    """Per-joint speed/accel ceiling for a move ``q0 -> q1``.

    The Cartesian tool-speed cap usually binds tighter than the joint one, so
    it's converted into joint terms via the tool speed per unit of path, taken
    at the move's worst pose. A joint that doesn't travel gets no Cartesian
    share, so it keeps its own ceiling. No Cartesian *acceleration* cap here --
    ``Robot`` only carries a tool*speed* ceiling, not a tool-acceleration one, so
    the acceleration ceiling is the joint one alone.
    """
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = q1 - q0
    cart = max(np.linalg.norm((robot.jacobian(q0 + u * d) @ d)[:3])
               for u in np.linspace(0.0, 1.0, 16))
    share = np.where(np.abs(d) > 1e-9, np.abs(d) / max(cart, 1e-9), np.inf)
    return np.minimum(robot.v_joint, share * robot.v_tcp), robot.a_joint


def movej(q0, q1, dt: float, robot: Robot, v: float = np.inf, a: float = np.inf) -> np.ndarray:
    """Commanded angles ``(n, N_JOINTS)`` for ``movej(q1, a, v)`` starting at ``q0``.

    A trapezoidal speed profile along the straight joint-space line, corners
    rounded by a ``SMOOTH``-second box filter -- what the real controller does:
    the move still takes the trapezoid's duration, just with rounded corners
    instead of an instant jump to a speed. The ceiling is the tightest of the
    requested ``v``/``a``, the per-joint limits, and the Cartesian one.
    """
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    v = v if v and v > 0 else np.inf
    a = a if a and a > 0 else np.inf
    d = q1 - q0
    if np.abs(d).max() < 1e-9:
        return q0[None].copy()
    reach = np.abs(d).clip(1e-12)
    v_lim, a_lim = limits(q0, q1, robot)
    sd = min(v / np.abs(d).max(), (v_lim / reach).min())
    sdd = min(a / np.abs(d).max(), (a_lim / reach).min())
    sd = min(sd, np.sqrt(sdd))                # too short to ever reach the ceiling
    T = 1.0 / sd + sd / sdd                    # trapezoid: cruise + one ramp
    t = np.arange(int(round(T / dt)) + 1) * dt
    speed = np.clip(np.minimum(sdd * t, sdd * (T - t)), 0.0, sd)
    nb = min(2 * round(SMOOTH / dt / 2) + 1, 2 * (len(t) // 4) + 1)
    speed = np.convolve(speed, np.ones(nb) / nb, mode="same")
    s = np.cumsum(speed)
    return q0 + (s / s[-1])[:, None] * d


class _Move:
    """One waypoint-to-waypoint move: its straight-line endpoints, the recorded
    trajectory (motion + settle window) to score a baseline against, and the
    gravity torque at each end (utils.Robot.gravity, precomputed once here since
    it only depends on the fixed geometry, not the candidate speed profile)."""

    __slots__ = ("q0", "q1", "baseline_q", "n_motion", "n_settle", "g0", "g1")

    def __init__(self, rec, s, robot: Robot):
        self.q0 = rec.target_q[s.i0]
        self.q1 = rec.target_q[max(s.i0, s.i1 - 1)]
        self.baseline_q = rec.target_q[s.i0:s.i2]
        self.n_motion = s.i1 - s.i0
        self.n_settle = s.i2 - s.i1
        self.g0 = robot.gravity(self.q0)
        self.g1 = robot.gravity(self.q1)


def build_moves(recordings, robot: Robot) -> list:
    moves = []
    for rec in recordings:
        for s in segments(rec):
            if s.i1 - s.i0 >= 3:          # too few rows: not a real move
                moves.append(_Move(rec, s, robot))
    return moves


# --- free functions behind MoveEnv's obs/action/candidate machinery ----------
# Pulled out of the class so a script that only wants to *replay* a trained
# agent's choices (export_ppo_path.py) can reuse the exact same mapping without
# constructing a full MoveEnv (whose __init__ does an O(n_moves) baseline
# precompute it would not need).

def obs_of(m: _Move) -> np.ndarray:
    dist = float(np.abs(m.q1 - m.q0).max())
    return np.concatenate([m.q0, m.q1, [dist],
                           m.g0 / GRAVITY_SCALE, m.g1 / GRAVITY_SCALE]).astype(np.float32)


def unmap_action(action):
    """Map a normalized action in [-1,1]^2 to (vel, acc) in rad/s, rad/s^2."""
    a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    vel = VEL_BOUNDS[0] + (a[0] + 1.0) / 2.0 * (VEL_BOUNDS[1] - VEL_BOUNDS[0])
    acc = ACC_BOUNDS[0] + (a[1] + 1.0) / 2.0 * (ACC_BOUNDS[1] - ACC_BOUNDS[0])
    return float(vel), float(acc)


def candidate_q(m: _Move, robot: Robot, vel: float, acc: float) -> np.ndarray:
    q = movej(m.q0, m.q1, DT, robot, vel, acc)
    if m.n_settle:
        q = np.vstack([q, np.tile(m.q1, (m.n_settle, 1))])
    return q


class MoveEnv(gym.Env):
    """One step per episode: pick a speed profile for a random recorded move."""

    metadata = {"render_modes": []}

    def __init__(self, model: DistillModel, robot: Robot, moves: list,
                 speed_weight: float = SPEED_WEIGHT, error_ceil: float = ERROR_CEIL,
                 error_penalty_weight: float = ERROR_PENALTY_WEIGHT, limit_w: float = LIMIT):
        super().__init__()
        if not moves:
            raise ValueError("no usable moves in the given recordings")
        self.model, self.robot = model, robot
        self.speed_weight, self.error_ceil = speed_weight, error_ceil
        self.error_penalty_weight, self.limit_w = error_penalty_weight, limit_w
        self.moves = moves
        # [q0, q1, dist, gravity(q0)/GRAVITY_SCALE, gravity(q1)/GRAVITY_SCALE]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4 * N_JOINTS + 1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._baseline = [self._score_q(m.baseline_q) for m in moves]
        self._i = 0

    def _obs(self, m: _Move) -> np.ndarray:
        return obs_of(m)

    def _score_q(self, q) -> tuple[float, float, float]:
        """(err, cycle_time, penalty) of a candidate trajectory ``q``, ``penalty``
        clipped to ``PEN_CLIP`` -- see its module docstring for why."""
        qt = torch.as_tensor(np.asarray(q, np.float32))
        err = float(error(self.model, qt).mean())
        pen = min(float(penalty(qt, self.robot)), PEN_CLIP)
        return err, len(q) * DT, pen

    def _unmap(self, action):
        return unmap_action(action)

    def _candidate(self, m: _Move, vel: float, acc: float) -> np.ndarray:
        return candidate_q(m, self.robot, vel, acc)

    def _cost(self, err, cycle, pen, before_err, before_T) -> float:
        over = max(0.0, err / before_err - self.error_ceil)
        return (self.speed_weight * cycle / before_T
                + self.error_penalty_weight * over + self.limit_w * pen)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._i = int(self.np_random.integers(len(self.moves)))
        return self._obs(self.moves[self._i]), {}

    def step(self, action):
        m = self.moves[self._i]
        before_err, before_T, _ = self._baseline[self._i]
        vel, acc = self._unmap(action)
        q = self._candidate(m, vel, acc)
        err, cycle, pen = self._score_q(q)
        cost = self._cost(err, cycle, pen, before_err, before_T)
        info = {"err": err, "cycle_time": cycle, "penalty": pen,
               "before_err": before_err, "before_cycle": before_T}
        return self._obs(m), -cost, True, False, info


class _TrainCallback(BaseCallback):
    """Records per-episode reward, err, and cycle_time during PPO training."""

    def __init__(self):
        super().__init__(verbose=0)
        self.records: list[dict] = []

    def _on_step(self) -> bool:
        for done, reward, info in zip(self.locals.get("dones", []),
                                      self.locals.get("rewards", []),
                                      self.locals.get("infos", [])):
            if done:
                self.records.append({
                    "timestep":   int(self.num_timesteps),
                    "reward":     float(reward),
                    "err":        float(info.get("err", float("nan"))),
                    "cycle_time": float(info.get("cycle_time", float("nan"))),
                })
        return True


def _rolling_mean(x: list, w: int) -> list:
    return [float(np.mean(x[max(0, i - w): i + 1])) for i in range(len(x))]


def log_training_run(data: str, steps: int, callback: _TrainCallback,
                     results_dir: str, dt_str: str, seed: int = None):
    """Save ``log.json`` and ``training_curve.png`` under ``results/<dt_str>_ppo/``."""
    run_dir = os.path.join(results_dir, f"{dt_str}_ppo")
    os.makedirs(run_dir, exist_ok=True)

    records = callback.records
    if not records:
        print("[results] no training records -- skipping")
        return run_dir

    timesteps   = [r["timestep"]   for r in records]
    rewards     = [r["reward"]     for r in records]
    errs        = [r["err"]        for r in records]
    cycle_times = [r["cycle_time"] for r in records]

    tail = max(1, len(errs) // 10)
    log = {
        "datetime": dt_str, "type": "train_ppo", "data": data,
        "steps": steps, "seed": seed,
        "summary": {
            "n_episodes":      len(records),
            "best_err":        float(np.nanmin(errs)),
            "final_err_mean":  float(np.nanmean(errs[-tail:])),
            "best_cycle_time": float(np.nanmin(cycle_times)),
        },
    }
    with open(os.path.join(run_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=2)

    w = max(1, len(errs) // 20)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(timesteps, errs, alpha=0.15, linewidth=0.6)
    axes[0].plot(timesteps, _rolling_mean(errs, w), linewidth=1.6, label=f"rolling mean (w={w})")
    axes[0].set_ylabel("Error (rad -- lower = better)")
    axes[0].set_title(f"PPO training curve | {steps} steps | {len(records)} episodes")
    axes[0].legend(fontsize=8)

    axes[1].plot(timesteps, cycle_times, alpha=0.15, linewidth=0.6)
    axes[1].plot(timesteps, _rolling_mean(cycle_times, w), linewidth=1.6, label=f"rolling mean (w={w})")
    axes[1].set_ylabel("Cycle time (s -- lower = faster)")
    axes[1].legend(fontsize=8)

    axes[2].plot(timesteps, rewards, alpha=0.15, linewidth=0.6)
    axes[2].plot(timesteps, _rolling_mean(rewards, w), linewidth=1.6, label=f"rolling mean (w={w})")
    axes[2].set_ylabel("Reward (-cost -- higher = better)")
    axes[2].set_xlabel("Timestep")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    plot_path = os.path.join(run_dir, "training_curve.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[results] log   -> {run_dir}/log.json")
    print(f"[results] plot  -> {plot_path}")
    return run_dir


def evaluate(agent: PPO, env: MoveEnv):
    """(before_err, err, before_T, T, cost) per move, deterministic policy."""
    rows = []
    for i, m in enumerate(env.moves):
        obs = env._obs(m)
        action, _ = agent.predict(obs, deterministic=True)
        before_err, before_T, _ = env._baseline[i]
        vel, acc = env._unmap(action)
        q = env._candidate(m, vel, acc)
        err, cycle, pen = env._score_q(q)
        cost = env._cost(err, cycle, pen, before_err, before_T)
        rows.append((before_err, err, before_T, cycle, cost))
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Train a PPO policy to re-time recorded moves against the model.")
    ap.add_argument("--data", default="data/ur5e", help="folder of recordings")
    ap.add_argument("--model", default="models/distill-ur5e.pkl", help="distilled model pickle")
    ap.add_argument("--robot", default="UR5e", choices=list(Robot.MODELS),
                    help="which arm's kinematics/limits the penalty term holds to")
    ap.add_argument("--steps", type=int, default=20000, help="PPO timesteps")
    ap.add_argument("--seed", type=int, default=0, help="PPO/env RNG seed")
    ap.add_argument("--out", default="models/agent_ppo.zip", help="trained agent save path")
    args = ap.parse_args()

    model = DistillModel.load(args.model)
    for p in model.parameters():
        p.requires_grad_(False)          # only the policy is trained
    robot = Robot(args.robot)
    recordings = load_recordings(args.data)
    moves = build_moves(recordings, robot)
    print(f"{len(recordings)} recordings, {len(moves)} moves from {args.data}")

    env = MoveEnv(model, robot, moves)
    # A 25-dim obs / 2-dim action MlpPolicy has nothing for a GPU to chew on --
    # kernel-launch latency on tensors this small makes "cuda" slower than "cpu".
    agent = PPO("MlpPolicy", env, verbose=0, seed=args.seed, device="cpu")
    cb = _TrainCallback()
    agent.learn(total_timesteps=args.steps, callback=cb)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    agent.save(args.out)
    print(f"trained PPO for {args.steps} steps -> {args.out}")

    dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_training_run(args.data, args.steps, cb, RESULTS_DIR, dt_str, args.seed)

    rows = evaluate(agent, env)
    eb = ea = tb = ta = 0.0
    for before_err, err, before_T, T, rel in rows:
        eb, ea, tb, ta = eb + before_err, ea + err, tb + before_T, ta + T
    n = len(rows)
    print(f"\n  total  error {eb / n * 1000:.3f} -> {ea / n * 1000:.3f} mrad   "
         f"cycle {tb:.3f} -> {ta:.3f} s   over {n} moves")


if __name__ == "__main__":
    main()
