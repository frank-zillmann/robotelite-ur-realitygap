"""Build the RL training data and train an agent against the models.

Pipeline (one command):

    1. run the scripts on URSim -> clean commanded target trajectory
    2. augment overwrites actual_* columns with the distill model's predictions
    3. metrics.add_score adds the ``score`` column (the RL label)
    4. train the agent, saving models/agent_<mode>.zip

Stages 1-3 write ``sim_to_real.csv`` in the repo root. To score a candidate the
agent proposes, ``dynamics`` builds its commanded trajectory (same joint geometry,
a new speed profile), the distill model predicts the actual_* channels, and the
metric scores them. One step per episode (a contextual bandit): reset returns a
move, step scores the candidate.

    observation : the move + its baseline (built in `observe`)
    action      : params -> [vel, acc] (deg/s); path -> [accel_frac, decel_frac, speed]
    reward      : -OBJECTIVE(score, cycle_time)

Modes (``--mode``, saved as ``models/agent_<mode>.zip``):
  params  agent picks one vel/acc per move (movej). Default.
  path    agent re-times the move into a servoj setpoint stream.

    python train_rla.py --model models/distill.pkl \
        --scripts scripts/vertical_swing.script scripts/horizontal_swing.script \
        scripts/triangle.script --robot-ip 127.0.0.1 --loop 5 --mode params --steps 20000

Requires `gymnasium` and `stable-baselines3` (see requirements.txt).
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
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from analysis import Recording
from common import segments
from dynamics import (DEG2RAD, GRID, MAX_JOINT_ACC, MAX_JOINT_SPEED, Dynamics,
                      default_dynamics, trapezoidal)
from train_distillation_model import DistillModel, augment
from metrics import CurrentGapMetric, EvaluationMetric, SCORE_COL, add_score
from preprocess import Identity, Preprocess, default_preprocess
from utils import ACC_COL, N_JOINTS, SCRIPT_COL, VEL_COL, get_block, set_block

# Training-data file (sim targets + predicted actuals + score label).
SIM_TO_REAL = "sim_to_real.csv"

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Action bounds in the URScript units (deg/s, deg/s^2). A movej speed above the
# joint limit (~180 deg/s = pi rad/s) clamps, so the useful range stays below it.
VEL_BOUNDS = (20.0, 180.0)
ACC_BOUNDS = (40.0, 600.0)

# --- objective the optimizer minimizes ---------------------------------------
# Cost of one move, weighting the metric ``score`` against ``cycle_time`` (move
# duration, s). The env uses reward = -cost; run.py scores with the same lambda.
SCORE_WEIGHT = 1.0
CYCLE_WEIGHT = 1.0
OBJECTIVE = lambda score, cycle_time: SCORE_WEIGHT * score + CYCLE_WEIGHT * cycle_time

# Path-mode cost. ``max_score`` is the worst per-row score over the path.
PATH_SCORE_WEIGHT = 1.0
PATH_CYCLE_WEIGHT = 1.0
PATH_OBJECTIVE = lambda max_score, cycle_time: \
    PATH_SCORE_WEIGHT * max_score + PATH_CYCLE_WEIGHT * cycle_time

# Path action: a trapezoidal speed profile (accel_frac, decel_frac) + a per-row
# servoj time from SERVO_DT_BOUNDS (smaller = faster). Scored on PATH_ROWS samples.
PATH_ROWS = 50
SERVO_DT_BOUNDS = (0.004, 0.02)


def move_line(rec: Recording, move):
    """Straight joint-space line of a movej: ``(start, travel, distance)``.

    A movej interpolates every joint from ``start`` to ``dest`` together; a
    candidate keeps this line and only sets the speed profile along it. ``start``
    and ``travel`` are ``(N_JOINTS,)`` in rad; ``distance`` is the widest joint
    travel (rad).
    """
    start = rec.target_q[move.i0]
    travel = rec.target_q[move.i1] - start
    return start, travel, float(np.abs(travel).max())


def _aggregate(x) -> np.ndarray:
    """RMS of ``x`` over the move window (axis 0). 2D -> (N_JOINTS,), 1D -> scalar."""
    return np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2, axis=0))


def observe(move, table, model: DistillModel) -> np.ndarray:
    """Observation vector for one move.

    ``[start, dest, dist, joint one-hot(6)]``, then the baseline of every
    ``model.predicts()`` channel (six joints each) and the baseline ``score``, all
    aggregated over the move window. ``table`` is the recording in the agent's
    space (``pre.transform_rla`` already applied).
    """
    df = table.iloc[move.i0:move.i2]
    onehot = [1.0 if i == move.joint else 0.0 for i in range(N_JOINTS)]
    feats = [move.start, move.dest, move.dist, *onehot]
    for base in model.predicts():
        feats += list(_aggregate(get_block(df, base)))
    feats.append(float(_aggregate(df[SCORE_COL].to_numpy())))
    return np.array(feats, dtype=np.float32)


def speed_profile(accel_frac, decel_frac, rows=PATH_ROWS) -> np.ndarray:
    """Monotonic progress ``s(t)`` in [0,1] from a trapezoidal speed profile.

    Speed ramps up over the first ``accel_frac`` of the time, cruises, then ramps
    down over the last ``decel_frac``: one accel-cruise-decel, no wobble. Progress
    is the normalized integral of that speed. Larger ``decel_frac`` = a slower
    approach into the stop (where the ring is). Both in [0, ~0.9], clipped so the
    ramps do not overlap.
    """
    a = float(np.clip(accel_frac, 0.0, 0.9))
    d = float(np.clip(decel_frac, 0.0, 0.9 - a))
    t = np.linspace(0.0, 1.0, rows)
    v = np.ones(rows)
    if a > 0:
        v = np.where(t < a, t / a, v)
    if d > 0:
        v = np.where(t > 1 - d, (1 - t) / d, v)
    s = np.cumsum(v)
    return (s - s[0]) / (s[-1] - s[0])


def evaluate(model: DistillModel, metric: EvaluationMetric, frame,
             pre: Preprocess) -> np.ndarray:
    """Per-row score for a commanded-trajectory ``frame``.

    Preprocess the frame, overwrite the actual_* channels with the model's
    predictions, revert, then read the metric per row. ``pre`` wraps the model as
    ``augment`` does, so env reward, run.py, and the saved dataset agree.
    """
    frame = pre.transform_distill(frame)
    preds = model.predict(frame)
    for base in model.predicts():
        set_block(frame, base, preds[base])
    frame = pre.revert_distill(frame)
    return metric.per_row(frame)


def collect_moves(scripts, robot_ip, loop=None, out=SIM_TO_REAL, settle_s=2.0):
    """Run each script on URSim and pool the target segments (waypoint to waypoint).

    Every row is tagged with its script (``SCRIPT_COL``) before the runs are
    concatenated into ``out``, so segmentation never spans two scripts. A
    ``settle_s`` pause between runs lets the robot come to rest before the next
    recording. ``send`` is imported locally so the socket code loads only on use.
    """
    import time
    import send

    frames, n_seg = [], 0
    for k, s in enumerate(scripts):
        send.record_run(robot_ip, s, out, loop=loop)
        raw = pd.read_csv(out)                                # tag rows with the script;
        frame = pd.concat([raw, pd.Series(s, index=raw.index, name=SCRIPT_COL)],
                          axis=1)                              # concat avoids fragmentation
        n_seg += len(segments(Recording(out, df=frame)))
        frames.append(frame)
        print(f"  {s}: {n_seg} segments pooled so far")
        if k + 1 < len(scripts):
            time.sleep(settle_s)
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    print(f"wrote combined recording -> {out} ({n_seg} segments)")


class _MoveEnv(gym.Env):
    """Shared base: score candidate motions of a recording's moves.

    Holds the models, the moves (``segments``), and a per-move dynamics grid.
    Subclasses set the action space and implement ``_cost(move, *action)``.
    """

    metadata = {"render_modes": []}

    def __init__(self, model: DistillModel, metric: EvaluationMetric, rec: Recording,
                 dyn: Dynamics = None, pre: Preprocess = None):
        super().__init__()
        self.model = model
        self.metric = metric
        self.rec = rec
        self.pre = pre or Identity()
        self.dyn = dyn or default_dynamics(rec)
        self.adf = self.pre.transform_rla(rec.df)   # recording in the agent's space
        self.targets = segments(rec)
        if not self.targets:
            raise ValueError("no target moves: the scripts recorded no joint motion")
        self._geo = {}                               # per-move geometry q(u), u uniform
        for m in self.targets:
            g = self._geometry(m)
            self._geo[m.i0] = g
            self.dyn.prepare(m.i0, g)                 # precompute pose terms along it
        dim = observe(self.targets[0], self.adf, model).shape[0]
        self.observation_space = spaces.Box(
            low=np.full(dim, -np.inf, dtype=np.float32),
            high=np.full(dim, np.inf, dtype=np.float32))
        self._i = 0

    def _obs(self, move) -> np.ndarray:
        return observe(move, self.adf, self.model)

    def _q_at(self, move, s) -> np.ndarray:
        """Joint angles along the move's geometry at progress ``s`` in [0,1]."""
        g = self._geo[move.i0]
        u = np.linspace(0.0, 1.0, len(g))
        return np.column_stack([np.interp(s, u, g[:, j]) for j in range(N_JOINTS)])

    def _candidate(self, move, s, dt, vel_deg, acc_deg):
        """Per-row score of the candidate whose progress is ``s(t)`` at step ``dt``.

        Places the move's geometry at the given progress, asks ``dynamics`` for the
        commanded frame, and scores it through the model and metric.
        """
        q = self._q_at(move, s)
        frame = self.dyn.frame(q, dt, vel_deg, acc_deg, s=s, key=move.i0)
        return evaluate(self.model, self.metric, frame, self.pre)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._i = int(self.np_random.integers(len(self.targets)))
        return self._obs(self.targets[self._i]), {}

    def step(self, action):
        move = self.targets[self._i]
        score, cycle, cost = self._cost(move, action)
        info = {"score": score, "cycle_time": cycle}
        return self._obs(move), -cost, True, False, info


class GapEnv(_MoveEnv):
    """params mode: one vel/acc per move (a movej).

    Geometry is the movej's straight joint line start -> dest; vel/acc set a
    trapezoidal speed profile along it.
    """

    def __init__(self, *args, objective=OBJECTIVE, **kw):
        super().__init__(*args, **kw)
        self.objective = objective
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32),
                                       high=np.array([1.0, 1.0], dtype=np.float32))

    def _geometry(self, move) -> np.ndarray:
        start, travel, _ = move_line(self.rec, move)
        return start + np.linspace(0.0, 1.0, GRID)[:, None] * travel

    def _unmap(self, action) -> tuple[float, float]:
        """Map a normalized action in [-1,1]^2 to (vel, acc) in deg/s, deg/s^2."""
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        vel = VEL_BOUNDS[0] + (a[0] + 1) / 2 * (VEL_BOUNDS[1] - VEL_BOUNDS[0])
        acc = ACC_BOUNDS[0] + (a[1] + 1) / 2 * (ACC_BOUNDS[1] - ACC_BOUNDS[0])
        return float(vel), float(acc)

    def score(self, move, vel_deg, acc_deg) -> tuple[float, float]:
        """(score, cycle_time) for a movej of ``move`` at vel/acc in deg/s."""
        _, _, distance = move_line(self.rec, move)
        vel = min(vel_deg * DEG2RAD, MAX_JOINT_SPEED)
        acc = min(acc_deg * DEG2RAD, MAX_JOINT_ACC)
        dt = self.rec.dt
        s = trapezoidal(distance, vel, acc, dt)
        return float(_aggregate(self._candidate(move, s, dt, vel_deg, acc_deg))), len(s) * dt

    def _cost(self, move, action) -> tuple[float, float, float]:
        score, cycle = self.score(move, *self._unmap(action))
        return score, cycle, self.objective(score, cycle)


class PathEnv(_MoveEnv):
    """path mode: replay the recorded move at a new speed profile (servoj stream).

    Geometry is the recorded joint trajectory of the move; the agent generates only
    the speed profile along it, so the path is preserved and only the timing changes.

        action : [accel_frac, decel_frac, speed]
        reward : -PATH_OBJECTIVE(max per-row score, cycle_time)

    ``accel_frac``/``decel_frac`` shape one trapezoidal accel-cruise-decel profile
    (see `speed_profile`); ``speed`` sets the per-row servoj time.
    """

    def __init__(self, *args, objective=PATH_OBJECTIVE, **kw):
        super().__init__(*args, **kw)
        self.objective = objective
        self.action_space = spaces.Box(low=np.full(3, -1.0, dtype=np.float32),
                                       high=np.full(3, 1.0, dtype=np.float32))

    def _geometry(self, move) -> np.ndarray:
        return self.rec.target_q[move.i0:move.i1]

    def unpack(self, action) -> tuple[float, float, float]:
        """Map a normalized action to (accel_frac, decel_frac, servoj dt)."""
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        accel_frac = (a[0] + 1.0) / 2.0 * 0.9
        decel_frac = (a[1] + 1.0) / 2.0 * 0.9
        lo, hi = SERVO_DT_BOUNDS
        dt = lo + (a[2] + 1.0) / 2.0 * (hi - lo)
        return accel_frac, decel_frac, dt

    def score(self, move, accel_frac, decel_frac, dt) -> tuple[float, float]:
        """(max per-row score, cycle_time) for a re-timed path of ``move``."""
        s = speed_profile(accel_frac, decel_frac, PATH_ROWS)
        vel = float(move.vel) if move.vel is not None else 0.0
        acc = float(move.acc) if move.acc is not None else 0.0
        return float(self._candidate(move, s, dt, vel, acc).max()), PATH_ROWS * dt

    def baseline(self, move) -> tuple[float, float]:
        """(max per-row score, cycle_time) for the recorded motion, its own timing.

        The recorded trajectory carries the controller's speed profile from the
        move's vel/acc, so this is the fixed baseline the re-timing is compared to.
        """
        g = self._geo[move.i0]                       # recorded q(i0:i1), full resolution
        n, dt = len(g), self.rec.dt
        s = np.linspace(0.0, 1.0, n)
        vel = float(move.vel) if move.vel is not None else 0.0
        acc = float(move.acc) if move.acc is not None else 0.0
        frame = self.dyn.frame(g, dt, vel, acc, s=s, key=move.i0)
        return float(evaluate(self.model, self.metric, frame, self.pre).max()), n * dt

    def _cost(self, move, action) -> tuple[float, float, float]:
        max_score, cycle = self.score(move, *self.unpack(action))
        return max_score, cycle, self.objective(max_score, cycle)


class _TrainCallback(BaseCallback):
    """Records per-episode reward, score, and cycle_time during PPO training."""

    def __init__(self):
        super().__init__(verbose=0)
        self.records: list[dict] = []

    def _on_step(self) -> bool:
        dones   = self.locals.get("dones",   [])
        rewards = self.locals.get("rewards", [])
        infos   = self.locals.get("infos",   [])
        for done, reward, info in zip(dones, rewards, infos):
            if done:
                self.records.append({
                    "timestep":   int(self.num_timesteps),
                    "reward":     float(reward),
                    "score":      float(info.get("score",      float("nan"))),
                    "cycle_time": float(info.get("cycle_time", float("nan"))),
                })
        return True


def _rolling_mean(x: list, w: int) -> list:
    return [float(np.mean(x[max(0, i - w): i + 1])) for i in range(len(x))]


def log_training_run(mode: str, scripts: list, steps: int, agent_path: str,
                     callback: _TrainCallback, results_dir: str, dt_str: str,
                     agent_versioned: str = None):
    """Save log.json, training_curve.png, and update runs_summary_rla.csv."""
    run_dir = os.path.join(results_dir, f"{dt_str}_rla_{mode}")
    os.makedirs(run_dir, exist_ok=True)

    records = callback.records
    if not records:
        print("[results] no training records — skipping")
        return

    timesteps   = [r["timestep"]   for r in records]
    rewards     = [r["reward"]     for r in records]
    scores      = [r["score"]      for r in records]
    cycle_times = [r["cycle_time"] for r in records]

    tail = max(1, len(scores) // 10)
    best_score  = float(np.nanmin(scores))
    final_score = float(np.nanmean(scores[-tail:]))
    best_cycle  = float(np.nanmin(cycle_times))

    # ---- log.json ------------------------------------------------------------
    log = {
        "datetime":          dt_str,
        "type":              "train_rla",
        "mode":              mode,
        "scripts":           scripts,
        "steps":             steps,
        "agent_path_latest": agent_path,
        "agent_path":        agent_versioned or agent_path,
        "summary": {
            "n_episodes":       len(records),
            "best_score":       best_score,
            "final_score_mean": final_score,
            "best_cycle_time":  best_cycle,
        },
    }
    log_path = os.path.join(run_dir, "log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[results] log  -> {log_path}")

    # ---- training_curve.png --------------------------------------------------
    w = max(1, len(scores) // 20)
    sm_score  = _rolling_mean(scores,      w)
    sm_reward = _rolling_mean(rewards,     w)
    sm_cycle  = _rolling_mean(cycle_times, w)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(timesteps, scores,    alpha=0.15, color="steelblue",  linewidth=0.6)
    axes[0].plot(timesteps, sm_score,  color="steelblue",  linewidth=1.6,
                 label=f"rolling mean (w={w})")
    axes[0].axhline(best_score, color="green", linestyle="--", linewidth=1.0,
                    label=f"best = {best_score:.4f}")
    axes[0].set_ylabel("Score")
    axes[0].set_title(f"Training Curve — {mode} mode | {steps} steps | "
                      f"{len(records)} episodes")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(timesteps, cycle_times, alpha=0.15, color="darkorange", linewidth=0.6)
    axes[1].plot(timesteps, sm_cycle,    color="darkorange", linewidth=1.6,
                 label=f"rolling mean (w={w})")
    axes[1].axhline(best_cycle, color="green", linestyle="--", linewidth=1.0,
                    label=f"best = {best_cycle:.3f} s")
    axes[1].set_ylabel("Cycle time (s)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(timesteps, rewards,    alpha=0.15, color="purple", linewidth=0.6)
    axes[2].plot(timesteps, sm_reward,  color="purple", linewidth=1.6,
                 label=f"rolling mean (w={w})")
    axes[2].set_ylabel("Reward (−objective)")
    axes[2].set_xlabel("Timestep")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(run_dir, "training_curve.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[results] plot -> {plot_path}")

    # ---- runs_summary_rla.csv ------------------------------------------------
    summary_path = os.path.join(results_dir, "runs_summary_rla.csv")
    row = {
        "datetime":         dt_str,
        "mode":             mode,
        "steps":            steps,
        "n_episodes":       len(records),
        "best_score":       best_score,
        "final_score_mean": final_score,
        "best_cycle_time":  best_cycle,
        "scripts":          ";".join(scripts),
    }
    summary_df = (pd.concat([pd.read_csv(summary_path), pd.DataFrame([row])],
                             ignore_index=True)
                  if os.path.exists(summary_path) else pd.DataFrame([row]))
    summary_df.to_csv(summary_path, index=False)
    print(f"[results] summary -> {summary_path}")
    print(f"[results] run complete -> {run_dir}")


def train_ppo(env, steps: int, out: str, versioned_out: str = None) -> tuple:
    """Train a PPO agent on an env, save it, and return (agent, callback).

    ``out`` is the standard "latest" path (e.g. models/agent_params.zip).
    ``versioned_out`` is an optional second save path for the timestamped copy.
    """
    cb    = _TrainCallback()
    agent = PPO("MlpPolicy", env, verbose=0)
    agent.learn(total_timesteps=steps, callback=cb)
    agent.save(out)
    if versioned_out:
        agent.save(versioned_out)
    return agent, cb


def build_dataset(model, metric, scripts, robot_ip, loop, pre=None, out=SIM_TO_REAL):
    """Run scripts on URSim, predict actuals, label score -> ``out`` csv.

    Writes the three data stages to one file: sim targets, distilled actual_*
    columns, and the ``score`` label. ``pre`` wraps the distill model in
    ``augment`` as in training. Returns the loaded ``Recording``; run.py reuses it,
    so both take the same URSim -> distill -> label path.
    """
    pre = pre or Identity()
    collect_moves(scripts, robot_ip, loop, out)
    augment(model, out, pre)
    df = add_score(pd.read_csv(out), metric)
    df.to_csv(out, index=False)
    print(f"labelled score -> {out}")
    return Recording(out)


def main():
    ap = argparse.ArgumentParser(description="Train an RL agent on the gap model.")
    ap.add_argument("--model", default="models/distill.pkl", help="distilled model")
    ap.add_argument("--mode", choices=("params", "path"), default="params",
                    help="params: pick vel/acc per move; path: shape a servoj path")
    ap.add_argument("--scripts", nargs="+", default=["scripts/shoulder_swing.script"],
                    help="URScript(s) to run on URSim; their moves are the training set")
    ap.add_argument("--robot-ip", default="127.0.0.1",
                    help="URSim IP to run the scripts on (default local URSim)")
    ap.add_argument("--loop", type=int, default=None,
                    help="repeat each script N times when collecting moves")
    ap.add_argument("--steps", type=int, default=20000, help="PPO timesteps")
    ap.add_argument("--out", default=None,
                    help="agent save path (default: models/agent_<mode>.zip)")
    args = ap.parse_args()
    out = args.out or f"models/agent_{args.mode}.zip"

    # Create the run directory now so sim_to_real.csv lands inside it rather
    # than at the case 2 root.
    dt_str  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(RESULTS_DIR, f"{dt_str}_rla_{args.mode}")
    os.makedirs(run_dir, exist_ok=True)

    model = DistillModel.load(args.model)
    metric = CurrentGapMetric()
    pre = default_preprocess()
    sim_csv = os.path.join(run_dir, "sim_to_real.csv")
    rec = build_dataset(model, metric, args.scripts, args.robot_ip, args.loop, pre, sim_csv)
    dyn = default_dynamics(rec)
    Env = GapEnv if args.mode == "params" else PathEnv
    env = Env(model, metric, rec, dyn=dyn, pre=pre)

    agent_versioned = os.path.join(run_dir, f"agent_{args.mode}.zip")
    print(f"mode: {args.mode}   training on {len(env.targets)} segments")
    _, cb = train_ppo(env, args.steps, out, versioned_out=agent_versioned)
    print(f"trained PPO ({args.mode}) for {args.steps} steps")
    print(f"  latest   -> {out}")
    print(f"  versioned -> {agent_versioned}")

    log_training_run(args.mode, args.scripts, args.steps, out, cb, RESULTS_DIR, dt_str,
                     agent_versioned=agent_versioned)


if __name__ == "__main__":
    main()
