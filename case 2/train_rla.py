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
import time
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

import ur_style
from analysis import Recording
from common import segments
from dynamics import (DEG2RAD, GRID, MAX_JOINT_ACC, MAX_JOINT_SPEED, Dynamics,
                      default_dynamics, trapezoidal)
from train_distillation_model import DistillModel, augment
from metrics import PositionGapMetric, EvaluationMetric, SCORE_COL, add_score
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
#
# These weights are metric-scale-dependent and must be recalibrated whenever
# the active EvaluationMetric changes (see metrics.py) -- they were originally
# 1.0/1.0, tuned for CurrentGapMetric, whose amps-scale score (~1-10 after
# _aggregate) happens to be roughly comparable to cycle_time in seconds.
# PositionGapMetric's score is in radians and nowhere near that scale.
# Measured directly against what the env actually calls during a step (not a
# proxy -- GapEnv.score()/PathEnv.score() with randomly sampled actions, real
# recordings, models/distill.pkl):
#   GapEnv  (RMS-aggregated score, GapEnv.score, 40 random-vel/acc samples):
#       mean score 0.00107 rad, mean cycle_time 2.71 s -> ratio ~2524x
#   PathEnv (max per-row score, PathEnv.score with sampled path actions --
#            NOT PathEnv.baseline(), which times the full recorded
#            trajectory and gave a misleading ~1929x on a first pass; the
#            actual per-step cycle_time is PATH_ROWS(50) * agent-chosen
#            servoj dt, a much shorter window):
#       mean score 0.03002 rad, mean cycle_time 0.6085 s -> ratio ~20x
# Leaving SCORE_WEIGHT/PATH_SCORE_WEIGHT at 1.0 with PositionGapMetric would
# make the objective approx= cycle_time alone -- the agent would optimize
# almost purely for speed, the vibration-minimization term silently
# negligible. Values below bring both terms to comparable influence,
# verified directly (case 2/results/../ see ModelReview.md): GapEnv's ratio
# came out 0.95 (balanced), PathEnv's 79.5 on a first pass with the wrong
# calibration, ~1 after the fix. CYCLE_WEIGHT/PATH_CYCLE_WEIGHT stay 1.0 as
# the stable reference the score weight is calibrated against.
#
# RECALIBRATED 2026-08-20 (see tune_reward_weights.py, RLAgent.md §4a): the
# 0.95 ratio above was measured from *random-action* sampling before any
# training happened. It doesn't hold once a policy is actually trained --
# checked against a real 20,000-step run's own episodes.csv (20,480
# episodes, results/2026-08-20_12-57-02_rla_params/): mean score 0.000158
# rad, mean cycle_time 1.715s. At the original SCORE_WEIGHT=2500 that's a
# 0.231:1 (score:cycle) weighted ratio, not ~1:1 -- cycle_time dominated
# the gradient the whole run, which is exactly why that run's score barely
# moved (even worsened slightly) while cycle_time dropped ~3s->~1.2s.
# First retargeted at a deliberate 2:1 ratio (21700, see git history),
# then explicitly set back to **1:1** (score and cycle_time weighted
# equally, no deliberate favoring of either) --
# `python tune_reward_weights.py --episodes <run>/episodes.csv --target-ratio 1.0`
# recommended 10838.5 against the same episode data; rounded. PATH_SCORE_WEIGHT
# is unchanged/unverified against this same drift -- no path-mode training
# run has happened yet to check it against.
SCORE_WEIGHT = 10800.0
CYCLE_WEIGHT = 1.0
OBJECTIVE = lambda score, cycle_time: SCORE_WEIGHT * score + CYCLE_WEIGHT * cycle_time

# Path-mode cost. ``max_score`` is the worst per-row score over the path.
PATH_SCORE_WEIGHT = 20.0
PATH_CYCLE_WEIGHT = 1.0
PATH_OBJECTIVE = lambda max_score, cycle_time: \
    PATH_SCORE_WEIGHT * max_score + PATH_CYCLE_WEIGHT * cycle_time

# Path action: a trapezoidal speed profile (accel_frac, decel_frac) + a per-row
# servoj time from SERVO_DT_BOUNDS (smaller = faster). Scored on PATH_ROWS samples.
PATH_ROWS = 50
SERVO_DT_BOUNDS = (0.004, 0.02)

# Settling window scored for vibration: seconds of held-still commanded
# position appended after a candidate's active motion, so the PDF's
# peak/rms ("for t in the settling window") formulas have a real post-stop
# window to score, rather than being computed over the motion itself.
# Measured, not guessed: bronze_tier/segment_stats.csv's real i1->i2 windows
# (2497 real segments) have median 0.749s -- rounded to 0.75s. That
# distribution is bimodal (~1/4 of segments are back-to-back moves with an
# i1->i2 near 0, the rest cluster tightly at ~0.75s, matching scripts that
# do pause between moves) -- 0.75s reflects the "there is a real settle
# period" regime, which is the one worth scoring.
SETTLE_S = 0.75


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

    def _score_settled(self, move, q, s, dt, vel_deg, acc_deg) -> np.ndarray:
        """Per-row score over the settling window after ``q``'s own motion.

        Appends ``SETTLE_S`` seconds of the commanded position held still at
        ``q[-1]`` (the destination), scores the *whole* extended trajectory
        through the model and metric, and returns only the appended rows'
        scores -- not the motion's. This is the fix for a real gap: without
        it, a candidate trajectory ended the instant its commanded motion did,
        so there was no post-stop window at all for the PDF's peak/rms
        ("for t in the settling window") to be computed over -- RMS/peak were
        silently being computed over the motion itself instead, structurally
        unable to see any post-stop ring even if the model could predict one.

        The held-still portion still goes through ``dynamics.Dynamics.frame``
        (not skipped) so ``target_qd``/``target_qdd`` come out as a
        continuous, physically sensible decay through the stop (via
        ``np.gradient`` over the *combined* array) rather than an abrupt
        jump to exactly zero, and so ``current()`` gets a real qdd=0 (holding
        torque = gravity torque, correctly) instead of an arbitrary value.
        ``s`` is held at its final value for every appended row, so the
        cached per-move pose terms (``Dynamics.prepare``) are looked up at
        the destination pose throughout, matching a robot that has arrived
        and stopped moving.
        """
        n_settle = max(1, int(round(SETTLE_S / dt)))
        q_settled = np.vstack([q, np.tile(q[-1], (n_settle, 1))])
        s_settled = np.concatenate([np.asarray(s, dtype=float), np.full(n_settle, float(s[-1]))])
        frame = self.dyn.frame(q_settled, dt, vel_deg, acc_deg, s=s_settled, key=move.i0)
        scores = evaluate(self.model, self.metric, frame, self.pre)
        return scores[len(q):]

    def _candidate(self, move, s, dt, vel_deg, acc_deg):
        """Settle-window score of the candidate whose active-motion progress
        is ``s(t)`` at step ``dt``.

        Places the move's geometry at the given progress, then scores only
        the settling window after it (``_score_settled``) -- matching the
        PDF's peak/rms definitions, not the motion itself.
        """
        q = self._q_at(move, s)
        return self._score_settled(move, q, s, dt, vel_deg, acc_deg)

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
        """(max settle-window score, cycle_time) for the recorded motion, its own timing.

        The recorded trajectory carries the controller's speed profile from the
        move's vel/acc, so this is the fixed baseline the re-timing is compared to.
        Routed through ``_score_settled`` (same as ``score()``'s ``_candidate``
        call) so this is a peak-over-settling-window number, not a peak-over-the-
        motion number -- otherwise baseline vs. optimized would be comparing two
        different quantities, not the same quantity at two speeds.
        """
        g = self._geo[move.i0]                       # recorded q(i0:i1), full resolution
        n, dt = len(g), self.rec.dt
        s = np.linspace(0.0, 1.0, n)
        vel = float(move.vel) if move.vel is not None else 0.0
        acc = float(move.acc) if move.acc is not None else 0.0
        return float(self._score_settled(move, g, s, dt, vel, acc).max()), n * dt

    def _cost(self, move, action) -> tuple[float, float, float]:
        max_score, cycle = self.score(move, *self.unpack(action))
        return max_score, cycle, self.objective(max_score, cycle)


class _TrainCallback(BaseCallback):
    """Records per-episode reward, score, and cycle_time during PPO training,
    and prints a progress line roughly every ``print_every`` seconds.

    PPO itself runs with ``verbose=0`` (see ``train_ppo``) and this callback
    previously only recorded episodes silently -- a run's entire terminal
    output was one line before training and one line after, with nothing in
    between regardless of how long ``total_timesteps`` takes. Time-based
    (not step-count-based) printing, since steps/sec varies a lot with the
    environment's per-step cost (dynamics/candidate scoring) -- a fixed
    step-count interval would either flood a fast run or stay silent for a
    slow one.
    """

    def __init__(self, total_steps: int, print_every: float = 5.0):
        super().__init__(verbose=0)
        self.records: list[dict] = []
        self.total_steps = total_steps
        self.print_every = print_every
        self._last_print = time.monotonic()

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
        now = time.monotonic()
        if self.print_every > 0 and now - self._last_print >= self.print_every:
            self._last_print = now
            pct    = 100.0 * self.num_timesteps / self.total_steps if self.total_steps else 0.0
            recent = self.records[-20:]
            mean_score = float(np.mean([r["score"] for r in recent])) if recent else float("nan")
            print(f"  step {self.num_timesteps}/{self.total_steps} ({pct:.0f}%)   "
                 f"episodes: {len(self.records)}   recent mean score: {mean_score:.4f}",
                 flush=True)
        return True


def _rolling_mean(x: list, w: int) -> list:
    return [float(np.mean(x[max(0, i - w): i + 1])) for i in range(len(x))]


def log_training_run(mode: str, scripts: list, steps: int, agent_path: str,
                     callback: _TrainCallback, results_dir: str, dt_str: str,
                     agent_versioned: str = None, seed: int = None):
    """Save log.json, training_curve.png, and update runs_summary_rla.csv.

    ``seed`` is recorded (not just used) so a later run-to-run comparison can
    tell whether two runs used the same seed (a real model/data diff) or
    different ones (partly just seed noise, see ``train_ppo``).
    """
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
        "seed":              seed,
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

    # ---- episodes.csv ---------------------------------------------------------
    # Full per-episode records (log.json only keeps aggregate stats) -- lets a
    # later run check whether the trained policy's actual score/cycle_time
    # distribution still balances the way SCORE_WEIGHT was calibrated against
    # (see tune_reward_weights.py), without retraining.
    episodes_path = os.path.join(run_dir, "episodes.csv")
    pd.DataFrame(records).to_csv(episodes_path, index=False)
    print(f"[results] episodes -> {episodes_path}")

    # ---- training_curve.png --------------------------------------------------
    ur_style.apply()
    w = max(1, len(scores) // 20)
    sm_score  = _rolling_mean(scores,      w)
    sm_reward = _rolling_mean(rewards,     w)
    sm_cycle  = _rolling_mean(cycle_times, w)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(timesteps, scores,    alpha=0.15, color=ur_style.BLUE, linewidth=0.6)
    axes[0].plot(timesteps, sm_score,  color=ur_style.BLUE, linewidth=1.6,
                 label=f"rolling mean (w={w})")
    axes[0].axhline(best_score, color=ur_style.GRAY, linestyle="--", linewidth=1.0,
                    label=f"best = {best_score:.4f}")
    axes[0].set_ylabel("Score (rad — lower = better)")
    axes[0].set_title(f"Training Curve — {mode} mode | {steps} steps | "
                      f"{len(records)} episodes")
    axes[0].legend(fontsize=8)

    axes[1].plot(timesteps, cycle_times, alpha=0.15, color=ur_style.MID_BLUE, linewidth=0.6)
    axes[1].plot(timesteps, sm_cycle,    color=ur_style.MID_BLUE, linewidth=1.6,
                 label=f"rolling mean (w={w})")
    axes[1].axhline(best_cycle, color=ur_style.GRAY, linestyle="--", linewidth=1.0,
                    label=f"best = {best_cycle:.3f} s")
    axes[1].set_ylabel("Cycle time (s — lower = faster)")
    axes[1].legend(fontsize=8)

    axes[2].plot(timesteps, rewards,    alpha=0.15, color=ur_style.DARK_BLUE, linewidth=0.6)
    axes[2].plot(timesteps, sm_reward,  color=ur_style.DARK_BLUE, linewidth=1.6,
                 label=f"rolling mean (w={w})")
    axes[2].set_ylabel("Reward (−cost — higher = better)")
    axes[2].set_xlabel("Timestep")
    axes[2].legend(fontsize=8)

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
        "seed":             seed,
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


def train_ppo(env, steps: int, out: str, versioned_out: str = None,
             seed: int = None, print_every: float = 5.0) -> tuple:
    """Train a PPO agent on an env, save it, and return (agent, callback).

    ``out`` is the standard "latest" path (e.g. models/agent_params.zip).
    ``versioned_out`` is an optional second save path for the timestamped copy.
    ``print_every`` (seconds) controls how often ``_TrainCallback`` prints a
    progress line during training; pass ``0`` to go back to silent.

    ``seed`` makes the run reproducible: SB3's ``PPO(seed=...)`` seeds the
    policy's weight init and its own RNG *and* seeds ``env`` (python/numpy/
    torch RNG plus the env's own ``np_random``, via
    ``set_random_seed``->``env.seed``) before the first ``reset()`` --
    without it, ``_MoveEnv.reset``'s ``self.np_random.integers(...)`` move
    pick is auto-seeded from OS entropy, so two "identical" runs pick a
    different sequence of moves and PPO initializes different weights, both
    contributing unseeded noise on top of whatever a real model change is
    supposed to show. Comparing two distill models' effect on the trained
    policy is only meaningful if this noise source is pinned down first --
    e.g. by rerunning the *same* model/seed pair to see how much the curve
    naturally wobbles run-to-run before trusting a model-to-model diff.
    """
    cb    = _TrainCallback(total_steps=steps, print_every=print_every)
    agent = PPO("MlpPolicy", env, verbose=0, seed=seed)
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
    ap.add_argument("--seed", type=int, default=0,
                    help="PPO/env RNG seed (default: %(default)s); fix this and "
                        "everything else to compare two distill models' effect "
                        "on the trained policy without seed noise confounding "
                        "it -- see train_ppo's docstring")
    ap.add_argument("--out", default=None,
                    help="agent save path (default: models/agent_<mode>.zip)")
    ap.add_argument("--print-every", type=float, default=5.0,
                    help="seconds between training-progress print lines "
                        "(default: %(default)s; 0 disables)")
    args = ap.parse_args()
    out = args.out or f"models/agent_{args.mode}.zip"

    # Create the run directory now so sim_to_real.csv lands inside it rather
    # than at the case 2 root.
    dt_str  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(RESULTS_DIR, f"{dt_str}_rla_{args.mode}")
    os.makedirs(run_dir, exist_ok=True)

    model = DistillModel.load(args.model)
    metric = PositionGapMetric()
    pre = default_preprocess()
    sim_csv = os.path.join(run_dir, "sim_to_real.csv")
    rec = build_dataset(model, metric, args.scripts, args.robot_ip, args.loop, pre, sim_csv)
    dyn = default_dynamics(rec)
    Env = GapEnv if args.mode == "params" else PathEnv
    env = Env(model, metric, rec, dyn=dyn, pre=pre)

    agent_versioned = os.path.join(run_dir, f"agent_{args.mode}.zip")
    print(f"mode: {args.mode}   training on {len(env.targets)} segments   seed: {args.seed}")
    _, cb = train_ppo(env, args.steps, out, versioned_out=agent_versioned, seed=args.seed,
                      print_every=args.print_every)
    print(f"trained PPO ({args.mode}) for {args.steps} steps")
    print(f"  latest   -> {out}")
    print(f"  versioned -> {agent_versioned}")

    log_training_run(args.mode, args.scripts, args.steps, out, cb, RESULTS_DIR, dt_str,
                     agent_versioned=agent_versioned, seed=args.seed)


if __name__ == "__main__":
    main()
