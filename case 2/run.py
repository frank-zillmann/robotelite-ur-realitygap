"""Optimize a motion against the models and write it out for the robot.

Runs train_rla.py's pipeline without training: builds the dataset for the chosen
script (`build_dataset`), loads the trained agent, asks it for a motion, scores it
against the models, and writes the result. Two modes:

  params  rewrite the script's `vel`/`acc` lines (movej). Writes
          `scripts/<name>.optimized.script`; run it with `send.py --script`.
  path    re-time each move into a servoj path. Writes `scripts/<name>.path`
          (a CSV of joint setpoints); run it with `send.py --path`.

    python run.py --mode params --script scripts/triangle.script --robot-ip 127.0.0.1
    python run.py --mode path   --script scripts/triangle.script --robot-ip 127.0.0.1

Requires an agent trained by `train_rla.py`; reads `models/agent_<mode>.zip`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ur_style
from metrics import PositionGapMetric
from preprocess import default_preprocess
from train_distillation_model import DistillModel
from dynamics import DEG2RAD
from train_rla import GapEnv, PathEnv, build_dataset, load_agent, speed_profile
from utils import get_param, load_script, set_param

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# --- params mode -------------------------------------------------------------

def script_moves(env):
    """The script's moves, dropping the first (the approach into the script from
    wherever the robot started, which is not part of the script's own motion)."""
    return env.targets[1:] if len(env.targets) > 1 else env.targets


def _gain(base, opt) -> float:
    """Percent reduction from baseline mean to optimized mean (positive = lower)."""
    return 100 * (base.mean() - opt.mean()) / base.mean()


def plot_baseline_vs_optimized(base: np.ndarray, opt: np.ndarray, mode: str,
                               out_path: str):
    """Mean score and cycle time, baseline vs. RL-optimized, side by side.

    ``base``/``opt`` are ``(n_moves, 2)`` arrays (column 0 = score, column 1 =
    cycle_time) from ``report``/``report_path``. This is the "did it actually
    work" plot the training curve doesn't cover -- that one shows how
    training progressed over timesteps, this one shows what the agent's
    final choice is worth against the script's original fixed vel/acc,
    which is the number the case brief actually asks for. Score and
    cycle_time are different units, so they get separate panels rather than
    one bar chart -- same reasoning as train_distillation_model.py's
    per-joint RMSE/R² panels.
    """
    ur_style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    labels = ["baseline\n(script)", "optimized\n(agent)"]

    for ax, col, ylabel, fmt in [
        (axes[0], 0, "Score (lower = less vibration)", "{:.4f}"),
        (axes[1], 1, "Cycle time (s)", "{:.3f}"),
    ]:
        b, o = float(base[:, col].mean()), float(opt[:, col].mean())
        bars = ax.bar(labels, [b, o], color=[ur_style.GRAY, ur_style.BLUE],
                      edgecolor=ur_style.NAVY)
        span = max(abs(b), abs(o)) or 1.0
        for bar, v in zip(bars, [b, o]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015 * span,
                    fmt.format(v), ha="center", va="bottom", fontsize=9.5,
                    color=ur_style.NAVY)
        gain = 100 * (b - o) / b if b else 0.0
        ax.set_ylabel(ylabel)
        ax.set_title(f"{gain:+.1f}%", color=(ur_style.BLUE if gain >= 0 else ur_style.NAVY),
                    fontweight="bold")

    fig.suptitle(f"Baseline vs. RL-optimized — {mode} mode", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[results] plot -> {out_path}")


def _compare(base, opt):
    """Print score and cycle-time change (base/opt are (score, cycle) arrays)."""
    print(f"predicted mean score: {base[:, 0].mean():.3f} -> {opt[:, 0].mean():.3f} "
          f"({_gain(base[:, 0], opt[:, 0]):+.1f}%)")
    print(f"predicted cycle time: {base[:, 1].mean():.3f} -> {opt[:, 1].mean():.3f}s "
          f"({_gain(base[:, 1], opt[:, 1]):+.1f}%)")


def search_agent(env: GapEnv, moves, path: str) -> tuple[float, float]:
    """Ask a trained agent for vel/acc (deg/s), averaged over the moves."""
    agent = load_agent(path)
    params = [env._unmap(agent.predict(env._obs(m), deterministic=True)[0]) for m in moves]
    return tuple(np.mean(params, axis=0))


def report(env: GapEnv, moves, vel: float, acc: float, label: str) -> np.ndarray:
    """Print mean score and cycle time for one (vel, acc); return (score, cycle) per move."""
    res = np.array([env.score(m, vel, acc) for m in moves])
    print(f"{label}: vel {vel:.0f}  acc {acc:.0f}  score {res[:, 0].mean():.3f}  "
          f"cycle {res[:, 1].mean():.3f}s (mean over {len(moves)} moves)")
    return res


def run_params(args, model, metric, rec, pre, run_dir: str) -> dict:
    """Optimize the script's vel/acc and write the optimized script.

    The script's `vel =`/`acc =` lines feed `movej(a=acc, v=vel)` directly --
    URScript's own units for a joint move, rad/s and rad/s^2 (see
    dynamics.py's ``MAX_JOINT_SPEED`` comment). ``GapEnv.score``/the agent's
    ``_unmap`` work in deg/s, deg/s^2 (``VEL_BOUNDS``/``ACC_BOUNDS``). Every
    read from the script is converted rad -> deg before use here, and the
    write back to the script is converted deg -> rad -- without that, the
    optimized script's numbers are off by 180/pi (~57.3x): a `movej` speed
    that far past the joint limit doesn't scale the motion, it just clamps to
    the robot's hard max regardless of what the agent actually chose.
    """
    text = load_script(args.script)
    env = GapEnv(model, metric, rec, pre=pre)
    moves = script_moves(env)

    base_vel = get_param(text, "vel") / DEG2RAD
    base_acc = get_param(text, "acc") / DEG2RAD
    base = report(env, moves, base_vel, base_acc, "baseline (from script)")
    vel, acc = search_agent(env, moves, args.agent)
    opt = report(env, moves, vel, acc, "optimized")
    _compare(base, opt)
    plot_baseline_vs_optimized(base, opt, args.mode,
                               os.path.join(run_dir, "baseline_vs_optimized.png"))

    out = re.sub(r"\.script$", ".optimized.script", args.script)
    with open(out, "w") as f:
        f.write(set_param(set_param(text, "vel", vel * DEG2RAD), "acc", acc * DEG2RAD))
    print(f"wrote {out}\n  run it: python send.py --script {out} --out optimized.csv")

    return {
        "baseline":    {"vel": float(base_vel), "acc": float(base_acc),
                        "score": float(base[:, 0].mean()),
                        "cycle_time": float(base[:, 1].mean())},
        "optimized":   {"vel": float(vel), "acc": float(acc),
                        "score": float(opt[:, 0].mean()),
                        "cycle_time": float(opt[:, 1].mean())},
        "improvement": {"score_pct":      float(_gain(base[:, 0], opt[:, 0])),
                        "cycle_time_pct": float(_gain(base[:, 1], opt[:, 1]))},
        "output":      out,
    }


# --- path mode ---------------------------------------------------------------

def agent_paths(env: PathEnv, moves, path: str):
    """Per move, the agent's (accel_frac, decel_frac, servoj dt)."""
    agent = load_agent(path)
    return [env.unpack(agent.predict(env._obs(m), deterministic=True)[0]) for m in moves]


def report_path(env: PathEnv, moves, plans, label: str) -> np.ndarray:
    """Print mean max-score and cycle time; return (score, cycle) per move."""
    res = np.array([env.score(m, af, df, dt) for m, (af, df, dt) in zip(moves, plans)])
    print(f"{label}: max score {res[:, 0].mean():.3f}  cycle {res[:, 1].mean():.3f}s "
          f"(mean over {len(moves)} moves)")
    return res


def build_full_path(rec, moves, plans):
    """6-DOF servoj setpoints for the whole script, one block per move.

    Each move replays its recorded joint trajectory, re-timed by the agent's speed
    profile, so the path is preserved and only the speed changes. Output at the
    recorded resolution (dense) so the servoj locus matches the recording. Each row
    is ``q0..q5`` plus the move's servoj dt (a 7th column send.py reads).
    """
    rows = []
    for m, (accel_frac, decel_frac, dt) in zip(moves, plans):
        rec_q = rec.target_q[m.i0:m.i1]                 # recorded geometry, all joints
        n = len(rec_q)
        u = np.linspace(0.0, 1.0, n)
        s = speed_profile(accel_frac, decel_frac, n)    # re-timed, recorded resolution
        block = np.column_stack([np.interp(s, u, rec_q[:, j]) for j in range(6)])
        for r in block:
            rows.append([*r, dt])
    return rows


def run_path(args, model, metric, rec, pre, run_dir: str) -> dict:
    """Re-time each move and write the path CSV."""
    env = PathEnv(model, metric, rec, pre=pre)
    moves = script_moves(env)

    base = np.array([env.baseline(m) for m in moves])
    print(f"baseline (recorded speed): max score {base[:, 0].mean():.3f}  "
          f"cycle {base[:, 1].mean():.3f}s (mean over {len(moves)} moves)")
    plans = agent_paths(env, moves, args.agent)
    opt = report_path(env, moves, plans, "optimized")
    _compare(base, opt)
    plot_baseline_vs_optimized(base, opt, args.mode,
                               os.path.join(run_dir, "baseline_vs_optimized.png"))

    rows = build_full_path(rec, moves, plans)
    out = re.sub(r"\.script$", ".path", args.script)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"q{j}" for j in range(6)] + ["dt"])
        w.writerows([f"{v:.6f}" for v in row] for row in rows)
    print(f"wrote {out}\n  run it: python send.py --path {out} --out optimized.csv")

    return {
        "baseline":    {"score": float(base[:, 0].mean()),
                        "cycle_time": float(base[:, 1].mean())},
        "optimized":   {"score": float(opt[:, 0].mean()),
                        "cycle_time": float(opt[:, 1].mean())},
        "improvement": {"score_pct":      float(_gain(base[:, 0], opt[:, 0])),
                        "cycle_time_pct": float(_gain(base[:, 1], opt[:, 1]))},
        "output":      out,
    }


def log_run_result(args, metrics: dict, sim_csv: str, results_dir: str, dt_str: str):
    """Save log.json for this run.py invocation to results/<datetime>_run_<mode>/."""
    run_dir  = os.path.join(results_dir, f"{dt_str}_run_{args.mode}")
    os.makedirs(run_dir, exist_ok=True)
    log = {
        "datetime":      dt_str,
        "type":          "run",
        "mode":          args.mode,
        "script":        args.script,
        "distill_model": args.model,
        "agent":         args.agent,
        "sim_csv":       sim_csv,
        "results":       metrics,
    }
    log_path = os.path.join(run_dir, "log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[results] log -> {log_path}")


def main():
    ap = argparse.ArgumentParser(description="Optimize a URScript motion against the model.")
    ap.add_argument("--mode", choices=("params", "path"), default="params",
                    help="params: rewrite vel/acc; path: shape a servoj path")
    ap.add_argument("--script", default="scripts/shoulder_swing.script",
                    help="URScript to optimize")
    ap.add_argument("--model", default="models/distill.pkl", help="distilled model")
    ap.add_argument("--robot-ip", default="127.0.0.1",
                    help="URSim IP to run the script on (default local URSim)")
    ap.add_argument("--loop", type=int, default=None,
                    help="repeat the script N times when collecting its moves")
    ap.add_argument("--agent", default=None,
                    help="trained agent, any train_rla.py --algo (default: models/agent_<mode>.zip)")
    args = ap.parse_args()
    args.agent = args.agent or f"models/agent_{args.mode}.zip"

    # Create the run directory early so the sim_to_real CSV lands inside it.
    dt_str  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(RESULTS_DIR, f"{dt_str}_run_{args.mode}")
    os.makedirs(run_dir, exist_ok=True)

    model = DistillModel.load(args.model)
    metric = PositionGapMetric()
    pre = default_preprocess()
    sim_csv = os.path.join(run_dir,
                           re.sub(r"\.script$", ".sim_to_real.csv", os.path.basename(args.script)))
    rec = build_dataset(model, metric, [args.script], args.robot_ip, args.loop, pre, sim_csv)
    metrics = (run_params if args.mode == "params" else run_path)(args, model, metric, rec, pre, run_dir)
    log_run_result(args, metrics, sim_csv, RESULTS_DIR, dt_str)


if __name__ == "__main__":
    main()
