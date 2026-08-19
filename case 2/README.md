# Case 2: Reality Gap and Motion Optimization

URSim shows a perfect robot: commanded angle equals measured angle. A real UR
overshoots and rings down at the end of a move, and each joint behaves a little
differently. That difference is the **reality gap**.

This case learns the gap from recorded runs, uses the learned model as a fast
stand-in for the real robot, optimizes a motion against it, and runs the result on
the robot to check whether the gain holds.

## The task

The pipeline runs end to end with baseline implementations. The work is to replace
them with better ones so the agent's improvement transfers to hardware. The swap
points are three models, `DistillModel`, `Dynamics`, `EvaluationMetric`, plus the
`Preprocess` helper; each is an interface with a working default (what each
does now and where to change it is in **What runs now** below). Then train the RL
agent, optimize a held-out motion, and compare baseline vs optimized on the robot.

## Prerequisites

```bash
pip install -r requirements.txt
```

- **URSim or a UR robot** in **Remote Control** mode, reachable at `--robot-ip`
  (default `127.0.0.1`). Without Remote Control the controller accepts the socket
  but does not run the script.
- **Recorded runs** in `data/`, one CSV per run. Collect your own with
  `collect.py` (see below), or log a program you drive from PolyScope:
  ```bash
  python record.py --robot-ip <ip> --out data/test-1.csv --float-register 1 vel 2 acc
  ```
  `record.py` reads the controller's RTDE stream and only logs; it never moves the
  robot. RTDE channels: <https://www.universal-robots.com/developer/communication-protocol/rtde/>.
- `gymnasium` and `stable-baselines3` (in `requirements.txt`) for training.

## Collecting data: units and coverage

**`vel` and `acc` are rad/s and rad/s², not deg.** A URScript `movej` takes `v` in
rad/s, and the controller silently clamps anything above the joint limit —
2.09 rad/s (120 °/s) on base/shoulder/elbow, 3.14 rad/s on the wrists.

The `data/test-*.csv` recordings shipped with the case were taken with programs
that sweep `vel` from 100 down to 20 and `acc` likewise. Every one of those numbers
is an order of magnitude above the limit, so **every run was recorded at maximum
speed**: `target_qd` peaks at exactly 2.094 / 3.142 rad/s in all of them while the
logged `vel` register reads 20, 50, 100. Check it yourself:

```bash
python -c "import pandas as pd,numpy as np; d=pd.read_csv('data/test-4.csv'); \
print(sorted(d.vel.unique())); print(np.abs(d[[f'target_qd{j}' for j in range(6)]]).max().values)"
```

A gap model fitted on that sees a `vel` column that varies and a trajectory that
does not, so the only thing it can learn is that speed has no effect — and the RL
agent then has nothing to optimize. The same two runs also barely move joints 3–5
and keep the arm in one configuration, so the model never sees the inertia and
gravity variation that the gap actually depends on.

`collect.py` fixes both. It sweeps `vel`/`acc` over values below the limit and runs
each combination on four coverage motions:

```bash
python collect.py --dry-run                    # the plan and a time estimate, no robot
python collect.py --robot-ip 127.0.0.1         # ~30 min, 60 runs -> data/sweep/
```

**Point it at hardware, not URSim, for the distillation data.** URSim has no
reality gap to record: it writes `actual_q == target_q` and
`actual_current == target_current` bit-for-bit, so a `DistillModel` fitted on a
URSim sweep learns the identity function. Confirm it on any URSim recording:

```bash
python -c "import pandas as pd,numpy as np; d=pd.read_csv('data/sweep/pooled.csv'); \
c=lambda b:[f'{b}{j}' for j in range(6)]; \
print(np.array_equal(d[c('actual_current')].values, d[c('target_current')].values))"
```

Running the sweep against URSim is still worth doing first: it is a dress
rehearsal that proves the motions, the units and the labelling before you spend
time on the robot. But the CSVs it produces belong to the *commanded* side of the
pipeline (what `build_dataset` collects for the RL env), not the training set for
the gap.

| Motion | What it covers |
|--------|----------------|
| `scripts/workspace_sweep.script` | 3×3 grid of TCP reach (0.35–1.05 m) × height (0.2–1.15 m), all six joints, gravity 29–94 Nm |
| `scripts/reach_extend.script` | extended ↔ folded arm: joint-space inertia `M[0,0]` swings ~6×, gravity 31–103 Nm |
| `scripts/wrist_sweep.script` | wrists 3/4/5 only, 1.1–4.4 rad, while the arm hangs extended |
| `scripts/short_moves.script` | 0.08–0.42 rad hops: the profile stays triangular, so `acc` is separable from `vel` |

One CSV per `(script, vel, acc)` lands in `--out-dir`, plus a pooled CSV tagged by
run so `common.segments` never runs a segment across two recordings. Each run is
checked as it finishes: a `CLAMPED` line means the commanded speed hit the joint
limit and that run is another copy of the max-speed data. Existing CSVs are skipped
unless `--overwrite`, so an interrupted sweep resumes.

Then distill on the sweep recorded **on the robot**, rather than the old runs:

```bash
python collect.py --robot-ip <real-robot-ip> --out-dir data/real_sweep
python train_distillation_model.py --csvs data/real_sweep/pooled.csv --out models/distill.pkl
```

## How to run

Distill once, then either mode reuses the model.

```bash
# 0. collect runs that actually vary in speed and cover the workspace.
#    ON THE ROBOT: URSim reports actual_* == target_*, so a sweep recorded there
#    has no gap in it. Against URSim this is a rehearsal of the motions only.
python collect.py --robot-ip <real-robot-ip> --out-dir data/real_sweep

# 1. distill the gap model from the recorded real runs
python train_distillation_model.py --csvs data/real_sweep/pooled.csv --out models/distill.pkl

# 2. train the RL agent on several scripts (--loop repeats each for more moves)
python train_rla.py --mode params --robot-ip 127.0.0.1 --loop 5 --model models/distill.pkl --steps 20000 \
    --scripts scripts/shoulder_swing.script scripts/vertical_swing.script scripts/horizontal_swing.script

# 3. optimize a held-out script the agent did not train on
python run.py --mode params --script scripts/triangle.script --model models/distill.pkl --robot-ip 127.0.0.1

# 4. run baseline and optimized on the robot, compare
python send.py --robot-ip 127.0.0.1 --script scripts/triangle.script --loop 10 --out baseline.csv
python send.py --robot-ip 127.0.0.1 --script scripts/triangle.optimized.script --loop 10 --out optimized.csv

# 5. run the analysis scripts to compare your results
python analysis.py --csv baseline.csv --joint 0
python analysis.py --csv optimized.csv --joint 0
# note: you might notice something is off. Is the pipeline not finished?
```

For **path** mode, use `--mode path` in steps 2 and 3, then stream the result:

```bash
python send.py --robot-ip 127.0.0.1 --path scripts/triangle.path --out optimized.csv
```

Step 3 tests transfer: if the drop the model predicted holds on hardware, the model
matched the robot; if not, it was missing something, which sends you back to the
distillation.

## Folder contents

| File | Role |
|------|------|
| `record.py` | passive RTDE logger: stream robot state to a CSV, never moves the robot |
| `collect.py` | sweep `vel`/`acc` over the coverage motions and record one CSV per run |
| `send.py` | send a URScript (or a `servoj` path) to the robot, run it, record it |
| `analysis.py` | `Recording` (shared CSV loader) + per-joint stats and a current plot |
| `common.py` | `segments`: split a recording into waypoint-to-waypoint moves |
| `dynamics.py` | `Dynamics` interface + `UR5eDynamics`: candidate target torque/current |
| `train_distillation_model.py` | `DistillModel` interface + `LinearModel`: predict the actual channels |
| `metrics.py` | `EvaluationMetric` interface + `CurrentGapMetric`: the per-row `score` to minimize |
| `preprocess.py` | `Preprocess` interface: reshape data into and out of the learners |
| `train_rla.py` | Gym envs over the models; trains a PPO agent (`GapEnv` params, `PathEnv` path) |
| `run.py` | ask the trained agent for a better motion, write the optimized script or path |
| `utils.py` | constants, UR5e physics (FK, Jacobian, gravity, mass matrix, Coriolis), URScript load/edit |

`scripts/` holds the URScript motions, `models/` the trained models, `data/` the
recordings.

**How it flows.** A recording is a CSV of per-joint channels over time (commanded
`target_*`, measured `actual_*`, `vel`/`acc`, the running URScript line);
`analysis.Recording` loads it, `common.segments` splits it into moves (one per
movej). `build_dataset` runs the scripts on URSim, has `DistillModel` fill the
`actual_*` columns and `EvaluationMetric` add a `score` column, into
`sim_to_real.csv`. The PPO agent proposes an action per move; to score it,
`Dynamics` builds the candidate's commanded trajectory, `DistillModel` predicts the
actuals, `EvaluationMetric` scores them.

**What runs now (all of it is yours to change):**

- **`DistillModel`** (`train_distillation_model.py`): a per-row least-squares
  `LinearModel` predicting `actual_current` from `[target_current, qd, qdd, pos,
  vel, acc, joint]`. Change the features (`_row_features`), the predicted channel
  (`predicts`), or the whole model.
- **`Dynamics`** (`dynamics.py`): `UR5eDynamics`, `tau = M(q)qdd + g(q)`,
  `current = tau/Kt` (Coriolis dropped); rad throughout, speed clamped to
  `MAX_JOINT_SPEED` as the controller does. Override `current(q, qd, qdd)` for
  friction, Coriolis, identified parameters.
- **`EvaluationMetric`** (`metrics.py`): `CurrentGapMetric`, `|actual_current -
  target_current|` summed over joints. Change `needs`/`per_row` for overshoot,
  jerk, a weighted mix.
- **`Preprocess`** (`preprocess.py`): `Identity` (no-op). Subclass to normalize or
  scale features into and out of the learners.
- **The RLA** (`train_rla.py`): observation = 16 numbers (the move + its distilled
  channels + baseline score); objective = `score + cycle_time` (`OBJECTIVE`).
  `params` action = `[vel, acc]` (rad/s, bounded below the joint limit by
  `VEL_BOUNDS`/`ACC_BOUNDS`), a trapezoidal speed along the movej line;
  `path` action = `[accel_frac, decel_frac, speed]`, replaying the recorded
  trajectory at a trapezoidal speed profile. Change `observe`, the `OBJECTIVE`
  weights, or the action.
- `utils.UR5e` supplies the robot physics (FK, Jacobian, gravity, mass matrix,
  Coriolis) for `Dynamics` and as `DistillModel` features.

## Tiers

- **Bronze, understand it:** run the pipeline end to end, use `analysis.py` to see
  the reality gap, explore the `Preprocess` step, and decide what `EvaluationMetric`
  should measure. Record more runs with `record.py` if you like.
- **Silver, build the model:** write your own `DistillModel`, choose the features
  and architecture, and beat the linear baseline on held-out runs.
- **Gold, optimize it:** improve the RL agent (observation, `OBJECTIVE`, reward) and
  the `Dynamics` torque model (friction, Coriolis, identified parameters), and beat
  a fixed baseline's score.
- **Diamond, push to real:** shape the servoj path (path mode) and transfer to a
  real UR5e, refit the `DistillModel`/`Dynamics` on the real recordings, and close
  the sim-to-real loop until the robot measurably improves.
