# Case 2: Reality Gap and Trajectory Optimization

URSim shows a perfect robot: commanded angle equals measured angle. A real UR
overshoots and rings down at the end of a move, and each joint behaves a little
differently. That difference is the **reality gap**.

This case learns the gap from recorded runs, then optimizes a motion *through* the
learned model: the whole chain from the path's parameters to the predicted measured
angles is one differentiable graph, so the score's gradient reaches the trajectory
directly.

## The task

The one model in the pipeline is **`DistillModel`** (`train_distillation_model.py`):
what the robot really does with a commanded trajectory. Make it better, optimize a
motion against it, run baseline and optimized on the robot, and see whether the
predicted improvement survives. What the *controller* does with a script is not
modelled — `convert.py` records it from the controller.

There are two solvers over that one model, minimizing the identical objective so
their numbers are comparable:

- **`optimize.py`** — differentiates the score down to a spline's control points.
  Exact, and it re-solves from scratch for every path.
- **`rl_optimize.py`** — a PPO policy trained across many paths, so an unseen one
  costs a forward pass (~6 ms) instead of a solve (~10 s). It reparameterizes the
  problem onto the `movable` blocks, which is where the gain actually is.

## Prerequisites

```bash
pip install -r requirements.txt
```

- **URSim or a UR robot** in **Remote Control** mode, reachable at `--robot-ip`.
  The container is in `simulation environment/`; pick the arm when you start it, and
  pass the same one to `optimize.py`:
  ```bash
  docker compose up -d                     # UR10, the default
  ROBOT_TYPE=UR5 docker compose up -d      # a UR5e instead
  ```
- **Recorded runs**, one folder per arm: `data/ur5e` (115 runs, 11 trajectories at
  three speeds, on real hardware) and `data/ur10e` (7 runs). `data/ur5e/heldout/` is
  set aside on purpose and never trained on. Record more with `collect_data.py`, or
  a single run with `record.py`, which only logs and never moves the robot:
  ```bash
  python collect_data.py --robot-ip <ip> --list        # preview the matrix
  python record.py --robot-ip <ip> --out data/ur5e/my-run.csv
  ```

## How to run

```bash
# 1. distill the gap model from one arm's recordings (logs to runs/distill/)
python train_distillation_model.py --data data/ur5e --out models/distill-ur5e.pkl

# 2. run the script on the controller, keep the trajectory it commanded
python convert.py --script scripts/triangle.script --robot-ip 127.0.0.1 --out scripts/triangle.path

# 3. optimize that path against the model (logs to runs/optimize/)
python optimize.py --path scripts/triangle.path --model models/distill-ur5e.pkl --robot UR5e

# 4. run both on the robot and compare
python send.py --robot-ip 127.0.0.1 --path scripts/triangle.path --loop 5 --out baseline.csv
python send.py --robot-ip 127.0.0.1 --path scripts/triangle.optimized.path --loop 5 --out optimized.csv

# 5. compare them: one plot, and the measured error / cycle time side by side
python analysis.py --csv baseline.csv optimized.csv --model models/distill-ur5e.pkl
```

`tensorboard --logdir runs` shows both stages. Step 4 is the test that matters: if
the improvement the model predicted holds on hardware, the model matched the robot;
if not, it was missing something, which sends you back to step 1.

### The learned solver

The RL lane needs **no robot at all** until the last step: `paths.py` rebuilds every
recording's commanded trajectory offline, which is the same signal `convert.py` gets
by running the script.

```bash
# a. every recording as a closed path, plus its baselines: the training bank
python paths.py --data data/ur5e --model models/distill-ur5e.pkl --robot UR5e \
    --out models/bank-ur5e.pkl

# b. is there anything here worth training for? (~5 min, no training)
python rl_optimize.py preflight --bank models/bank-ur5e.pkl \
    --model models/distill-ur5e.pkl --robot UR5e

# c. clone a search, then improve on it with PPO (~25 min; the clone is cached)
python rl_optimize.py train --bank models/bank-ur5e.pkl --model models/distill-ur5e.pkl \
    --robot UR5e --rl-points 0 --steps 500000 --bc 150 --agent models/ppo-ur5e.zip

# d. score it on paths it never saw
python paths.py --data data/ur5e/heldout --model models/distill-ur5e.pkl --robot UR5e \
    --held-out --no-sub-paths --out models/bank-ur5e-heldout.pkl
python rl_optimize.py eval --bank models/bank-ur5e-heldout.pkl \
    --model models/distill-ur5e.pkl --robot UR5e --rl-points 0 --agent models/ppo-ur5e.zip

# e. write the optimized motions, as URScript and as a servoj path
python rl_optimize.py apply --bank models/bank-ur5e-heldout.pkl \
    --model models/distill-ur5e.pkl --robot UR5e --agent models/ppo-ur5e.zip \
    --out-dir optimized

# f. run every baseline/optimized pair on the robot, one command
python run_tests.py --robot-ip <ip>
python hw_check.py --robot-ip <ip>      # if a program will not start
```

Step **b** is a gate, not a formality: it measures how much of each cycle is dead
time, sweeps each action coordinate, and runs a CEM search as the ceiling any
black-box optimizer could reach. If CEM cannot beat the gradient lane there is
nothing for a policy to amortize, and it says so before you spend the training run.

## Folder contents

| File | Role |
|------|------|
| `record.py` | passive RTDE logger: stream robot state to a CSV, never moves the robot |
| `collect_data.py` | record a whole matrix of trajectories x speeds x reps, with a manifest |
| `send.py` | send a URScript (or a `servoj` path) to the robot, run it, record it |
| `convert.py` | run a script on the controller, keep the trajectory it commanded; writes a `.path` or a retimed `.script` |
| `optimize.py` | differentiate a score through the model down to the path's parameters; also the block-aligned basis both solvers share |
| `paths.py` | rebuild every recording's commanded path offline — no robot — as the RL training bank |
| `rl_optimize.py` | a PPO policy over the same objective `optimize.py` differentiates |
| `run_tests.py` | run every baseline/optimized pair on the robot and report what changed |
| `hw_check.py` | why a program will not start, using one that moves nothing |
| `train_distillation_model.py` | `DistillModel` interface + `CNNModel`: predict the actual channels |
| `common.py` | `segments` (split a recording into moves), `features`, `MoveDataset`/`loaders` |
| `analysis.py` | `Recording` (shared CSV loader) + a plotly viewer and the measured error/cycle-time table |
| `utils.py` | constants, `Robot` (UR10e/UR5e kinematics + the controller's ceilings), URScript load/edit |

`scripts/` holds the URScript motions (`_generated/` the speed variants
`collect_data.py` writes), `models/` the trained models, `data/<arm>/` the
recordings.

## How it flows

```
URScript ──► the controller ──► recorded target_q                (convert.py)
                                       │
    offset, retiming ──► reference path + B-spline offset ──► commanded q(t)
                                       │
                   common.features (sin/cos q, qd, qdd, qddd)
                                       │
                    CNNModel ──► predicted actual q, and its uncertainty
                                       │
      loss = |actual - commanded|/base + k·sd + α·T/base + limits
                                       │
                                 .backward()
```

Both parameter groups start at zero, which reproduces the recorded path *exactly* —
so the optimizer can leave it alone if that is already best, and every number is
reported against it.

**What runs now (all of it is yours to change):**

- **`DistillModel`**: `CNNModel`, a causal dilated-convolution net over the last
  ~1 s of the commanded trajectory, predicting the gap `actual_q - target_q` for all
  six joints at once. `predict` returns `{"mean", "var", "var_aleatoric",
  "var_epistemic"}`. It is a sequence model because the gap is dynamic — the
  ring-down after a stop is invisible to any per-row model. `members=K` makes it a
  deep ensemble, and the members' disagreement is what tells the optimizer where it
  is guessing.
- **`convert.py`**: runs the script twice and keeps the second pass, which starts
  where the first ended — the cycle in its steady state, closing on itself so it can
  be looped, and independent of where the robot happened to be.
- **`optimize.py`**: a B-spline offset added to the reference, plus a retiming that
  gives each slice of the path its own duration (so a pause can be cut without
  speeding up the moves). The offset is masked to zero wherever the reference stands
  still, so the waypoints the script holds are kept exactly while the moves between
  them are free — a constraint by construction rather than another penalty to weigh.
  The knobs are constants at the top of the file: `ALPHA` (time against error), `K`
  (uncertainty added to the error), `LIMIT`, and the spline/retiming sizes.
- **`utils.Robot`**: `Robot("UR5e")` swaps the DH table and the joint speed limits.
  The distilled model is *not* interchangeable — it is trained on one arm's
  recordings, so each arm has its own folder and its own pickle
  (`models/distill-ur5e.pkl`, `models/distill-ur10e.pkl`). Keep `--data`, `--model`
  and `--robot` pointing at the same arm.

## Known gaps

- `convert.py` needs the controller in Remote Control and *moves the robot*.
- Everything runs on `utils.DT`, one 8.00 ms grid (125 Hz): what `record.py` asks
  the stream for, what the model is trained on, and what a path is written at. 8 ms
  divides both control cycles UR ships (2 ms e-Series, 8 ms CB3). `MoveDataset`
  refuses a recording made at another rate. A path's `dt` column may still differ per
  row and is honoured from at least 1 ms to 50 ms, but below the controller's 2 ms
  cycle it cannot act on each setpoint separately.
- **The two solvers want different things from the limit penalty.** `penalty` scores
  the mean overshoot *and* a fourth-power norm of it, because the mean alone dilutes a
  brief excursion into noise — a path a tool-speed limit over the cap for a tenth of a
  second scored 9e-5, which at any weight is worth buying. But a barrier stiff enough
  to resist half a million policy evaluations is hostile to gradient descent
  approaching it from inside: measured, the gradient lane degrades monotonically with
  `LIMIT` and finds nothing at all above ~100, while CEM returns the same feasible
  answer anywhere from 30 to 1000. `LIMIT = 30` is where both still work. Note the
  ceilings are the controller's *spec* values; `Robot.MARGIN` exists so a differenced
  recording is not misread as a violation, and handing it to an optimizer would just
  convert tolerance into 2% more speed the controller then clamps.
- The joint acceleration ceilings are measured, not specified — UR publishes none —
  and the UR5e reuses the UR10e's for want of anything better.
- **The gradient lane finds a local optimum, and the basis is why.** A recorded cycle
  spends 25–43% of itself standing still on `sleep`, but `phase`'s 24 uniform slices
  straddle moves and pauses — a pause spans anywhere from 0.1 to 4.8 of them — so
  nothing can shorten one without stretching the move beside it. `optimize.bounds`
  reparameterizes onto the `movable` blocks instead (4–9 of them, each wholly a move
  or wholly a pause), and a search in that basis reaches a median 0.50 against the
  gradient lane's 0.12 on the same objective. That gap is what the RL lane exists for.
- **The policy is conservative, deliberately.** It reaches ~79% of what a per-path CEM
  search finds. Cloning the search alone scores *higher* on the median (0.40 vs 0.31)
  but produced motions the controller would refuse on 5 of 9 paths — one commanding
  1.5× the joint-speed limit and 3.9× the tool-speed cap. PPO trades a fifth of the
  score for solutions that are feasible every time, which is the right trade when an
  infeasible path is worth nothing.
- **The action is a fixed-width vector, so block count is a coverage question.**
  Coordinate *k* drives block *k*. Trained on paths of 2–7 blocks, a policy has never
  exercised the coordinates an 8-block path uses and emits an untrained answer for
  them — measured, a limit violation of 1.9 on exactly those paths. `paths.repeat`
  fills the range by running each closed cycle twice, which is a real trajectory
  (`send.py --loop 2`), not a synthetic one. Sub-paths cannot do it: every one has
  *fewer* blocks than its parent.
- **The gap model is ~22% optimistic on runs it never saw.** Against the held-out
  recordings it predicts 0.076 mrad where 0.097 was measured — a consistent factor
  (0.66–0.85, no outliers) rather than noise, so the *ratio* of optimized to baseline
  survives it better than the absolute does. Its uncertainty is honest: 70% of rows
  fall inside the predicted 1-sd band against 0.68 for perfect calibration.
- The tool-speed limit uses the Jacobian at the current trajectory, refreshed each
  step from detached poses: the value is right, the gradient is the speed's alone.

## Tiers

- **Bronze, understand it:** run the pipeline end to end, use `analysis.py` to see the
  reality gap on a recording, and read `optimize.loss` — decide for yourself what
  `ALPHA` should be for your arm, given how big its gap actually is.
- **Silver, build the model:** write your own `DistillModel`, choose the features and
  architecture, and beat `CNNModel` on held-out runs. Score it the way `Known gaps`
  does: predicted against measured on recordings it never trained on, and whether its
  uncertainty band is calibrated.
- **Gold, optimize it:** improve either solver against the same objective —
  `optimize.py`'s `error`/`penalty`/spline for the gradient lane, or
  `rl_optimize.py`'s observation, action basis and reward for the policy. Beat the
  recorded path on motions the optimizer never saw, with every result feasible.
- **Diamond, push to real:** transfer to hardware, refit the `DistillModel` on the new
  recordings, and close the loop until the arm measurably improves. `run_tests.py`
  runs every baseline/optimized pair in one command; `hw_check.py` says why a program
  will not start. A simulator cannot finish this tier — there the measured angle *is*
  the commanded one, so it can confirm the cycle time and the controller's acceptance
  of a path, and nothing about the gap.
