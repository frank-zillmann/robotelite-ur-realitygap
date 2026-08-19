# Case 2: Reality Gap and Trajectory Optimization

URSim shows a perfect robot: commanded angle equals measured angle. A real UR
overshoots and rings down at the end of a move, and each joint behaves a little
differently. That difference is the **reality gap**.

This case learns the gap from recorded runs, then optimizes a motion *through* the
learned model: the whole chain from a spline's control points to the predicted
measured angles is one differentiable graph, so the score's gradient reaches the
trajectory directly.

## The task

The pipeline runs end to end. The one model in it is **`DistillModel`**
(`train_distillation_model.py`): what the robot really does with a commanded
trajectory. Make it better, then optimize a motion against it, run baseline and
optimized on the robot, and see whether the predicted improvement survives.

What the *controller* does with a script is not modelled at all — `convert.py`
records it from the controller.

## Prerequisites

```bash
pip install -r requirements.txt
```

- **URSim or a UR robot** in **Remote Control** mode, reachable at `--robot-ip`
  (default `127.0.0.1`). Without Remote Control the controller accepts the socket
  but does not run the script. See `simulation environment/` for the container.
- **Recorded runs** in `data/`, one CSV per run. Use the provided runs or record
  your own:
  ```bash
  python record.py --robot-ip <ip> --out data/test-1.csv --float-register 1 vel 2 acc
  ```
  `record.py` reads the controller's RTDE stream and only logs; it never moves the
  robot. RTDE channels: <https://www.universal-robots.com/developer/communication-protocol/rtde/>.

## How to run

```bash
# 1. distill the gap model from every run in data/ (losses, errors and calibration
#    go to runs/, watch them with `tensorboard --logdir runs`)
python train_distillation_model.py --out models/distill.pkl

# 2. run the script on the controller and keep the trajectory it commanded
python convert.py --script scripts/triangle.script --out scripts/triangle.path

# 3. optimize that path against the model
python optimize.py --path scripts/triangle.path --model models/distill.pkl

# 4. run both on the robot and compare
python send.py --robot-ip 127.0.0.1 --path scripts/triangle.path --out baseline.csv
python send.py --robot-ip 127.0.0.1 --path scripts/triangle.optimized.path --out optimized.csv

# 5. look at both
python analysis.py --csv baseline.csv --model models/distill.pkl
python analysis.py --csv optimized.csv --model models/distill.pkl
```

Step 4 is the test that matters: if the drop the model predicted holds on hardware,
the model matched the robot; if not, it was missing something, which sends you back
to step 1.

## Folder contents

| File | Role |
|------|------|
| `record.py` | passive RTDE logger: stream robot state to a CSV, never moves the robot |
| `send.py` | send a URScript (or a `servoj` path) to the robot, run it, record it |
| `analysis.py` | `Recording` (shared CSV loader) + a plotly target/actual/script/model viewer |
| `common.py` | `segments` (split a recording into moves), `features`, and the torch `MoveDataset`/`loaders` |
| `convert.py` | run a script on the controller, keep the trajectory it commanded, as a path |
| `motion.py` | the differentiable B-spline, and the speed/acceleration envelope |
| `train_distillation_model.py` | `DistillModel` interface + `CNNModel`: predict the actual channels |
| `optimize.py` | differentiate a score through the model down to the path's parameters |
| `utils.py` | constants, UR10e kinematics (FK, Jacobian), URScript load/edit |

`scripts/` holds the URScript motions, `models/` the trained models, `data/` the
recordings.

## How it flows

```
URScript ──► the controller ──► recorded target_q          (convert.py)
                                      │
   offset, retiming, T ──► reference path + B-spline offset ──► commanded q(t)
                                      │
                    common.features (sin/cos q, qd, qdd, qddd)
                                      │
                     CNNModel ──► predicted actual q, and its uncertainty
                                      │
     loss = |actual - commanded|/base + k·sd + α·T/base + limits + drift
                                      │
                                .backward()
```

Everything after the parameters is torch, so one `backward()` moves the shape of
the trajectory, where its time goes, and how long it takes. All three parameter
groups start at zero, which reproduces the recorded path *exactly* — so the
optimizer can leave it alone if that is already best, and every number is reported
against it.

**What runs now (all of it is yours to change):**

- **`DistillModel`** (`train_distillation_model.py`): `CNNModel`, a causal
  dilated-convolution net over the last ~1 s of the commanded trajectory,
  predicting the gap `actual_q - target_q` for all six joints at once. `predict`
  returns `{"mean", "var", "var_aleatoric", "var_epistemic"}`. It is a sequence
  model because the gap is dynamic — the ring-down after a stop is invisible to any
  per-row model. `members=K` makes it a deep ensemble; the disagreement between the
  members is what tells the optimizer where the model is guessing.
- **`convert.py`**: runs the script on the controller and keeps `target_q`. Pauses,
  blends, speed profiles and the Cartesian caps come out exactly right because the
  controller produced them. The first move is dropped — it is the robot travelling
  to the script's first waypoint, not part of the cycle.
- **The parameters** (`optimize.py`): a B-spline offset added to the reference, a
  monotone retiming that can shrink a pause without slowing a move, and the cycle
  time. Where the reference stands still it is holding a waypoint, so the path is
  held ten times harder there — the corners stay put while the moves are free.
- **The objective** (`optimize.py`): tracking error (with `--k` standard deviations
  of the model's own uncertainty added, so wandering into trajectories the model has
  never seen is not free), cycle time, a penalty for exceeding a joint's ceilings,
  and one for leaving the straight line. Change `error`, `penalty`, or add your own
  term — anything differentiable.

## Known gaps

- `convert.py` needs the controller running and in Remote Control, and it *moves the
  robot* to record it. On URSim that is free; on hardware, watch the cell.
- The RTDE recording comes out at ~8.25 ms per row while the model was trained at
  7.83 ms, so `optimize.py` resamples the reference onto the model's grid. The
  duration is preserved, only the spacing changes.
- The tool-speed limit is checked with the Jacobian at the current trajectory,
  refreshed each step from detached poses: the value is right, the gradient is the
  speed's alone. Trajectories that stray far from the reference are still worth
  checking against the printed peak tool speed.
- The optimizer keeps the best iterate it saw rather than the last one — Adam
  wanders near the optimum, and the reported numbers are of the path that is
  actually written out.
- `--k` adds the model's own uncertainty to the error, which is what stops the
  optimizer from wandering into trajectories the model has never seen. It is not a
  substitute for the explicit limit penalty: outside the envelope the model is
  extrapolating, and a confident wrong answer there costs nothing.

## Tiers

- **Bronze, understand it:** run the pipeline end to end, use `analysis.py` to see
  the reality gap, explore the `Preprocess` step, and decide what `EvaluationMetric`
  should measure. Record more runs with `record.py` if you like.
- **Silver, build the model:** write your own `DistillModel`, choose the features
  and architecture, and beat `CNNModel` on held-out runs.
- **Gold, optimize it:** improve the RL agent (observation, `OBJECTIVE`, reward) and
  the `Dynamics` torque model (friction, Coriolis, identified parameters), and beat
  a fixed baseline's score.
- **Diamond, push to real:** shape the servoj path (path mode) and transfer to a
  real UR10e, refit the `DistillModel`/`Dynamics` on the real recordings, and close
  the sim-to-real loop until the robot measurably improves.
