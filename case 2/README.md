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

## Prerequisites

```bash
pip install -r requirements.txt
```

- **URSim or a UR robot** in **Remote Control** mode at `--robot-ip` (default
  `127.0.0.1`); see `simulation environment/` for the container.
- **Recorded runs** in `data/`, one CSV per run. Use the provided ones or record
  your own — `record.py` only logs, it never moves the robot:
  ```bash
  python record.py --robot-ip <ip> --out data/test-1.csv --float-register 1 vel 2 acc
  ```

## How to run

```bash
# 1. distill the gap model from every run in data/ (logs to runs/distill/)
python train_distillation_model.py --out models/distill.pkl

# 2. run the script on the controller, keep the trajectory it commanded
python convert.py --script scripts/triangle.script --out scripts/triangle.path

# 3. optimize that path against the model (logs to runs/optimize/)
python optimize.py --path scripts/triangle.path --model models/distill.pkl

# 4. run both on the robot and compare
python send.py --path scripts/triangle.path --loop 5 --out baseline.csv
python send.py --path scripts/triangle.optimized.path --loop 5 --out optimized.csv

# 5. look at them, with the model's prediction and its uncertainty band
python analysis.py --csv baseline.csv --model models/distill.pkl
python analysis.py --csv optimized.csv --model models/distill.pkl
```

`tensorboard --logdir runs` shows both stages. Step 4 is the test that matters: if
the improvement the model predicted holds on hardware, the model matched the robot;
if not, it was missing something, which sends you back to step 1.

## Folder contents

| File | Role |
|------|------|
| `record.py` | passive RTDE logger: stream robot state to a CSV, never moves the robot |
| `send.py` | send a URScript (or a `servoj` path) to the robot, run it, record it |
| `convert.py` | run a script on the controller, keep the trajectory it commanded |
| `optimize.py` | differentiate a score through the model down to the path's parameters |
| `train_distillation_model.py` | `DistillModel` interface + `CNNModel`: predict the actual channels |
| `common.py` | `segments` (split a recording into moves), `features`, `MoveDataset`/`loaders` |
| `analysis.py` | `Recording` (shared CSV loader) + a plotly target/actual/model viewer |
| `utils.py` | constants, UR10e kinematics (FK, Jacobian), the speed/acceleration ceilings, URScript load/edit |

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
      loss = |actual - commanded|/base + k·sd + α·T/base + limits + drift
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
  speeding up the moves). The knobs are constants at the top of the file: `ALPHA`
  (time against error), `K` (uncertainty added to the error), `STAY` (how tightly
  the recorded path is held), `LIMIT`, and the spline/retiming sizes.

## Known gaps

- `convert.py` needs the controller in Remote Control and *moves the robot*.
- The RTDE recording arrives at 8.00 ms per row and the model was trained at
  7.83 ms, so `optimize.py` resamples; the duration is preserved.
- **The optimizer finds a local optimum.** On `scripts/triangle.script` it scores
  1.72 against the recorded path's 2.00 (−12% tracking error, −16% cycle time), but
  a hand-built solution that simply deletes the pauses scores 1.45 — the model says
  the pauses buy only 1.4% of error for 2.4 s of cycle time, and gradient descent
  does not get there from the reference.
- The tool-speed limit uses the Jacobian at the current trajectory, refreshed each
  step from detached poses: the value is right, the gradient is the speed's alone.

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
