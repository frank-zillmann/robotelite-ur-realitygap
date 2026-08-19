# Case 2: Reality Gap and Trajectory Optimization

URSim shows a perfect robot: commanded angle equals measured angle. A real UR
overshoots and rings down at the end of a move, and each joint behaves a little
differently. That difference is the **reality gap**.

This case learns the gap from recorded runs, then optimizes a motion *through* the
learned model: the whole chain from a spline's control points to the predicted
measured angles is one differentiable graph, so the score's gradient reaches the
trajectory directly.

## The task

The pipeline runs end to end. The work is to make its two models better, so that
the improvement it predicts survives contact with hardware:

- **`DistillModel`** (`train_distillation_model.py`) — what the robot really does.
- **`motion.movej`** (`motion.py`) — what the controller commands in the first place.

Then optimize a motion, run baseline and optimized on the robot, and compare.

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

# 2. optimize a script's moves against it; writes a servoj path next to the script
python optimize.py --script scripts/triangle.script --model models/distill.pkl

# 3. run baseline and optimized on the robot, compare
python send.py --robot-ip 127.0.0.1 --script scripts/triangle.script --loop 10 --out baseline.csv
python send.py --robot-ip 127.0.0.1 --path scripts/triangle.path --out optimized.csv

# 4. look at both, and at what the script alone implies
python analysis.py --csv baseline.csv --model models/distill.pkl
python analysis.py --csv optimized.csv --model models/distill.pkl
python analysis.py --csv data/test-1.csv --script data/test-1.script --model models/distill.pkl
```

Step 3 is the test that matters: if the drop the model predicted holds on hardware,
the model matched the robot; if not, it was missing something, which sends you back
to step 1.

## Folder contents

| File | Role |
|------|------|
| `record.py` | passive RTDE logger: stream robot state to a CSV, never moves the robot |
| `send.py` | send a URScript (or a `servoj` path) to the robot, run it, record it |
| `analysis.py` | `Recording` (shared CSV loader) + a plotly target/actual/script/model viewer |
| `common.py` | `segments` (split a recording into moves), `features`, and the torch `MoveDataset`/`loaders` |
| `motion.py` | `movej` (what the controller commands) and `bspline` (what replaces it) |
| `train_distillation_model.py` | `DistillModel` interface + `CNNModel`: predict the actual channels |
| `optimize.py` | differentiate a score through the model down to the spline's control points |
| `utils.py` | constants, UR10e kinematics (FK, Jacobian), URScript load/edit |

`scripts/` holds the URScript motions, `models/` the trained models, `data/` the
recordings.

## How it flows

```
control points, T ──► motion.bspline ──► commanded q(t)
                                              │
                       common.features (sin/cos q, qd, qdd, qddd)
                                              │
                        CNNModel ──► predicted actual q, and its uncertainty
                                              │
      loss = |actual - commanded|/base + k·sd  +  α·T/base  +  limits  +  straightness
                                              │
                                        .backward()
```

Everything after the control points is torch, so one `backward()` moves both the
shape of the trajectory and its duration. `T` is a free parameter, so nothing
fixes how long the move takes — `--alpha` sets what a percent of cycle time is
worth in percent of tracking error.

**What runs now (all of it is yours to change):**

- **`DistillModel`** (`train_distillation_model.py`): `CNNModel`, a causal
  dilated-convolution net over the last ~1 s of the commanded trajectory,
  predicting the gap `actual_q - target_q` for all six joints at once. `predict`
  returns `{"mean", "var", "var_aleatoric", "var_epistemic"}`. It is a sequence
  model because the gap is dynamic — the ring-down after a stop is invisible to any
  per-row model. `members=K` makes it a deep ensemble; the disagreement between the
  members is what tells the optimizer where the model is guessing.
- **`motion.movej`** (`motion.py`): a trapezoidal speed profile along the straight
  joint-space line, smoothed by a fixed 60 ms box. Its ceilings are identified from
  the recordings, and the binding one is usually Cartesian: the tool runs at exactly
  1.35 m/s through the middle of every recorded move. It reproduces the recorded
  `target_q` to ~22 mrad — see **Known gaps**.
- **The objective** (`optimize.py`): tracking error (with `--k` standard deviations
  of the model's own uncertainty added, so wandering into trajectories the model has
  never seen is not free), cycle time, a penalty for exceeding a joint's ceilings,
  and one for leaving the straight line. Change `error`, `penalty`, or add your own
  term — anything differentiable.

## Known gaps

- `motion.movej` reproduces the recorded `target_q` to about **22 mrad** peak on
  runs 1–5 and worse on 6–7, which is larger than the reality gap it is supposed to
  frame (~1–3 mrad). The controller is closed-source; this is an identification, not
  its code. So the *absolute* baseline number `optimize.py` prints carries that
  error — the honest comparison is to run both motions on the robot (step 3).
- The recorded `target_q` and `target_qd` are not consistent with each other:
  differentiating `target_q` gives ~2% more speed than the `target_qd` channel says.
  Everything here differentiates `target_q`, and `motion.py`'s caps are calibrated
  to match it.
- Each move is optimized on its own, starting from rest. A script whose moves blend
  into one another is not modelled that way.
- **The optimizer only wins for `--alpha` below ~0.1.** A trapezoid is time-optimal
  under an acceleration limit, and a B-spline is smooth, so it cannot match a
  `movej`'s duration within the same ceilings — it buys accuracy with time. On
  `scripts/triangle.script` it reaches −7% tracking error for +50% cycle time.
  Every move now prints its score against the `movej` it replaces, so you can see
  which way the trade went.

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
