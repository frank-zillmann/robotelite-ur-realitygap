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

# 2. convert the .script to a .path using URSim
python convert.py scripts/triangle.script --robot-ip 127.0.0.1

# 3. optimize that path against the model (logs to runs/optimize/)
python optimize.py --path scripts/triangle.path --model models/distill-ur5e.pkl --robot UR5e

# 4a. run both on URSim to verify safety (--engine batch for URSim)
python send.py scripts/triangle.path --robot-ip 127.0.0.1 --engine batch --loop 5
python send.py scripts/triangle.retime.path --robot-ip 127.0.0.1 --engine batch --loop 5

# 4b. run both on a real robot to verify the improvement holds (--engine stream for real)
python send.py scripts/triangle.path --robot-ip 192.168.1.100 --engine stream --loop 5
python send.py scripts/triangle.retime.path --robot-ip 192.168.1.100 --engine stream --loop 5

# 5. compare them: one plot, and the optimizer objective side by side
python analysis.py --csv scripts/triangle.stream.csv scripts/triangle.retime.stream.csv --path scripts/triangle.path scripts/triangle.retime.path --model models/distill-ur5e.pkl --robot UR5e
```

`tensorboard --logdir runs` shows both stages. Step 4 is the test that matters: if
the improvement the model predicted holds on hardware, the model matched the robot;
if not, it was missing something, which sends you back to step 1.

## Folder contents

| File | Role |
|------|------|
| `record.py` | passive RTDE logger, never moves the robot |
| `collect_data.py` | records a matrix of trajectories x speeds x reps |
| `send.py` | sends a `.script` or `.path` to the robot and records it |
| `convert.py` | runs a script, keeps the trajectory the controller commanded |
| `optimize.py` | differentiates a score through the model to the path |
| `train_distillation_model.py` | `DistillModel` + `CNNModel`: predict actual channels |
| `common.py` | `segments`, `features`, `MoveDataset`/`loaders` |
| `analysis.py` | `Recording` CSV loader, plotly viewer, objective table |
| `utils.py` | constants, `Robot` kinematics/ceilings, URScript load/edit |

`scripts/` the URScript motions, `models/` the trained models, `data/<arm>/`
the recordings.

## How it flows

```
URScript ──► the controller ──► recorded target_q                (convert.py)
                                       │
                        original q, learned per-edge dt ──► commanded q(t)
                                       │
                   common.features (sin/cos q, qd, qdd, qddd)
                                       │
                    CNNModel ──► predicted actual q, and its uncertainty
                                       │
 loss = cycle time + c·sqrt(mean(gap² + exploration·variance)) + smooth barriers
                                       │
                                 .backward()
```

**What runs now (all of it is yours to change):**

- **`DistillModel`**: `CNNModel`, a causal dilated-conv net over the last ~1 s of
  the trajectory, predicting `actual_q - target_q` and its uncertainty for all six
  joints. A sequence model because the gap is dynamic (ring-down after a stop).
  `members=K` deep-ensembles it; disagreement tells the optimizer where it's guessing.
- **`convert.py`**: runs the script twice, keeps the second pass (steady state,
  closes on itself, independent of the robot's starting pose).
- **`optimize.py`**: learns one time interval per recorded waypoint. An edge below
  `PAUSE_TOL` (not a real pose, just noise) shrinks freely instead of floored at
  `MIN_DT`. A log barrier (true to +infinity at each limit) keeps it inside joint
  and TCP speed/accel/position limits, its weight decaying early in the run so the
  tail is free to optimize cycle time and gap alone. Output is resampled onto the
  model's uniform 8 ms grid (a servoJ streamer only ticks at one fixed rate anyway).
- **`utils.Robot`**: swaps DH table and limits per arm. The model isn't
  interchangeable between arms — keep `--data`/`--model`/`--robot` matched.

## Known gaps

- `convert.py` needs the controller in Remote Control and *moves the robot*.
- `send.py`'s `.path` streaming: `--engine batch` (URSim) embeds the whole path
  as one program, which a real controller silently drops past ~30 KB. `--engine
  stream` (real hardware, via `ur_rtde`) fixes that but doesn't work against
  PolyScope X URSim (`RTDEControlInterface` won't connect there yet). No engine
  works on both. A hand-rolled real-time alternative was tried and dropped —
  it caused a real fault on hardware.
- Everything runs on `utils.DT`, one 8 ms grid (125 Hz) — what's recorded,
  trained on, and written to every `.path` (`optimize.py` resamples onto it).
- `analysis.py` scores gap, not the barrier (an optimization device, not a
  property of a finished trajectory), over every recorded target row.
- A log barrier pushed to gradient-descent's limits is not a hard guarantee —
  check an emitted path independently before running it on hardware.
- Joint acceleration ceilings are measured, not UR-specified; UR5e reuses UR10e's.
- The tool-speed Jacobian is refreshed each step from detached poses: the value
  is right, the gradient is the speed's alone.

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
