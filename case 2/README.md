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
  +
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

# 2. get the trajectory the controller commands: run the script, or lift it
#    straight out of a recording you already have
python convert.py scripts/triangle.script --robot-ip 127.0.0.1
python convert.py data/ur5e/heldout/T10_medium_r1.csv --out scripts/T10_medium.path

# 3. optimize that path against the model (logs to runs/optimize/)
python optimize.py --path scripts/triangle.path --model models/distill-ur5e.pkl --robot UR5e --mode retime
# Or reduce it to a direct, piecewise-linear route between the same endpoints.
python optimize.py --path scripts/triangle.path --model models/distill-ur5e.pkl --robot UR5e --mode reshape

# 4. run both on the robot (each records to <path name>.csv)
# --engine script on URSim, --engine ur_rtde on real hardware (see Known gaps)
python send.py scripts/triangle.path --robot-ip 127.0.0.1 --engine script --loop 5
python send.py scripts/triangle.retime.path --robot-ip 127.0.0.1 --engine script --loop 5

# 5. compare them: one plot, and the measured error / cycle time side by side
python analysis.py --csv scripts/triangle.csv scripts/triangle.retime.csv \
    --model models/distill-ur5e.pkl
```

`tensorboard --logdir runs` shows both stages. Step 4 is the test that matters: if
the improvement the model predicted holds on hardware, the model matched the robot;
if not, it was missing something, which sends you back to step 1.

## Folder contents

| File | Role |
|------|------|
| `record.py` | passive RTDE logger: stream robot state to a CSV, never moves the robot |
| `collect_data.py` | record a whole matrix of trajectories x speeds x reps, with a manifest |
| `send.py` | send a URScript (or a `servoj` path) to the robot, run it, record it |
| `convert.py` | run a script on the controller, keep the trajectory it commanded |
| `optimize.py` | differentiate a score through the model down to the path's parameters |
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
 retime: original q + new dt; reshape: direct linear waypoints ──► commanded q(t)
                                       │
                   common.features (sin/cos q, qd, qdd, qddd)
                                       │
                    CNNModel ──► predicted actual q, and its uncertainty
                                       │
 loss = cycle time + c·sqrt(mean(gap² + exploration·variance)) + smooth barriers
                                       │
                                 .backward()
```

`retime` starts from the recorded timing. `reshape` uses
`CONTROL_POINT_FACTOR` of the recorded poses, with its first and last pose pinned.

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
- **`optimize.py`**: `retime` learns one direct interval per original edge; only an
  exact stationary edge may shrink towards zero. `reshape` learns direct interior
  waypoints and intervals for a small linear path, retaining only the endpoints.
  Both are interpolated to the model's 8 ms grid and use smooth barriers for joint
  position, joint speed, acceleration, and TCP speed.
- **`utils.Robot`**: `Robot("UR5e")` swaps the DH table and the joint speed limits.
  The distilled model is *not* interchangeable — it is trained on one arm's
  recordings, so each arm has its own folder and its own pickle
  (`models/distill-ur5e.pkl`, `models/distill-ur10e.pkl`). Keep `--data`, `--model`
  and `--robot` pointing at the same arm.

## Known gaps

- `convert.py` needs the controller in Remote Control and *moves the robot*.
- `send.py`'s `.path` streaming needs `--engine script` (URSim) or
  `--engine ur_rtde` (real hardware) -- there's no single engine that works on
  both. "script" embeds the whole path as one program, like a `.script` run;
  fine on URSim, but a real controller silently drops any program over ~30 KB
  of text. "ur_rtde" streams it live via the `ur_rtde` package instead
  (verified working on real hardware); it does not work against the
  PolyScope X URSim in `simulation environment/` (`RTDEControlInterface`
  fails to connect -- that simulator image doesn't seem to implement the
  real-time control handshake yet, though `RTDEReceiveInterface` alone does
  work against it). A hand-rolled real-time protocol over raw sockets was
  tried as a single cross-target approach and dropped: Python/OS scheduling
  cannot reliably hit servoJ's timing, and it caused a fault on real hardware.
- Everything runs on `utils.DT`, one 8.00 ms grid (125 Hz): what `record.py` asks
  the stream for, what the model is trained on, and what a path is written at. 8 ms
  divides both control cycles UR ships (2 ms e-Series, 8 ms CB3). `MoveDataset`
  refuses a recording made at another rate. A path's `dt` column may still differ per
  row and is honoured from 2 ms to 50 ms, but below the controller's 2 ms
  cycle it cannot act on each setpoint separately.
- `analysis.py` reports what a run actually took, not what its path asked for. On
  URSim the two agree to within a few percent, at 8, 16 and 32 ms per setpoint.
- A smooth barrier is not a safety controller. Check the emitted path independently
  before running it on hardware; UR5e uses the 1.5 m/s and 191°/s limits here.
- The joint acceleration ceilings are measured, not specified — UR publishes none —
  and the UR5e reuses the UR10e's for want of anything better.
- `reshape` is intentionally low-dimensional, but it can still find a local optimum.
  Adjust `CONTROL_POINT_FACTOR` in `optimize.py` for a simpler or more detailed route.
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
