# Case 2 — Command Reference

All commands are run from inside `case 2/`. Results are saved automatically to `results/`
with a datetime stamp — you never need to worry about overwriting files.

---

## Pipeline order

```
1. record.py          record real robot runs  →  data/test-N.csv
2. train_distillation_model.py  learn the gap model  →  models/distill.pkl
3. train_rla.py       train the RL agent  →  models/agent_params.zip
4. run.py             optimise a held-out script  →  scripts/triangle.optimized.script
5. send.py            run scripts on robot and record  →  results/<datetime>_<name>.csv
6. analysis.py        inspect a recording  →  results/<datetime>_analysis_<name>/
```

---

## Simulator

```bash
cd "../simulation environment"
docker compose up -d          # start URSim (~40 s first boot)
docker compose down           # stop it
```

Open http://localhost, power the robot on, release the brakes until it reads RUNNING.
URSim must be in **Remote Control** mode for scripts to actually execute.

---

## 1. record.py — passive RTDE logger

Records the robot's state stream to a CSV. Never moves the robot; the robot must
already be running a script (sent separately).

```bash
# Basic recording (registers 1=vel, 2=acc are needed by the distillation pipeline)
python record.py --robot-ip 127.0.0.1 \
                 --out data/test-1.csv \
                 --float-register 1 vel 2 acc

# Different robot IP
python record.py --robot-ip 192.168.1.100 --out data/test-1.csv --float-register 1 vel 2 acc
```

**Output:** `data/test-N.csv` — raw recording (target_*, actual_*, vel, acc columns).

---

## 2. train_distillation_model.py — learn the reality gap

Trains a model that predicts `actual_current` from the commanded trajectory.
Must be trained on real robot recordings (not URSim — URSim has zero gap).

Uses a **fixed file-level train/test split by default**: `fit()` only sees
`data/test-1,2,3,6.csv`; every printed/plotted/logged metric comes from
predicting `data/test-4,5,7.csv`, which the model never trains on. This split
is the same no matter which `--model` or `Preprocess` (`preprocess.py`) is
active, so numbers stay comparable across runs. The saved model is fit on the
training files only — it is *not* refit on the test files afterward.

```bash
# Default split (train 1,2,3,6 / test 4,5,7), save model
python train_distillation_model.py --out models/distill.pkl

# Override the split explicitly (must not overlap)
python train_distillation_model.py \
    --train-csvs data/test-1.csv data/test-2.csv \
    --test-csvs data/test-3.csv \
    --out models/distill.pkl

# Select a different registered DistillModel (MODELS dict in the file)
python train_distillation_model.py --model linear --out models/distill.pkl
```

**Output (automatic, no flags needed):**
```
models/distill.pkl                         ← latest model (overwritten each run)
results/<datetime>/
  distill.pkl                              ← versioned copy
  log.json                                 ← features, coefficients, held-out + in-sample RMSE/R²
  residuals_actual_current.png             ← error distribution (held-out files)
  per_joint_rmse_actual_current.png        ← per-joint RMSE bar chart (held-out files)
results/runs_summary.csv                   ← one row per run for comparison
results/comparison_plot.png                ← RMSE / R² trend across all runs
```

---

## 3. train_rla.py — train the RL agent

Runs the training scripts on URSim, builds the dataset, then trains a PPO agent
to minimise score (current gap) while keeping cycle time short.

```bash
# params mode: agent picks vel/acc per move (default, recommended to start)
python train_rla.py \
    --mode params \
    --model models/distill.pkl \
    --robot-ip 127.0.0.1 \
    --scripts scripts/shoulder_swing.script scripts/vertical_swing.script scripts/horizontal_swing.script \
    --loop 5 \
    --steps 20000

# path mode: agent shapes the speed profile along the recorded path
python train_rla.py \
    --mode path \
    --model models/distill.pkl \
    --robot-ip 127.0.0.1 \
    --scripts scripts/shoulder_swing.script scripts/vertical_swing.script \
    --loop 5 \
    --steps 20000

# More training steps for a better agent
python train_rla.py --mode params --model models/distill.pkl \
    --scripts scripts/shoulder_swing.script scripts/vertical_swing.script \
    --steps 100000
```

**Key flags:**
- `--scripts` — training motions; keep triangle out (it is the test script)
- `--loop N` — repeat each script N times for more diverse moves
- `--steps` — PPO timesteps; more = better policy but slower

**Output (automatic):**
```
models/agent_params.zip                    ← latest agent (overwritten each run)
results/<datetime>_rla_params/
  agent_params.zip                         ← versioned copy
  sim_to_real.csv                          ← training dataset
  log.json                                 ← mode, steps, best/final score
  training_curve.png                       ← score / cycle time / reward over training
results/runs_summary_rla.csv               ← one row per run for comparison
```

---

## 4. run.py — optimise a held-out script

Runs the triangle script on URSim once, asks the trained agent for better
motion parameters, scores the improvement, and writes the optimised script.

```bash
# params mode
python run.py \
    --mode params \
    --script scripts/triangle.script \
    --model models/distill.pkl \
    --robot-ip 127.0.0.1

# path mode
python run.py \
    --mode path \
    --script scripts/triangle.script \
    --model models/distill.pkl \
    --robot-ip 127.0.0.1

# Use a specific versioned model + agent (not just latest)
python run.py \
    --mode params \
    --script scripts/triangle.script \
    --model results/2026-08-10_14-32-05/distill.pkl \
    --agent results/2026-08-10_15-10-22_rla_params/agent_params.zip \
    --robot-ip 127.0.0.1
```

**Output (automatic):**
```
scripts/triangle.optimized.script          ← rewritten script with new vel/acc
scripts/triangle.path                      ← (path mode only) dense servoj setpoints
results/<datetime>_run_params/
  triangle.sim_to_real.csv
  log.json                                 ← script, model, agent, baseline vs optimised metrics
```

Terminal prints predicted improvement, e.g.:
```
predicted mean score: 0.847 -> 0.612 (-27.7%)
predicted cycle time: 1.230 -> 0.891s (-27.6%)
```

---

## 5. send.py — run a script on the robot and record it

Sends a URScript (or servoj path) to the robot and logs every sample to a CSV.
This is the only step that puts real motion on hardware.

```bash
# Run the original script (auto-named output)
python send.py --script scripts/triangle.script --loop 10

# Run the optimised script (auto-named output)
python send.py --script scripts/triangle.optimized.script --loop 10

# Explicit output name
python send.py --script scripts/triangle.script --loop 10 --out baseline.csv

# Path mode (servoj stream)
python send.py --path scripts/triangle.path --loop 10

# Real robot IP
python send.py --robot-ip 192.168.1.100 --script scripts/triangle.script --loop 10

# Change sample rate (default 125 Hz)
python send.py --script scripts/triangle.script --hz 500
```

**Key flags:**
- `--loop N` — repeat the motion N times (average out noise; 10 is typical)
- `--out` — explicit output path; if omitted, auto-generates `results/<datetime>_<script>.csv`
- `--robot-ip` — defaults to 127.0.0.1 (URSim); change for real hardware

**Output (automatic):**
```
results/<datetime>_triangle.csv            ← recording (if --out not given)
results/<datetime>_triangle.csv.json       ← sidecar: script, loop, n_samples, stop_reason
```

**Note:** URSim recordings have zero actual gap (perfect simulator). Meaningful
comparison requires a real robot.

---

## 6. analysis.py — inspect a recording

Prints per-joint stats and saves plots. Use this to compare baseline vs optimised.

```bash
# Analyse a recording, show shoulder joint (1) plot
python analysis.py --csv results/2026-08-10_16-45-00_triangle.csv --joint 1

# Analyse optimised recording
python analysis.py --csv results/2026-08-10_16-45-05_triangle.optimized.csv --joint 1

# Save results without opening a window (useful on headless / remote)
python analysis.py --csv baseline.csv --no-plot

# Default: data/test-4.csv, joint 1
python analysis.py
```

**Joint indices:** 0=base, 1=shoulder, 2=elbow, 3=wrist1, 4=wrist2, 5=wrist3

**Output (automatic):**
```
results/<datetime>_analysis_triangle/
  log.json                                 ← n_rows, dt_ms, per-joint stats
  current_shoulder.png                     ← target vs actual current for requested joint
  all_joints_overview.png                  ← gap RMS, gap max, position error for all 6 joints
```

**What to look for when comparing baseline vs optimised:**
- `gap RMS` and `gap max` going down → less vibration / overshoot
- `pos err` going down (path mode) → tighter position tracking
- Numbers matching run.py's predicted improvement → model is accurate

---

## Augmenting URSim recordings (simulator-only workaround)

When running on URSim the `actual_*` columns are zero (perfect simulator).
To see predicted gap values from the distillation model:

```python
from train_distillation_model import DistillModel, augment

model = DistillModel.load("models/distill.pkl")
augment(model, "results/2026-08-10_16-45-00_triangle.csv")
augment(model, "results/2026-08-10_16-45-05_triangle.optimized.csv")
# now run analysis.py — it will show predicted gap instead of zero
```

These are predictions, not real measurements. Real validation requires hardware.

---

## results/ folder structure

```
results/
  <datetime>/                              # distillation run
    distill.pkl
    log.json
    residuals_actual_current.png
    per_joint_rmse_actual_current.png
  <datetime>_rla_<mode>/                   # RL training run
    agent_<mode>.zip
    sim_to_real.csv
    log.json
    training_curve.png
  <datetime>_run_<mode>/                   # run.py optimisation
    <script>.sim_to_real.csv
    log.json
  <datetime>_<script>.csv                  # send.py recording
  <datetime>_<script>.csv.json             # send.py sidecar
  <datetime>_analysis_<csv>/              # analysis.py run
    log.json
    current_<joint>.png
    all_joints_overview.png
  runs_summary.csv                         # distillation runs comparison
  runs_summary_rla.csv                     # RL training runs comparison
  comparison_plot.png                      # distillation RMSE trend
```
