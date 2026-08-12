# Case 2 Review — Reality Gap Modeling & RL Motion Optimization

Working notes on the repository: what every file does, how data actually flows
through the pipeline, what we've verified by running it, and what needs to
change to satisfy the case brief (`docs/` PDF, "#6_UR Reality Gap Modeling &
RL Motion Optimization"). Written after running the full baseline pipeline
end to end on `apostolos` branch.

## 1. What the case actually asks for

Three layers (PDF "Challenge" section):

1. **Analyze the reality gap** — using RTDE recordings from a real UR10e
   (multiple runs, varied speed/accel/blend), characterize where and why
   tracking error and vibration occur: which joints, at what speeds, under
   what loads.
2. **Learn the gap** — train `actuals = f(targets, joint_config, velocity,
   acceleration, payload)`. A linear regression baseline is provided; beat it.
3. **Optimize with RL** — wrap the learned model as a Gym environment, train
   an agent to move as fast as possible while minimizing vibration.

Two vibration metrics the PDF specifies explicitly:

```
peak = max(|actual(t) - target(t)|)              for t in settling window
rms  = sqrt(mean((actual(t) - target(t))^2))      for t in settling window
```

**This is the first mismatch to know about**: the repo's shipped
`EvaluationMetric` (`CurrentGapMetric`) measures current-tracking error, not
either of these two position-based vibration metrics. See §5.

## 2. Repository structure, file by file

### Root — the pipeline code

| File | Role | Notes from testing |
|---|---|---|
| `utils.py` | `N_JOINTS`, column-name constants, `UR10e` (pure-numpy DH kinematics/dynamics: FK, Jacobian, gravity, mass matrix, Coriolis), URScript `vel`/`acc` read/replace helpers | `UR10e.coriolis()` exists but `UR10eDynamics` doesn't call it (see `dynamics.py`) |
| `analysis.py` | `Recording` — the CSV loader every other file depends on (wraps a run as numpy arrays: `target_q`, `actual_q`, `target_current`, `actual_current`, `vel_cmd`/`acc_cmd`, `scl`, `script`). CLI: `--csv --joint` prints per-joint stats and plots target vs actual current | The right tool to look at **real** `data/test-*.csv` gap; not useful on URSim-only recordings |
| `plot_target_actual.py` | Standalone quick-look plotter, added this session. Generalizes `analysis.py`'s current-only plot to *any* `target_<value><i>`/`actual_<value><i>` column pair in a recording CSV — `q`, `qd`, `current`, `TCP_pose`, `TCP_speed` — for one chosen component. No stats/log.json, just target-vs-actual and the gap, for eyeballing a channel `analysis.py` doesn't cover. See usage below and §5 (Bronze). | Reads the CSV directly via `utils.get_block`/`joint_cols`, doesn't go through `Recording` — works on any of the six `target_*`/`actual_*` channel pairs, not just the three `Recording` loads |
| `common.py` | `segments()` — splits a recording into per-`movej` `Segment`s using `script_control_line` to find move boundaries; shared by distillation and RL so both cut the same way | Confirmed: 30 segments from our 3 training scripts × 5 loops × 2 poses |
| `record.py` | Passive RTDE logger (raw socket protocol, port 30004). Never moves the robot | Also supplies `record_stream`, reused by `send.py` |
| `send.py` | Pushes a URScript or `servoj` path to port 30002, records via `record.py`. Auto-stops on a done-flag register. Needs Remote Control **and** the robot powered on — silently sits at a ~5s timeout otherwise (`_done_check`'s "program never started" reason, which `train_rla.py`'s `collect_moves` doesn't print, so it just looks like `0 segments pooled`) | This is what we hit twice this session — robot state, not code |
| `dynamics.py` | `Dynamics` interface + `UR10eDynamics` — commanded joint current for a *candidate* motion, without running it anywhere. `tau = M(q)qdd + g(q)` (Coriolis dropped), `current = tau/Kt`. Builds candidate frames with `actual_current = 0.0` placeholder | The `0.0` placeholder here is one of the two "no ground truth" cases discussed in §4 |
| `train_distillation_model.py` | `DistillModel` interface + `LinearModel` baseline — one shared-slope linear fit predicting `actual_current` from `[target_current, qd, qdd, pos, vel, acc, joint one-hot]`. `augment()` overwrites a recording's `actual_*` columns with predictions | Only predicts `actual_current` — `actual_q` is never touched anywhere in the pipeline unless this changes |
| `metrics.py` | `EvaluationMetric` interface + `CurrentGapMetric` baseline (`Σ\|actual_current − target_current\|`) | Doesn't implement the PDF's peak-overshoot/RMS-position spec |
| `preprocess.py` | `Preprocess` interface + `Identity` no-op — hook for scaling features into/out of the distill model and the RL agent | Untouched in the baseline run; `pos` (rad) and `vel`/`acc` (raw register ints up to ~1000) are on very different scales, flagged in `LinearModel`'s own docstring as a thing to fix |
| `train_rla.py` | `GapEnv`/`PathEnv` (Gym envs wrapping distill model + dynamics + metric) and `train_ppo()`. One-step contextual-bandit env: `reset()` picks a random move, `step(action)` scores a candidate `(vel,acc)` and ends the episode | `OBJECTIVE = score + cycle_time` (1:1 weights) is what the agent actually optimizes, not raw score |
| `run.py` | Loads a trained agent, asks it for vel/acc on a **held-out** script, prints predicted score/cycle-time before/after, writes `<script>.optimized.script` (or `.path`) | Everything it prints is a *prediction* from the distill model, not a measurement — see §4 |
| `README.md` | Setup, run commands, tiers | The "you might notice something is off" note in step 5 is the URSim-zero-gap issue, confirmed in §4 |
| `requirements.txt` | numpy/pandas/matplotlib/scikit-learn/scipy + gymnasium/stable-baselines3 | — |

### `data/` — real UR10e recordings (Git LFS)

`test-1.csv` … `test-7.csv`, each with a matching `.script` (`test-6-7.script`
covers both). **These are genuine real-hardware recordings**, not synthetic —
confirmed from the `.script` contents:

- `test-1`: vel and acc stepped together, 100→10 (steps of 10), on a fixed
  two-point move (`Point_1` ↔ `Point_2`)
- `test-2`: acc stepped 100→10, vel fixed at 100
- `test-3`: vel stepped 100→10, acc fixed at 100
- `test-4`/`test-5`: same idea with `sleep(0.75)` pauses between moves
- `test-6`/`test-7`: random vel/acc draws over a richer 5-point path

`test-4/5/6` is what the baseline distillation is trained on (per README).
`test-1/2/3` are clean single-variable sweeps — the best data for both Bronze
analysis and for validating a model's predicted-vs-real gap across a speed
range it may not have trained on.

**Required one-time setup**: `.gitattributes` tracks `*.csv`/`*.pptx` via Git
LFS. Without `git-lfs` installed, these files are a few bytes of pointer
text, and `Recording.__init__` fails with `KeyError: 't'`. Fixed this session
by installing `git-lfs` and running `git lfs pull`.

### `scripts/` — URScript motions

`shoulder_swing.script`, `vertical_swing.script`, `horizontal_swing.script` —
hand-written two-pose swings, `vel=999 acc=999` by default, used to train the
RL agent. `triangle.script` — the held-out script `run.py` optimizes.
`triangle.optimized.script` — generated by `run.py`, same script with
`vel`/`acc` rewritten to the agent's averaged choice.

### `models/` — trained artifacts (generated, not shipped)

`distill.pkl` — pickled `DistillModel`. `agent_params.zip` — Stable-Baselines3
PPO policy (`policy.pth` + optimizer state + version metadata — no
human-readable metrics inside; you have to *ask* the agent for actions to see
what it learned). `__init__` is a stray empty file, not a real package
marker.

### Root-level generated files (from running the pipeline)

`sim_to_real.csv` — step 2's pooled training set (targets from URSim +
distilled actuals + `score`). `triangle.sim_to_real.csv` — same for step 3's
held-out script. `baseline.csv`/`optimized.csv` — step 4's on-robot (in our
case, on-URSim) recordings.

## 3. How information actually flows

```mermaid
flowchart TD
    RealData["data/test-4,5,6.csv<br/>(real UR10e recordings)"] -->|train_distillation_model.py| DistillPkl["models/distill.pkl<br/>(DistillModel)"]

    Scripts["scripts/*.script<br/>(shoulder/vertical/horizontal)"] -->|send.py via URSim| Targets["clean target_* trajectory<br/>(sim_to_real.csv)"]
    Targets -->|augment(): model.predict| Injected["actual_current overwritten<br/>with model's prediction"]
    Injected -->|metrics.add_score| Scored["+ score column<br/>(CurrentGapMetric)"]
    Scored -->|common.segments| Moves["~30 Segment objects<br/>(one per movej)"]

    Moves --> GapEnv["GapEnv (Gym)"]
    DistillPkl -.scores candidates.-> GapEnv
    DynPhysics["dynamics.py: UR10eDynamics<br/>(M(q)qdd + g(q), no hardware)"] -.commanded current.-> GapEnv
    GapEnv -->|PPO, train_rla.py| Agent["models/agent_params.zip"]

    HeldOut["scripts/triangle.script<br/>(never trained on)"] -->|run.py: build_dataset| HeldOutData["triangle.sim_to_real.csv"]
    Agent -->|search_agent: predict per move, average| OptParams["optimized vel/acc"]
    HeldOutData --> OptParams
    OptParams -->|set_param| OptScript["scripts/triangle.optimized.script"]

    OptScript -->|send.py --robot-ip REAL ROBOT| RealValidation["actual on-hardware recording"]
    Scripts -->|send.py --robot-ip REAL ROBOT| BaselineValidation["baseline on-hardware recording"]
    RealValidation --> Analysis["analysis.py: does the predicted<br/>improvement hold?"]
    BaselineValidation --> Analysis
```

**The two tracks, and why this matters.** Every recording in this pipeline
carries a `target_*` track and an `actual_*` track. `target_*` is always
trustworthy — it's just the commanded geometry, and URSim reproduces it
exactly, same as physics-only `Dynamics.frame()` candidates. `actual_*` is
only meaningful if it came from **real hardware**, or was **explicitly
overwritten by the distill model's prediction**. Left alone, URSim reports
`actual_* = target_*` (verified: `actual_current` bit-for-bit equal to
`target_current` in `baseline.csv`/`optimized.csv`), and a freshly-built
candidate frame reports `actual_current = 0.0` (literal placeholder in
`dynamics.py`). Both are equally uninformative — zero gap, or an undefined
one. `augment()` is the only thing in the whole pipeline that puts a real
(learned) gap into a trajectory that never touched hardware.

**Consequence, confirmed this session**: `DistillModel.predicts()` for
`LinearModel` returns only `["actual_current"]`. So `augment()` only
overwrites that one column — `actual_q` stays exactly equal to `target_q`
*everywhere* in the pipeline, including inside `sim_to_real.csv` after
`augment()` has run. A position-based metric (which is what the PDF asks
for) would see zero gap at every single row, right now, regardless of what
the RL agent does.

**Consequence for step 4/5 validation**: `send.py --robot-ip 127.0.0.1`
targets URSim, which — like the untouched `target_*` track above — has no
reality gap by construction. We confirmed this directly: `baseline.csv` and
`optimized.csv`, both recorded against URSim, show `actual − target = 0.000A`
flatline for every joint, in both runs. **This validation step is only
meaningful against real hardware.** Everything upstream of it (steps 1-3) can
be fully exercised and reasoned about without a robot; step 4/5 as written
cannot, unless you either get real hardware time or substitute the offline
validation strategy in §5 (Diamond).

## 4. Findings from actually running it this session

- **Git LFS pointers, not data** — one-time env fix, not a pipeline bug.
- **URSim starts powered off; Remote Control alone isn't enough** — the
  robot also needs powering on / brakes released (Robot State: RUNNING), or
  `send.py` silently times out at ~5s and every script "records" the same
  suspiciously-identical sample count with 0 segments.
- **URSim has zero reality gap by construction** — `actual_current` exactly
  equals `target_current`, confirmed numerically and visually (flat gap
  plots, target/actual traces overlapping exactly).
- **`data/test-*.csv` are real hardware sweeps**, not synthetic — a resource
  for Bronze analysis and offline validation that doesn't require live
  hardware access.
- **`train_distillation_model.py`'s held-out split is row-level**, not
  run-level (every 5th row, `--holdout 0.2`) — optimistic, since a held-out
  row's neighbors are still in training. Our run: RMSE 0.627A, R² 0.978
  pooled across all 6 joints — but joint current scales vary hugely (base
  ±13A vs wrist3 ±0.0005A), so a single pooled RMSE can hide a bad fit on the
  small joints. Not yet checked per-joint.
- **Coefficients from our run**: `target_current +1.09` (expected),
  `qd -0.374` (real signal — instantaneous commanded speed matters),
  `vel +0.0001` / `acc +0.0001` (the raw movej register barely matters
  *directly* — the model only feels speed through the resulting `qd`/`qdd`
  shape of the trajectory).
- **Agent behavior (previewed offline on the 30 training moves)**: always
  chooses much lower vel (~97–155 vs script's 999) and moderately lower acc
  (~268–532 vs 999), trading ~55-60% longer cycle time for ~32.5% lower
  predicted score — a genuine ~18% improvement in the actual `OBJECTIVE`
  (score + cycle_time), not just gaming one term.
- **`sim_to_real.csv`'s worst-predicted-gap rows are all `shoulder_swing` at
  `vel=999,acc=999`** — the untouched fast baseline — a sensible, physically
  plausible signal for the agent to have learned from.

## 5. What needs to change, tier by tier

### Prerequisite (blocks a correct Gold/Diamond, do first)

- **Metric**: implement the PDF's `peak overshoot` and `RMS position error`
  in `metrics.py` as a new `EvaluationMetric`, using `common.Segment`'s
  `i1→i2` as the settling window.
- **Model output**: extend `DistillModel`/`LinearModel` (or your replacement)
  to also predict `actual_q`, not just `actual_current` — otherwise the new
  position metric sees zero gap everywhere, same dead end as URSim, one
  layer deeper.

### Bronze — characterize the gap (fully offline)

- Use `analysis.py` on `data/test-1.csv` … `test-5.csv` (real sweeps), not on
  URSim-derived `baseline.csv`/`optimized.csv`.
- `test-2` (acc sweep, vel fixed) and `test-3` (vel sweep, acc fixed) are the
  cleanest single-variable views — plot gap RMS/max vs the swept parameter,
  per joint.
- For channels `analysis.py` doesn't plot — `q`, `qd`, `TCP_pose`,
  `TCP_speed` — use `plot_target_actual.py` (run with the `urenv` conda env,
  which has pandas/matplotlib; base conda does not):

  ```bash
  # target vs actual current, shoulder joint (same channel analysis.py plots)
  python plot_target_actual.py --csv data/test-3.csv --value current --joint 1

  # position tracking error, base joint — useful once the metric/model work
  # below also predicts actual_q, not just actual_current
  python plot_target_actual.py --csv data/test-3.csv --value q --joint 0

  # TCP-space pose gap, z axis (component index 0..5 = x,y,z,rx,ry,rz)
  python plot_target_actual.py --csv data/test-3.csv --value TCP_pose --joint 2

  # save instead of opening a window, e.g. for a headless run
  python plot_target_actual.py --csv data/test-2.csv --value qd --joint 3 \
      --save wrist1_qd_gap.png --no-show
  ```

  Top subplot: target vs actual. Bottom: actual − target (the gap). `--joint`
  is the component index 0..5 — a joint for `q`/`qd`/`current`, or
  x/y/z/rx/ry/rz for the two `TCP_*` channels. No results/ output is written
  (unlike `analysis.py`) — this is a quick-look tool, not a pipeline step.
- Decide and justify what `EvaluationMetric` should measure (the README
  frames this as a Bronze task) — this is where the peak-overshoot/RMS work
  above gets motivated and written up.
- Explore `Preprocess`: `pos` (rad) vs `vel`/`acc` (raw ints to ~1000) are on
  very different scales — worth normalizing before Silver.

### Silver — build and honestly validate a `DistillModel`

- Replace `LinearModel`: an MLP, random forest, or per-joint fit (shared
  slopes across joints is a known limitation of the baseline, per its own
  docstring) — something that can capture the post-stop ring a linear model
  structurally cannot.
- Fix the held-out evaluation to be **run-level**: train on
  `test-1,2,4,5`, evaluate entirely on `test-3` (a full run, and a parameter
  sweep — vel — that `test-2` never varied). Report per-joint RMSE/R², not
  just pooled.
- Once the model predicts `actual_q`, validate that channel the same way.
- Offline predicted-vs-actual check: for the held-out sweep, plot the
  model's predicted metric vs the real measured metric across the swept
  vel/acc range — this is the strongest generalization evidence you can get
  without hardware.

### Gold — improve the agent and the physics model

- `Dynamics`: `UR10eDynamics` drops Coriolis (`utils.UR10e.coriolis()`
  already exists, just isn't called) and uses generic UR10e parameters —
  add friction and/or identify parameters from your real data.
- `train_rla.py`: retrain `GapEnv` once the metric/model changes land;
  revisit `OBJECTIVE`'s 1:1 `SCORE_WEIGHT`/`CYCLE_WEIGHT` if the new metric's
  scale differs a lot from `CurrentGapMetric`'s.
- Compare the agent's chosen params against a fixed-parameter baseline using
  the same held-out-real-data method as Silver, not URSim.
- Path mode (`--mode path`) is available and has the same caveats — treat it
  as a second thing to validate the same way, not a shortcut around
  validation.

### Diamond — real hardware (or the offline substitute)

- **With hardware**: rerun `send.py` for baseline and optimized with
  `--robot-ip` pointed at a real UR10e instead of `127.0.0.1` — this is the
  step that was silently meaningless every time we ran it against URSim this
  session. If the predicted improvement doesn't hold, refit
  `DistillModel`/`Dynamics` on the new real recordings (the README's own
  words: "sends you back to the distillation").
- **Without hardware yet**: the offline validation built up through Bronze/
  Silver/Gold (predicted-vs-actual on held-out real sweeps) is the closest
  substitute, and is worth writing up as evidence even before hardware time
  is available — it demonstrates the same sim-to-real transfer claim, just
  against already-collected data instead of a live run.

## 6. Suggested order of work

1. Metric + model-output fix (unblocks everything else being *meaningful*)
2. Bronze analysis on real `data/` sweeps (cheap, informs feature choices)
3. Silver model + run-level held-out validation
4. Gold agent + Dynamics improvements, validated the same offline way
5. Diamond — real hardware when available; offline substitute otherwise
