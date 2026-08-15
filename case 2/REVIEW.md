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
either of these two position-based vibration metrics. See §6.

## 2. Repository structure, file by file

### Root — the pipeline code

| File | Role | Notes from testing |
|---|---|---|
| `utils.py` | `N_JOINTS`, column-name constants, `UR10e` (pure-numpy DH kinematics/dynamics: FK, Jacobian, gravity, mass matrix, Coriolis), URScript `vel`/`acc` read/replace helpers | `UR10e.coriolis()` exists but `UR10eDynamics` doesn't call it (see `dynamics.py`) |
| `analysis.py` | `Recording` — the CSV loader every other file depends on (wraps a run as numpy arrays: `target_q`, `actual_q`, `target_current`, `actual_current`, `vel_cmd`/`acc_cmd`, `scl`, `script`). CLI: `--csv --joint` prints per-joint stats and plots target vs actual current | The right tool to look at **real** `data/test-*.csv` gap; not useful on URSim-only recordings |
| `plot_target_actual.py` | Standalone quick-look plotter, added this session. Generalizes `analysis.py`'s current-only plot to *any* `target_<value><i>`/`actual_<value><i>` column pair in a recording CSV — `q`, `qd`, `current`, `TCP_pose`, `TCP_speed` — for one chosen component. No stats/log.json, just target-vs-actual and the gap, for eyeballing a channel `analysis.py` doesn't cover. See usage below and §6 (Bronze). | Reads the CSV directly via `utils.get_block`/`joint_cols`, doesn't go through `Recording` — works on any of the six `target_*`/`actual_*` channel pairs, not just the three `Recording` loads |
| `common.py` | `segments()` — splits a recording into per-`movej` `Segment`s using `script_control_line` to find move boundaries; shared by distillation and RL so both cut the same way | Confirmed: 30 segments from our 3 training scripts × 5 loops × 2 poses |
| `record.py` | Passive RTDE logger (raw socket protocol, port 30004). Never moves the robot | Also supplies `record_stream`, reused by `send.py` |
| `send.py` | Pushes a URScript or `servoj` path to port 30002, records via `record.py`. Auto-stops on a done-flag register. Needs Remote Control **and** the robot powered on — silently sits at a ~5s timeout otherwise (`_done_check`'s "program never started" reason, which `train_rla.py`'s `collect_moves` doesn't print, so it just looks like `0 segments pooled`) | This is what we hit twice this session — robot state, not code |
| `dynamics.py` | `Dynamics` interface + `UR10eDynamics` — commanded joint current for a *candidate* motion, without running it anywhere. `tau = M(q)qdd + g(q)` (Coriolis dropped), `current = tau/Kt`. Builds candidate frames with `actual_current = 0.0` placeholder | The `0.0` placeholder here is one of the two "no ground truth" cases discussed in §5 |
| `train_distillation_model.py` | `DistillModel` interface + `LinearModel` baseline — one shared-slope linear fit predicting `actual_current` from `[target_current, qd, qdd, pos, vel, acc, joint one-hot]`. `augment()` overwrites a recording's `actual_*` columns with predictions | Only predicts `actual_current` — `actual_q` is never touched anywhere in the pipeline unless this changes |
| `metrics.py` | `EvaluationMetric` interface + `CurrentGapMetric` baseline (`Σ\|actual_current − target_current\|`) | Doesn't implement the PDF's peak-overshoot/RMS-position spec |
| `preprocess.py` | `Preprocess` interface + `Identity` no-op — hook for scaling features into/out of the distill model and the RL agent | Untouched in the baseline run; `pos` (rad) and `vel`/`acc` (raw register ints up to ~1000) are on very different scales, flagged in `LinearModel`'s own docstring as a thing to fix |
| `train_rla.py` | `GapEnv`/`PathEnv` (Gym envs wrapping distill model + dynamics + metric) and `train_ppo()`. One-step contextual-bandit env: `reset()` picks a random move, `step(action)` scores a candidate `(vel,acc)` and ends the episode | `OBJECTIVE = score + cycle_time` (1:1 weights) is what the agent actually optimizes, not raw score |
| `run.py` | Loads a trained agent, asks it for vel/acc on a **held-out** script, prints predicted score/cycle-time before/after, writes `<script>.optimized.script` (or `.path`) | Everything it prints is a *prediction* from the distill model, not a measurement — see §5 |
| `README.md` | Setup, run commands, tiers | The "you might notice something is off" note in step 5 is the URSim-zero-gap issue, confirmed in §5 |
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
validation strategy in §6 (Diamond).

## 4. The RL training loop, step by step

### `movej`, briefly

URScript's joint-space move: `movej(pose, a=acc, v=vel, r=blend)` interpolates
all six joints simultaneously from the current angles to `pose` (a joint-space
path, not a straight line in TCP space — that's `movel`), ramping speed with a
trapezoidal profile bounded by `a`/`v`, optionally blending into the next move
with radius `r`. Every script in `scripts/` is a sequence of `movej` calls
between fixed poses. `vel`/`acc` are exactly the two numbers read/written by
`utils.get_param`/`set_param`, and the two numbers `run.py`/`GapEnv` search
over. `common.segments()` cuts a recording into one `Segment` per `movej`
using `script_control_line` to find where one move ends and the next begins —
that Segment is the unit `train_rla.py` scores and picks parameters for.

### What the agent is trying to do

Faster `vel`/`acc` finishes a move sooner but overshoots and rings more when
the robot stops. `GapEnv`/`PathEnv` search for `vel`/`acc` (or a re-timed
speed profile) per move that minimizes that ringing without giving up too
much cycle time: `reward = -(SCORE_WEIGHT * score + CYCLE_WEIGHT *
cycle_time)`, `OBJECTIVE` at `train_rla.py:69-71`.

### The training loop (`train_rla.py`)

1. **Get real target geometry, cheaply.** `collect_moves` (`train_rla.py:157-180`)
   runs the training scripts on URSim — not for the gap (URSim has none, §3),
   just to get clean `target_q` trajectories to cut into moves via
   `common.segments`.
2. **Inject the learned reality gap.** `build_dataset` (`train_rla.py:487-501`)
   calls `augment()`, which overwrites `actual_current` with `DistillModel`'s
   prediction — the only place a *learned*, real-data-derived gap enters the
   loop — then `metrics.add_score` labels every row. Result: `sim_to_real.csv`.
3. **One-step contextual bandit, not sequential RL** (`_MoveEnv`,
   `train_rla.py:183-243`): `reset()` picks a random move; `step(action)`
   scores it and the episode ends immediately. No credit assignment across a
   trajectory — each `movej` is optimized independently, which fits since its
   parameters don't depend on history.
4. **Scoring a candidate the agent proposes** (`GapEnv._cost` → `score` →
   `_candidate`, `train_rla.py:224-281`): the action maps to `(vel, acc)`;
   `dynamics.trapezoidal()` builds the speed profile; `Dynamics.frame()`
   (physics-only, §2/§5) builds the *commanded* trajectory for that
   hypothetical speed — never run anywhere, real or simulated; `evaluate()`
   runs it through the same `DistillModel` to predict `actual_current`; the
   metric scores the gap. PPO (`train_ppo`, `train_rla.py:472-484`) trains on
   thousands of these synthetic episodes.
5. Output: `models/agent_<mode>.zip`, plus `results/<dt>_rla_<mode>/log.json`
   + `training_curve.png`, and a row appended to `runs_summary_rla.csv`.

`path` mode (`PathEnv`) keeps the recorded geometry and only re-times it — the
agent picks `accel_frac`/`decel_frac`/servoj `dt` (`speed_profile`,
`train_rla.py:120-138`) instead of a straight-line `vel`/`acc`.

### From trained agent to a robot script (`run.py` → `send.py`)

`run.py` repeats steps 1-2 above for one **held-out** script (`triangle.script`,
never in training), asks the trained agent for its choice per move
(`search_agent`, `run.py:59-63`), averages it, and either rewrites the
script's `vel`/`acc` (`run_params`, `run.py:74-101`, writes
`<script>.optimized.script`) or writes a dense servoj setpoint CSV
(`run_path`, `run.py:140-168`, writes `<script>.path`). Everything `run.py`
prints ("predicted mean score: 0.847 → 0.612") is the distill model's
opinion, not a measurement (§2, `run.py` row) — the only way to check it is
`send.py --script triangle.optimized.script --robot-ip <real IP>`, then
compare the recording against the baseline with `analysis.py`/
`plot_target_actual.py`. Point `send.py` at URSim instead and both baseline
and optimized come back with the same flat zero gap, per §3/§4.

What this loop misses (dropped Coriolis, generic dynamics params,
`actual_current`-only distill model, no model/agent provenance on the
resulting recording) is covered where it's actionable, in §5-§6 — not
repeated here.

## 5. Findings from actually running it this session

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
- **`train_distillation_model.py`'s held-out split was row-level**, not
  run-level (every 5th row, `--holdout 0.2`) — optimistic, since a held-out
  row's neighbors are still in training. Our original run: RMSE 0.627A, R²
  0.978 pooled across all 6 joints — but joint current scales vary hugely
  (base ±13A vs wrist3 ±0.0005A), so a single pooled RMSE can hide a bad fit
  on the small joints. **Resolved 2026-08-15**: the script now defaults to a
  fixed file-level split (train `test-1,2,3,6.csv`, test `test-4,5,7.csv`,
  never mixed) and reports per-joint RMSE/R² — see §9.2. With that split,
  held-out RMSE 0.680A/R² 0.977 vs in-sample RMSE 0.670A/R² 0.973: close
  enough that the linear model isn't meaningfully overfitting the 4 training
  runs, but wrist2/wrist3 R² is weak (0.43/0.07) — see §9.2's per-joint
  caveat.
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
- **A concrete `run.py` result, and a real extrapolation problem**:
  `results/2026-08-10_20-44-14_run_params/log.json` — optimizing the
  held-out `triangle.script` — reports baseline `vel=999,acc=999` → score
  `2.075`, optimized `vel=180.0,acc=469.74` → score `1.855` (**+10.6%
  score, but −22.4% cycle time**, i.e. cycle time got *worse*, traded off
  because `OBJECTIVE` weights score and cycle time 1:1). But `data/test-1/2/3`
  — the only real hardware `vel`/`acc` sweeps this repo has — never go above
  100 for either parameter, while `GapEnv`'s search space
  (`VEL_BOUNDS=(20,180)`, `ACC_BOUNDS=(40,600)`, `train_rla.py:63-64`) let the
  agent land on `acc=469.74`, more than 4.5x past anything the distill model
  was ever trained or validated against. The predicted 10.6% score
  improvement above is therefore mostly extrapolation, not just an unverified
  number — see §8 for what that means for presenting this result.

## 6. What needs to change, tier by tier

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
- ~~Fix the held-out evaluation to be **run-level**~~ — done 2026-08-15: the
  script now defaults to a fixed file-level split (train `test-1,2,3,6.csv`,
  test `test-4,5,7.csv`) and reports per-joint RMSE/R² alongside pooled, for
  whichever `DistillModel` is selected (§9.2).
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
- **Provenance**: a `send.py` recording of `<script>.optimized.script`
  doesn't record which `distill.pkl`/`agent_params.zip` produced its
  `vel`/`acc` — only `run.py`'s own `results/<dt>_run_<mode>/log.json`
  (`model`, `agent` fields) has that, correlated to the later recording only
  by timestamp order. Worth a small sidecar (or a field in `send.py`'s
  `.csv.json`) once you're iterating on multiple model/agent versions and
  need to know which pairing a given on-robot result came from.

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

## 7. Suggested order of work

1. Metric + model-output fix (unblocks everything else being *meaningful*)
2. Bronze analysis on real `data/` sweeps (cheap, informs feature choices)
3. Silver model + run-level held-out validation
4. Gold agent + Dynamics improvements, validated the same offline way
5. Diamond — real hardware when available; offline substitute otherwise

## 8. Validating without hardware, and what to present as results

No real robot time closes the loop this pipeline is built around — but that
doesn't mean there's nothing defensible to show. It means being precise about
which numbers are *measured* and which are *predicted*, and building the
strongest offline case for the predicted ones. This section is the plan for
that, and how to turn it into something to hand to the company.

### The validation plan (do this, in order)

1. ~~**Run-level held-out split for the distill model**~~ (§6 Silver) — done
   2026-08-15: `train_distillation_model.py` now defaults to a fixed
   file-level split (train `test-1,2,3,6.csv`, test `test-4,5,7.csv` — full
   real runs the model never saw during `fit()`) and reports per-joint
   RMSE/R², since pooled RMSE hides a bad fit on small joints (wrist current
   is ~1/20,000th of base current — §5). See §9.2 for how to read the
   result.
2. **Predicted-vs-real curve across a swept parameter.** `test-2` (acc
   100→10, vel fixed) and `test-3` (vel 100→10, acc fixed) are clean
   single-variable sweeps. Hold one out, plot the model's *predicted* gap
   metric next to the *real measured* one as a function of the swept value.
   Matching shape and magnitude is real evidence the model has learned the
   actual speed-vs-vibration relationship — not just that it fits noise.
   (No script does this plot yet — a natural extension of
   `plot_target_actual.py`, which currently plots one CSV's target vs actual,
   not a metric curve across several files/settings.)
3. **Trust-region check on every optimized script before reporting its
   predicted improvement.** Real data (`test-1/2/3`) only covers `vel`/`acc`
   in `[10, 100]`. `GapEnv`'s search space goes up to `(180, 600)`
   (`train_rla.py:63-64`). Before presenting a number, check where the
   agent's chosen point actually landed — §5's worked example
   (`results/2026-08-10_20-44-14_run_params/log.json`) landed at
   `acc=469.74`, over 4x past anything real data ever covered. Inside
   `[10,100]²`: treat the predicted number as reasonably grounded. Outside
   it: say so explicitly, don't present it as a number, present it as a
   hypothesis to test.
4. **Physical plausibility as a free sanity check.** Step 2's real sweeps
   already tell you the true direction of the effect (slower ⇒ less gap).
   If the model or agent ever predicts something that contradicts that
   direction, that's a red flag you can catch without any hardware at all.
5. **Optional, more rigor**: train a few distill models on bootstrap
   resamples of the real training data and check how much they disagree at
   the agent's chosen operating point. High disagreement = the model is
   unsettled there even by its own standards; low disagreement = at least
   internally consistent (still not proof, but a second independent check).

### Be precise about what "improvement" means right now

Two caveats that matter more for a company presentation than for your own
notes:

- **The metric being optimized is `CurrentGapMetric`** (Σ|actual − target|
  current), **not** the case brief's peak-overshoot/RMS-position vibration
  spec (§1, §6 Prerequisite). A "10.6% score improvement" is a 10.6%
  reduction in *predicted current-tracking gap*, not a measured reduction in
  vibration or position error. Don't let that distinction blur when writing
  the headline.
- **Baseline and optimized are scored through the same model**
  (`run.py`'s `_compare`, `run.py:51-56`), so the *delta* between them is
  internally consistent even before hardware validates the absolute numbers
  — that's a legitimate thing to say. "The model predicts X% less gap" is
  defensible today. "We achieved X% less gap" is not, until it's measured.

### What to present to the company

Structure the story around what's actually verified vs. predicted — this is
more credible than a single "our RL agent improves things by 18%" headline,
and it's also just accurate:

1. **What we measured on real hardware (defensible today, no caveats
   needed).** The Bronze-tier characterization from `data/test-*.csv`: which
   joints show the most gap, at what speeds, and the RMS/peak numbers per
   joint from `analysis.py`/`plot_target_actual.py`. This is real,
   hardware-verified evidence of the reality gap existing and where it lives
   — the strongest slide in the deck because nothing about it depends on the
   model being right.
2. **What we learned to predict it, and how well (predicted, with an honest
   accuracy number attached).** The distill model's *run-level* held-out
   RMSE/R² from step 1 above (not the current row-level 0.627A/0.978 in
   `results/runs_summary.csv` — that number is optimistic and shouldn't go
   in front of the company until it's redone run-level). Pair every accuracy
   number with what it was validated against, e.g. "predicts real current
   gap on `test-3`, a full run held out during training, with RMSE Y."
3. **What the RL agent recommends, and why it should be trusted or not
   (clearly labeled predicted).** The `run.py` before/after table
   (`results/<dt>_run_params/log.json`) — vel/acc, predicted score, predicted
   cycle time, % change — plus the trust-region check from step 3 above as an
   explicit column or footnote: "within validated range" or "extrapolated,
   pending hardware confirmation." Frame the result as the tradeoff it is:
   faster cycle time (throughput) traded against vibration (wear on the arm
   and tooling, precision) — that framing is what makes this relevant to a
   company, more than the raw percentage.
4. **Honest next steps.** One slide: real-hardware validation is the
   remaining step to convert "predicted" into "measured" (§6 Diamond); until
   then, every number in slide 3 is a hypothesis backed by slides 1-2's
   evidence, not a result.

**Artifacts already sitting in `results/` to pull numbers and plots from**,
rather than re-deriving them for a presentation: `runs_summary.csv` /
`runs_summary_rla.csv` (one row per training run, for "how did the model/
agent improve over iterations" if you retrained more than once),
`comparison_plot.png` / `training_curve.png` (already-rendered trend plots),
and each run's `log.json` (exact numbers, reproducible by rerunning the same
command). Cite the file path alongside any number you present, the same way
this document does — it's the difference between a number a reviewer can
check and one they have to take on faith.

## 9. The full pipeline, command by command

A practitioner's runbook: the six scripts in run order, the exact command,
what each one does on *every* invocation (including the side effects that
don't show up in the printed output), the files it leaves behind and how to
read them, and — the part `COMMANDS.md` doesn't cover — what to actually
check to tell whether a new run gave you a *better* distill model or RL
agent, not just *a different* one. See `COMMANDS.md` for the flag reference;
this section is about interpreting results, not syntax.

Two things apply to every stage below and are easy to miss:

- **"latest" paths are silently overwritten.** `models/distill.pkl` and
  `models/agent_<mode>.zip` get replaced by whatever you last trained —
  there's no confirmation and no diff. The versioned copy in
  `results/<datetime>/` (or `results/<datetime>_rla_<mode>/`) is the only
  record of what an older model actually was. If you want to compare two
  models deliberately, point `run.py --model`/`--agent` at the versioned
  paths, not the latest ones.
- **Provenance is manual.** `train_rla.py`'s `log.json` (`train_rla.py:390-404`)
  records `mode`, `scripts`, `steps`, and the agent's own paths — it does
  *not* record which `distill.pkl` (`args.model`) it trained against. Two
  agents trained on different distill models will look identical in
  `runs_summary_rla.csv` even though their scores aren't comparable (§9.3).
  Write the distill model path down yourself (file name, commit message,
  whatever) until this is fixed — it's the same gap already flagged for
  `send.py` recordings in §6's Gold tier, one step earlier in the chain.

### 9.1 `record.py` — capture real data

```bash
python record.py --robot-ip 127.0.0.1 --out data/test-7.csv \
    --float-register 1 vel 2 acc
```

Passively logs the RTDE state stream while a separately-sent script runs;
never moves the robot itself. Every run appends nothing anywhere — it just
writes the one CSV at `--out`.

**File:** `data/test-N.csv` — one row per RTDE sample, `target_*`/`actual_*`
columns per joint (`q`, `qd`, `current`, …) plus whatever `--float-register`
named (`vel`, `acc` — read by the distillation pipeline, §2's file table).

**What to check before trusting it downstream:** open it in
`analysis.py --csv data/test-7.csv --no-plot` and confirm `n_rows` and
`dt_ms` look sane for the recording length, and that `vel`/`actual_current`
aren't all zero (URSim source, or the robot wasn't actually RUNNING —
§5's Remote Control finding). This CSV silently becomes training data for
step 9.2, so a bad recording here corrupts the distill model with no error
anywhere else in the pipeline.

### 9.2 `train_distillation_model.py` — fit the reality-gap model

```bash
python train_distillation_model.py --out models/distill.pkl
```

As of 2026-08-15 this is a **fixed file-level train/test split by default** —
`model.fit()` only ever sees `DEFAULT_TRAIN_CSVS` (`data/test-1,2,3,6.csv`),
and every printed/plotted/logged metric comes from predicting the disjoint
`DEFAULT_TEST_CSVS` (`data/test-4,5,7.csv`), which the model never trains on.
The pairing isn't arbitrary: `{2,4}` are both acc sweeps at vel=100, `{3,5}`
are both vel sweeps at acc=100, `{6,7}` are both wide random vel/acc combos
(`test-1` is a standalone low-range grid) — holding out one file from each
pair means the test set covers the same regimes as training, so the number
measures generalization to a new run, not extrapolation into an unseen
speed/accel range. Override with `--train-csvs`/`--test-csvs` (the two must
not overlap — the script exits if they do); pick `--model` to train a
different registered `DistillModel` subclass (`MODELS` at the top of the
file) — the split logic doesn't change based on which one you pick. The
model saved to `--out` (default `models/distill.pkl`) is fit on the training
files only — it is *not* refit on the test files afterward, so the reported
numbers describe the exact model that gets deployed.

**Files, and how to read them:**
- `results/<datetime>/log.json` — `held_out_metrics.actual_current.overall.
  {rmse,r2}` is the number that matters: RMSE in amps (lower is better), R²
  in [0,1] (closer to 1 is better). `in_sample_metrics` is the same model
  evaluated on the *training* files — a sanity check only (should look at
  least as good as held-out; if it doesn't, the fit itself is broken, not a
  generalization issue). `training.{train_csvs,test_csvs}` records exactly
  which files were used for each. `params` is the fitted `LinearModel`
  coefficients (`coef` per feature).
- `per_joint_rmse_actual_current.png` — bar per joint, computed on the
  held-out files; check no single joint (usually wrist2/wrist3, lighter
  links, less current signal — see §5) is dragging the overall RMSE up so
  much that the "average" number hides a joint the model can't predict at
  all.
- `residuals_actual_current.png` — held-out residuals; should look centered
  on zero with no obvious structure (a slope or curve here means the linear
  model is missing a term, not just noisy).
- `results/runs_summary.csv` / `results/comparison_plot.png` — one row/point
  per training run ever done, `*_rmse`/`*_r2` columns are the held-out
  numbers above; this is the only place you can see the trend across runs
  rather than one run's number in isolation.

**Is this distill model better than the last one?**
1. Compare `held_out_metrics.actual_current.overall.{rmse,r2}` against the
   *previous row* in `runs_summary.csv`, not just against the number in your
   head — RMSE down and R² up is the win condition. Because the split is
   now file-level and fixed, these numbers are directly comparable across
   runs in a way the old row-level holdout wasn't (§8's caveat about
   adjacent-row leakage no longer applies to this metric — it still applies
   to anything that reintroduces a row-level split).
2. Look at `per_joint_rmse` before declaring victory — an improved overall
   RMSE that comes from getting the already-good joints slightly better
   while a bad joint stays bad isn't the same as a genuinely better model.
3. Compare `held_out_metrics` against `in_sample_metrics` in the same
   `log.json`: a large gap (held-out much worse than in-sample) means the
   model is overfitting to the four training runs, not learning the
   underlying gap.
4. `--train-csvs`/`--test-csvs` change what "better" means — only compare
   runs that used the same split (the default, unless you deliberately
   swept it).

### 9.3 `train_rla.py` — train the RL agent

```bash
python train_rla.py --mode params --model models/distill.pkl \
    --robot-ip 127.0.0.1 \
    --scripts scripts/shoulder_swing.script scripts/vertical_swing.script scripts/horizontal_swing.script \
    --loop 5 --steps 20000
```

Every run: runs `--scripts` on URSim once to get target geometry
(`collect_moves`), builds `sim_to_real.csv` by scoring every candidate
through the `--model` you pointed it at (§4 step 2 — this is where the
distill model enters), trains a fresh PPO policy from random initialization
for `--steps` timesteps (not resumed from any previous agent — every run is
a cold start), overwrites `models/agent_<mode>.zip`, and appends a row to
`results/runs_summary_rla.csv`.

**Files, and how to read them:**
- `results/<dt>_rla_<mode>/log.json` — `summary.best_score` (lowest score
  seen during training — score is the current-gap metric, §4, lower is
  better) and `summary.final_score_mean` (mean score over the last 10% of
  episodes — the policy's *converged* performance, more representative than
  `best_score`, which can be one lucky episode). `summary.best_cycle_time`
  is the fastest move found, reported separately because score and cycle
  time trade off against each other (§4's `OBJECTIVE`) — a lower score with
  a much longer cycle time isn't a free win.
- `training_curve.png` — three stacked plots (score, cycle time, reward)
  vs. timestep, each with a light per-episode scatter and a rolling mean.
  **What to look for:** the rolling-mean score curve should flatten out
  before training ends — if it's still trending down at the last timestep,
  `--steps` was too low and `final_score_mean` understates what the agent
  could reach. A curve that's noisy/flat from the start with no downward
  trend at all means the agent isn't learning (check the `--model` you
  passed is actually fit, not a stale/default one).
- `sim_to_real.csv` in the same folder — the exact synthetic dataset PPO
  trained on; useful to re-inspect what geometries/speeds it was actually
  exposed to if an agent behaves oddly on a script outside that mix.
- `results/runs_summary_rla.csv` — one row per training run: `mode`,
  `steps`, `best_score`, `final_score_mean`, `best_cycle_time`, `scripts`.

**Is this RL agent better than the last one?**
1. Compare `final_score_mean` (not `best_score`, which cherry-picks) across
   rows of `runs_summary_rla.csv` — but **only between rows trained against
   the same `distill.pkl`**. Since that path isn't logged (this section's
   intro), cross-check by date/your own notes before comparing two rows;
   a lower score against a different (e.g. newly refit, §9.2) distill model
   isn't evidence of a better policy, it's evidence of a different reward
   function.
2. Same caveat for `--scripts` and `--loop`: an agent trained on more/other
   scripts sees a different move distribution, so its `final_score_mean`
   isn't directly comparable to one trained on fewer. Compare agents that
   were trained on the same script set when judging "did more `--steps`
   help."
3. Check `best_cycle_time` moved in a direction you'd accept — an agent
   that drove score down by making every move much slower "solved" the
   metric, not the actual problem (§4's `CYCLE_WEIGHT` is what's supposed
   to prevent this; verify it did).
4. This is still all against the distill model's opinion, not reality —
   treat "better agent" here as "better at the offline objective," and only
   promote it to "better in fact" after 9.4 + 9.5 + `analysis.py` agree
   (§8).

### 9.4 `run.py` — evaluate the agent on the held-out script

```bash
python run.py --mode params --script scripts/triangle.script \
    --model models/distill.pkl --robot-ip 127.0.0.1
```

Every run: runs `triangle.script` (never in `train_rla.py`'s `--scripts`,
so it's the one held-out test of generalization) on URSim once, scores the
script's own `vel`/`acc` as the baseline and the trained agent's choice as
`optimized` — both through the *same* `--model`, so the delta between them
is internally consistent even before hardware confirms the absolute numbers
(§8) — then writes `scripts/triangle.optimized.script`.

**Files, and how to read them:**
- Terminal + `log.json`'s `results.improvement.{score_pct,cycle_time_pct}` —
  positive = the agent's choice scores/cycles lower than the script's
  original numbers, *predicted*. This is the number from §5's `acc=469.74`
  finding: a big percentage here can mean genuine improvement or the agent
  exploiting a region the distill model was never trained on (§8's
  trust-region check) — treat a large gain with suspicion, not celebration.
- `log.json`'s `results.optimized.{vel,acc}` (params mode) — **check these
  against the real data's coverage before trusting the score above.**
  `train_rla.py`'s `VEL_BOUNDS`/`ACC_BOUNDS` (`train_rla.py:63-64`) bound
  what the agent could *pick*, but that's much wider than what `data/test-*.csv`
  ever actually swept — if `optimized.acc` is near the search bound rather
  than near the real data's range, the predicted score is extrapolation
  (§5, §8's trust-region check).
- `log.json`'s `distill_model`/`agent` fields — this is the one place in the
  pipeline that *does* record which model/agent produced a result; useful
  as the provenance record §9.3 doesn't give you.
- `scripts/triangle.optimized.script` — hand this to 9.5, don't edit it by
  hand.

**Is this a better result than a previous `run.py` call?** Only comparable
if `--model` (and, less directly, `--agent`) are the same between the two
calls — otherwise you're comparing predictions from two different reward
functions again (same caveat as 9.3). The number that actually matters is
whether 9.5 + `analysis.py` confirm the prediction on hardware, not whether
this predicted percentage is bigger than last time's.

### 9.5 `send.py` — put it on the robot (or URSim) and record

```bash
python send.py --script scripts/triangle.optimized.script --loop 10 \
    --out results/triangle_optimized.csv
```

The only step that puts real motion on hardware (or URSim). Every run:
loops the script/path `--loop` times, records every RTDE sample, and writes
a `.json` sidecar next to the CSV.

**File — sidecar `.json`:** `n_samples`, `stop_reason` (`"program finished"`
on a clean run; `"program stopped (aborted? check URSim Log Messages)"`,
`"program never started (is the robot in Remote Control mode?)"`,
`"Ctrl-C"`, or a connection-error string otherwise —
`send.py:151-176`/`record.py:215-218` — treat anything but `"program
finished"` as a run that didn't finish cleanly and exclude it from score
comparisons), `loop`, `hz`. Check `stop_reason` before trusting the CSV at
all — a truncated recording will still "work" in `analysis.py` but its
stats won't mean what you think.

**Reminder from §3/§4:** on URSim (`127.0.0.1`), `actual_*` columns come
back zero (perfect simulator) — baseline and optimized will look identical
here. Either point `--robot-ip` at real hardware, or use the
`augment()`-based workaround in `COMMANDS.md`'s last section to overlay the
distill model's *predicted* gap for a sanity check that's still not a
measurement.

### 9.6 `analysis.py` / `plot_target_actual.py` — read the recording

```bash
python analysis.py --csv results/triangle_baseline.csv --joint 1
python analysis.py --csv results/triangle_optimized.csv --joint 1
```

Every run: computes per-joint stats over the whole recording and saves
plots — read-only, doesn't touch models or write anything but this one
`results/<dt>_analysis_<name>/` folder.

**File:** `log.json`'s `per_joint` list — `gap_rms_A`/`gap_max_A` (current
tracking error, lower is better) and `pos_err_mrad` (position error, only
meaningful in path mode). `all_joints_overview.png` gives all six joints at
a glance; `current_<joint>.png` is the one requested joint's target-vs-
actual trace over time.

**This is the step that actually answers "is the agent better," not 9.4:**
run it on the baseline and optimized `send.py` recordings (real hardware,
not URSim — see 9.5's reminder) and compare:
- `gap_rms_A`/`gap_max_A` down on the optimized run → real improvement.
- The *measured* percentage change here vs. `run.py`'s *predicted*
  `improvement.score_pct` (9.4) — close agreement means the distill model
  is trustworthy for this region; a large mismatch means it isn't, and no
  amount of retraining the agent (9.3) will fix that until the distill
  model (9.2) is retrained on data covering the region in question.
- For a targeted joint/channel comparison beyond `analysis.py`'s
  `current`-only plot, use `plot_target_actual.py --csv ... --value <q|qd|
  current|TCP_pose|TCP_speed> --joint <0-5|x|y|z|rx|ry|rz>` (CLAUDE.md
  change log, 2026-08-12).

## Appendix: UR robot programming basics for this project

Background to read the rest of this doc and the code without stumbling — not
UR's full manual, just what this repo actually touches.

### URScript

UR's native robot language — a Python-like DSL the controller interprets
directly. `scripts/*.script` are plain URScript text files; `send.py` doesn't
compile anything, it wraps the file's statements in a `def prog(): ... end`
block (`wrap_program`, `send.py:56-100`) and streams the text over a socket.

- `movej(pose, a=acc, v=vel, r=blend)` — joint-space move: interpolates all
  six joints together from the current angles to `pose` (a `[q0..q5]` array,
  radians). Path shape is whatever joint-space interpolation produces, not a
  straight line in Cartesian space (that's `movel`). `a`/`v` bound a
  trapezoidal accel/cruise/decel profile; `r` optionally blends into the next
  move instead of stopping fully.
- `sleep(s)` — pause; every script in `scripts/` sleeps ~0.75-0.8s between
  moves so the arm settles before the next one — that settling window is
  literally the interval the case brief's peak/RMS vibration metrics are
  meant to be measured over (`common.Segment`'s `i1→i2`, per §6
  Prerequisite).
- `write_output_float_register(i, value)` — the only way a running program
  talks back out to the world; `record.py --float-register i name` reads it
  as an extra RTDE field, which is how `vel`/`acc` end up as CSV columns
  (`triangle.script:20-21`).
- Variables assigned at top level (not inside `def`) get hoisted by
  `wrap_program` to initialize once before a `--loop N` wrapper, and must
  stay non-`global` — URScript rejects a `global` declaration inside a
  `while` loop, which is why every script's `vel = 999` / `acc = 999` are
  plain assignments, not `global vel = 999`.

### Units and numbers that look wrong but aren't

- Joint angles/velocities: radians, rad/s (`q0..q5`, base→wrist3).
- TCP pose: 6 numbers — position (m) + rotation as an axis-angle vector
  `rx,ry,rz` (radians), not a quaternion, not roll/pitch/yaw.
- `movej`'s native `a`/`v` are rad/s²/rad/s. A script like `triangle.script`
  passing `v=999` isn't asking for 999 rad/s (~570 rev/s) — the controller
  silently clamps any speed above the joint's physical limit
  (`MAX_JOINT_SPEED = π rad/s`, `dynamics.py:31`), so `999` just means "as
  fast as the robot allows," a deliberate max-speed baseline. The pipeline's
  own code (`dynamics.py`, `train_rla.py`) then treats these same raw
  numbers as a **deg/s-scale convention** for its own bookkeeping
  (`VEL_BOUNDS = (20, 180)`, `vel_deg * DEG2RAD` before clamping) — two
  different unit conventions for the same-looking number, worth keeping
  straight when reading `utils.get_param`/`set_param` vs. `dynamics.py`.

### Talking to the controller: the interfaces this repo uses

No `ur_rtde` dependency anywhere — every socket protocol here is hand-rolled
from the official UR guides, stdlib only.

| Port  | Interface            | Used for                                        | Where in this repo |
|-------|-----------------------|--------------------------------------------------|---|
| 30001 | Primary               | upload/run a script                              | `case 1/ur_client.py` |
| 30002 | Secondary             | upload/run a script                              | `case 2/send.py` (`SCRIPT_PORT`) |
| 30004 | RTDE                  | 125Hz state stream (read-only) — target/actual q, qd, current, TCP pose, etc. | `case 2/record.py`, `case 1/ur_client.py` |
| 80    | Web UI (PolyScope X)  | manual control, power on / release brakes        | — |
| 29999 | Dashboard             | not used — PolyScope X doesn't speak the classic dashboard protocol | — |

RTDE itself is a small binary protocol: negotiate a protocol version,
register a "recipe" of fields you want, then read fixed-size packed
doubles/vectors at the controller's rate. `record.py`'s `RECIPE`
(`record.py:56-68`) is exactly that negotiated field list — the same order
the CSV's `target_*`/`actual_*` columns come from.

### Remote Control, and why a script can "run" and do nothing

The controller/URSim must be in **Remote Control** mode for a script pushed
over 30001/30002 to actually execute — in Local Control it accepts the TCP
connection and the bytes, and does nothing. This is the single most common
way this pipeline silently produces empty/identical recordings (`send.py`'s
`_done_check` times out at ~5s with "program never started," which
`train_rla.py`'s `collect_moves` doesn't surface — it just reports `0
segments pooled`, flagged in §5).

### URSim specifics (this repo's simulator)

- Boots **powered off**, same as real hardware: open `http://localhost`,
  power on, release brakes until the state reads RUNNING/Active — one-time
  per container boot (`simulation environment/README.md`).
- Defaults to a `UR10`, not a `UR10e` (`ROBOT_TYPE` in `docker-compose.yml`)
  — `utils.UR10e`'s DH/mass/inertia parameters are UR10e-specific, so
  URSim's *physics* isn't quite the robot the kinematics model assumes; this
  only matters for interpreting simulated dynamics, not the
  target-trajectory geometry the pipeline actually relies on URSim for.
- Has **zero reality gap by construction** (§3/§5) — useful for exercising
  the pipeline's plumbing, useless for anything gap-related; that's what
  `data/test-*.csv` (real hardware) and `augment()`'s injected predictions
  are for.
