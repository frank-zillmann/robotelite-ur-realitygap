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
`actual_current`-only distill model, row-level held-out split, no
model/agent provenance on the resulting recording) is covered where it's
actionable, in §5-§6 — not repeated here.

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

1. **Run-level held-out split for the distill model** (§6 Silver). Train on
   `test-1,2,4,5`, evaluate entirely on `test-3` — a full real run the model
   never saw, not just held-out rows from runs it partly trained on (today's
   split is row-level, `--holdout 0.2`, and optimistic per §5). Report
   per-joint RMSE/R², since pooled RMSE hides a bad fit on small joints
   (wrist current is ~1/20,000th of base current — §5).
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
