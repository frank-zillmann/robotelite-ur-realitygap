# RL Agent Review — `train_rla.py` / `run.py`

Living document, same rules as `ModelReview.md`. **Update this file every time
`train_rla.py`, `run.py`, or the objective/metric/environment they use
changes** — add a dated entry to the Changelog at the bottom, and edit the
sections above it so they always describe the *current* state.

This file is the RL-specific counterpart to `ModelReview.md`, which stays
focused on the distillation model itself (`train_distillation_model.py`,
`metrics.py`). Don't duplicate that content here — link to it.

## 1. What's being optimized

One `movej` (or one re-timed path) at a time: given a commanded move the
robot already performs in a script, pick the `(vel, acc)` — or, in path
mode, a re-timed speed profile — that minimizes a weighted combination of
**predicted position-tracking gap** (vibration/overshoot) and **cycle time**
(how long the move takes). This is the case brief's actual ask: move as fast
as possible while minimizing vibration.

Two modes, `--mode params`/`--mode path`:

| | `GapEnv` (`params`, default) | `PathEnv` (`path`) |
|---|---|---|
| geometry | straight joint-space line, start→dest | the move's *recorded* joint trajectory (path preserved) |
| action | `(vel, acc)`, deg/s and deg/s² | `(accel_frac, decel_frac, servoj dt)` — re-times the same path |
| score reduction | RMS over the settling window's rows | max over the settling window's rows (peak, not average) |

Both are **one-step contextual bandits**, not multi-step episodes:
`reset()` picks a random move from the script(s), `step(action)` scores that
one candidate and ends the episode immediately. There's no sequential
decision-making across a trajectory — each move is scored independently.

## 2. The environment (`_MoveEnv`, `GapEnv`, `PathEnv`)

- **Observation**: `[start, dest, dist, joint one-hot(6)]` for the move,
  plus the *baseline* (the move's own recorded speed) aggregated value of
  every channel the distill model predicts, plus the baseline `score` —
  built by `observe()`. The agent sees "what does this move look like and
  how did it do at its original speed" before picking an action.
- **Action bounds** (`GapEnv`): `VEL_BOUNDS = (20, 180)` deg/s,
  `ACC_BOUNDS = (40, 600)` deg/s² — the upper end is chosen so a `movej`
  speed above the joint limit (~180 deg/s = π rad/s) clamps anyway, so the
  useful range stays below where clamping would make the action meaningless.
- **Scoring a candidate** (`_candidate`/`_score_settled`/`evaluate`): places
  the move's geometry at the agent's chosen speed profile, appends
  `SETTLE_S` seconds held at the destination (see §2a), asks
  `dynamics.Dynamics` for the commanded trajectory (target_q/qd/current)
  over the whole extended array, asks the **distill model** to predict
  `actual_q`, then the **metric** scores the predicted gap — but only the
  appended settling-window rows are returned/reduced, matching the case
  brief's "for t in the settling window" definitions. Nothing here touches
  a real robot — every score during training is a prediction from the
  distill model, evaluated on a synthetic candidate trajectory built by
  `dynamics.py`, not a measurement.
- **Reward**: `-OBJECTIVE(score, cycle_time)` (params) or
  `-PATH_OBJECTIVE(max_score, cycle_time)` (path) — see §4.

## 2a. The settling window (`_score_settled`) — fixed 2026-08-19

**The gap.** The case brief's peak/RMS formulas are explicit:
`peak = max(|actual(t) - target(t)|)` and `rms = sqrt(mean(...))`, both **"for
t in the settling window"** — the period *after* a move arrives, where a real
robot rings/overshoots before settling. Before this fix, a candidate
trajectory in `_MoveEnv` ended the instant its commanded motion stopped: the
last row scored was the destination-arrival row itself, and there was no
row after it at all. `_aggregate()` (RMS, `GapEnv`) and `.max()` (peak,
`PathEnv`) were therefore computed **over the motion**, not the settling
window — a structural gap, not a documentation caveat: even a model that
could predict a perfect ring had no post-stop rows to show it on.

**The fix.** `_MoveEnv._score_settled(move, q, s, dt, vel_deg, acc_deg)`
appends `SETTLE_S` seconds of the commanded position held still at the
move's destination (`q[-1]`) after the candidate's own trajectory, scores
the *whole* extended array through `dynamics.Dynamics.frame` and
`evaluate()` as usual, and returns only the appended rows' scores — never
the motion's. `_candidate()` (called by both `GapEnv.score()` and
`PathEnv.score()`) now delegates to it, so both envs' `.score()` picked up
the fix with **no change to `GapEnv.score()`/`PathEnv.score()` themselves**
— they already just reduce whatever `_candidate()` returns.
`PathEnv.baseline()` built its frame directly rather than through
`_candidate()`, so it needed its own one-line fix (call `_score_settled`
too) — otherwise the "before" number in `baseline_vs_optimized.png` would be
a peak-over-the-motion number compared against an "after" number that's
peak-over-the-settling-window, an apples-to-oranges comparison hiding
inside a chart that looks like a fair one.

The held-still portion is still run through `Dynamics.frame` (not
hand-set to zero) so `target_qd`/`target_qdd` come out of `np.gradient` as a
continuous decay through the stop rather than an abrupt jump, giving
`current()` a physically sensible qdd≈0 (holding torque = gravity torque)
for those rows; `s` is held at its final value throughout the appended
rows so the cached per-move pose terms (`Dynamics.prepare`) stay at the
destination pose, matching a robot that has arrived and stopped.

**`SETTLE_S = 0.75`, measured not guessed**: `bronze_tier/segment_stats.csv`
has real `i1`→`i2` (motion-stop → next-move-start) windows for 2497 real
segments; `(i2 - i1) * dt` has median 0.749s. The distribution is bimodal —
about a quarter of segments are back-to-back moves with an `i1`→`i2` near
zero, the rest cluster tightly around 0.75s (scripts that pause between
moves) — 0.75s was chosen to reflect the "there is a real settle period"
regime, since that's the one actually worth scoring for vibration.

**Verified offline** (`data/test-4.csv`, `models/distill.pkl`, no live
robot/URSim): `SETTLE_S / dt` produced exactly the expected number of
settle-window rows (96 at this recording's `dt≈0.00783s`); `cycle_time`
(computed from `len(s) * dt`, the active-motion length only) was confirmed
unaffected by the settle-window extension, as it should be — cycle time is
about how long the move itself takes, not the settle period after it;
`GapEnv`/`PathEnv` `.step()` and a 200-step `PPO.learn()` both ran without
exceptions on both envs.

**Not yet done**: no re-run of the real 20,000-step training campaign from
§5 with this fix — the `best_score=0.000423`/`final_score_mean=0.001356`
numbers recorded there predate it and are settle-window scores now, not
motion scores, so they are **not directly comparable** to a future run's
numbers even though the metric class and objective weights haven't
changed. A fresh training run is needed before drawing any new
before/after conclusions.

## 3. The metric: `PositionGapMetric`, not `CurrentGapMetric`

Wired in 2026-08-19 (previously `CurrentGapMetric`, which needed
`actual_current` — a channel this project's distill models no longer
predict; see `ModelReview.md` §1). Two-line change:
`train_rla.py:546`/`run.py:213`, `metric = PositionGapMetric()`. Full
reasoning for *why* position over current lives in `ModelReview.md` §1 —
not repeated here.

One thing worth restating here specifically: `PositionGapMetric.per_row()`
returns `sum(|actual_q - target_q|)` across all 6 joints, in **radians**.
`_aggregate()` (RMS, `GapEnv`) or `.max()` (`PathEnv`) reduce that per-row
array to the single `score` the objective uses — these two reductions are
exactly the case brief's `rms`/`peak` formulas, applied to this metric's
per-row output (see `metrics.py`'s own docstring).

## 4. The objective weights — measured, not assumed

`OBJECTIVE = SCORE_WEIGHT * score + CYCLE_WEIGHT * cycle_time` (and the path
equivalent) were `1.0/1.0`, tuned for `CurrentGapMetric` whose amps-scale
score happened to be roughly comparable to `cycle_time` in seconds.
`PositionGapMetric`'s score is in radians — nowhere near that scale, and
leaving the weights at 1.0 would make the objective ≈ `cycle_time` alone,
silently dropping the vibration term.

**Measured directly** (not guessed) against what `GapEnv.score()`/
`PathEnv.score()` actually compute — built the envs on a real recording
(`data/test-4.csv`, no live robot needed for this), sampled real actions,
read off the true score/cycle_time magnitudes:

| env | raw score (mean) | raw cycle_time (mean) | ratio | weight chosen |
|---|---|---|---|---|
| `GapEnv` | 0.00107 rad | 2.71 s | ~2524x | `SCORE_WEIGHT = 2500.0` |
| `PathEnv` | 0.03002 rad | 0.6085 s | ~20x | `PATH_SCORE_WEIGHT = 20.0` |

`CYCLE_WEIGHT`/`PATH_CYCLE_WEIGHT` stay at `1.0` — the fixed reference the
score weight is calibrated against.

**A real mistake made and caught while doing this**: the first `PathEnv`
measurement used `PathEnv.baseline()` (times the *full recorded trajectory*
at its own resolution, ~2.97 s typical) instead of `PathEnv.score()` (what
`.step()`/training actually call — `PATH_ROWS(50) × the agent's chosen
servoj dt`, more like ~0.6 s). That gave a wrong ~1929x ratio and a weight
overshooting by ~80x — verified after the fact by stepping the real env with
the chosen weight and checking both objective terms were actually
comparable in magnitude (they weren't, until fixed). Lesson: measure against
the literal call path the training loop uses, not a plausible-looking proxy
function with a similar name.

**Verified after the fix** (stepping both envs with random actions,
`SCORE_WEIGHT/PATH_SCORE_WEIGHT`-weighted score term vs. `CYCLE_WEIGHT`-
weighted cycle term): `GapEnv` ratio 0.93, `PathEnv` ratio 0.46 — both within
a 0.2–5x "neither term is negligible" band.

## 4a. The calibration didn't survive contact with a trained policy (2026-08-20)

The 0.93 ratio above is a **random-action** measurement, taken before any
training — it describes the *starting* distribution of (score, cycle_time)
a fresh policy sees, not what a *trained* one settles into. New tool
`tune_reward_weights.py` (repo root: `case 2/`) automates that same
random-sampling calibration, and adds a second mode it didn't have before:
checking balance against a **completed run's own episodes** (needs
`train_rla.py`'s `log_training_run` to save one — see §6, added same day).

Checked against a real 20,000-step run (`results/2026-08-20_12-57-02_rla_params/`,
20,480 episodes — trained *with* lag-tap features in the distill model, see
below; a near-identical earlier run, `2026-08-20_12-31-40_rla_params/`,
predates the `episodes.csv` logging this check needs, so this slightly
later re-run is the one actually used here):

```
python tune_reward_weights.py --episodes results/2026-08-20_12-57-02_rla_params/episodes.csv --mode params --target-ratio 1.0
  mean score:      0.000158 rad
  mean cycle_time: 1.715 s
  current SCORE_WEIGHT=2500  ->  weighted (score:cycle) = 0.231:1
  target ratio 1:1  ->  recommended SCORE_WEIGHT = 10838.5
```

At `SCORE_WEIGHT=2500`, the *trained* policy's own score/cycle_time pairs
weight to **0.231:1**, not the ~1:1 the pre-training calibration found —
cycle_time dominated the gradient for the entire run. This lines up exactly
with that run's `training_curve.png`: cycle_time's rolling mean dropped
steadily (~3.5s → ~1.2s) while score's barely moved (and even ticked up
slightly) — the agent had far more reward available from getting faster
than from getting smoother, so that's what it optimized.

**Retargeted at 1:1** (score and cycle_time weighted equally — no
deliberate favoring of either; a `2:1` version favoring vibration reduction
was tried briefly first, see git history, then explicitly reverted to `1:1`):
`SCORE_WEIGHT` changed **2500.0 → 10800.0** (rounded from the tool's exact
10838.5 recommendation against the same episode data, same rounding
convention the original 2524→2500 calibration used). `PATH_SCORE_WEIGHT`
is **not** re-checked here — no `path`-mode training run has happened yet
to measure its own drift against; assume it needs the same treatment
before trusting it.

**Why this matters alongside §5's "no lag features" explanation**: the run
being recalibrated against here was trained with the lag-tap-enabled
distill model (`qdd_lag4/16/64`, ported into this branch's
`train_distillation_model.py` the same day — see Changelog), *not* the
no-history model §5's original discussion assumed. Score still barely
moved despite the model now being able to represent the settle-window ring
in principle — strong evidence the reward-weight imbalance was doing (at
least) as much to suppress the score signal as the missing-lag-features gap
was. Both are real; a fresh run is needed with both fixes in place before
attributing "flat score" to either one alone (see §7).

## 5. Verification so far

**Offline, no live robot/URSim** (not available in this dev environment):
built `GapEnv`/`PathEnv` directly on `data/test-4.csv` (`_MoveEnv` only needs
a `Recording`, doesn't care whether it came from `collect_moves` or a CSV
already on disk), stepped both with random actions, confirmed the objective
balance above, and ran a short `PPO.learn()` (200 timesteps) on both without
exceptions.

**A real 20,000-step training run happened** (2026-08-19,
`results/2026-08-19_16-02-34_rla_params/`, 3 scripts —
`shoulder_swing`/`vertical_swing`/`horizontal_swing`, 20,480 episodes,
`best_score=0.000423`, `final_score_mean=0.001356`,
`best_cycle_time=0.570s`). Reading its `training_curve.png`: **cycle_time's
rolling mean dropped steadily (~3s → ~2s) while score's rolling mean stayed
roughly flat (~0.0013 rad throughout, noisy, no clear trend)**. The agent
learned to go faster; it did not learn to reduce predicted vibration.

*(Superseded by §2a as of 2026-08-19: this run predates the settling-window
fix, so its `score` numbers are peak/RMS over the motion, not the settling
window the case brief asks for. Kept here as the historical record of the
"score is flat, cycle time drops" finding, which is still the relevant
question to re-check once §2a's fix has a fresh training run behind it.)*

**A second real 20,000-step run happened** (2026-08-20,
`results/2026-08-20_12-31-40_rla_params/`, same 3 scripts, 20,480 episodes,
`best_score=0.0000602`, `final_score_mean=0.000164`, `best_cycle_time=0.570s`).
This one *does* have the settling-window fix, *and* was trained against the
lag-tap-enabled distill model (`qdd_lag4/16/64`, ported into this branch's
`train_distillation_model.py` the same day so it could load the
`PerJointTreeModel` pickle copied in from `apostolosDistillModel` — see
Changelog), *and* the company-provided UR5e physics files (`dynamics.py`/
`utils.py` swap, same day). Its `training_curve.png` shows the identical
qualitative pattern the superseded run above did: cycle_time's rolling mean
dropped steadily (~3.5s → ~1.2s), score's stayed essentially flat (even
ticked up slightly). Repeating with a model that *can* represent history and
a metric scored over the right window and still seeing a flat score ruled
out "wrong window" and weakened "model can't represent it" as sole
explanations — which is what led to checking the objective weights
directly (§4a) and finding the real driver: at the old `SCORE_WEIGHT=2500`,
this run's own score/cycle_time distribution weighted to 0.231:1, not ~1:1.
This run predates the `SCORE_WEIGHT=10800` recalibration — not yet re-run
with it.

This is consistent with, not contrary to, `ModelReview.md`'s documented
limitation about lag/history features — both that gap and the objective
imbalance (§4a) push in the same direction (suppress the score signal the
agent has to climb), and the second run above is evidence the weight
imbalance alone is sufficient to reproduce the flat-score pattern even with
lag features present. Disentangling how much each one individually
contributed needs a fresh run with the recalibrated weight; see §7.

**Not yet done**: no run against a real robot or a fresh live URSim
connection from this session; no comparison of `params` vs `path` mode
results; no PPO hyperparameter tuning (default `MlpPolicy` throughout); no
check of whether training for longer than 20k steps keeps improving cycle
time or plateaus; no run yet with the recalibrated `SCORE_WEIGHT=10800`.

## 6. Plots each run produces

| file | from | shows |
|---|---|---|
| `training_curve.png` | `train_rla.py`'s `log_training_run` | score / cycle_time / reward vs. timestep, raw (faint) + rolling mean + best-so-far reference line, one subplot each |
| `episodes.csv` (not a plot, 2026-08-20) | `train_rla.py`'s `log_training_run` | full per-episode `timestep`/`reward`/`score`/`cycle_time` — previously only aggregated into `log.json`'s summary; now saved raw so a completed run's actual distribution can be checked (`tune_reward_weights.py --episodes ...`) without retraining |
| `baseline_vs_optimized.png` | `run.py`'s `run_params`/`run_path` (new, 2026-08-19) | mean score and cycle_time, baseline (script's original vel/acc) vs. the trained agent's choice, side by side with %-change titles |
| `weight_sensitivity.png` (2026-08-20) | `tune_reward_weights.py` | score term's share of mean total weighted cost vs. candidate `SCORE_WEIGHT` (log scale) — where the current and recommended weights sit relative to a target balance |
| `score_vs_cycle.png` (2026-08-20) | `tune_reward_weights.py` | raw score-vs-cycle_time samples with iso-cost lines for the current and recommended weights overlaid — shows concretely how weight choice reshapes which points count as "good" |

Both use `ur_style` (added 2026-08-19, replacing hardcoded `steelblue`/
`darkorange`/`purple`/`green`) — same palette/outline/grid convention as
`train_distillation_model.py`'s plots, so a training-curve figure and a
distill-model figure read as the same visual family in a report.
`training_curve.png`'s axis labels carry a short parenthetical (units +
which direction is better) since the plot is meant to be read on its own,
not only alongside this file.

`baseline_vs_optimized.png` is the more important of the two for "did this
work" — the training curve shows how optimization progressed over
timesteps, this one shows what the final agent is worth against the
script's original fixed vel/acc, which is the number the case brief actually
asks for. Not yet generated from a real trained agent as of this writing
(needs `run.py` run against the agent from §5's real training run).

## 7. Not yet done, known limitations

- **No training run yet with the recalibrated `SCORE_WEIGHT=10800`** (§4a) —
  the immediate next step. Both real runs so far (§5) used the old
  `SCORE_WEIGHT=2500`, under which cycle_time dominated the objective
  ~4.3:1 rather than the intended 1:1 balance; a fresh run is needed to see
  whether score actually improves once the agent has real gradient on it.
- Now that lag features (below) and the weight recalibration are both in
  place, disentangling how much each contributed to the previous flat-score
  runs needs a run with both fixes active — not done yet (§4a/§5).
- ~~No lag/history features in the distill model~~ — **done** 2026-08-20:
  `qdd_lag4/16/64` ported into this branch's `train_distillation_model.py`
  (was 8 features, now 11, matching `apostolosDistillModel`'s model — this
  was also *required* just to load the `PerJointTreeModel` pickle copied in
  from that branch, which crashed with a feature-count `ValueError`
  otherwise). Effect on RL training still unclear — see §5's second real run,
  which had this fix and still showed a flat score, now attributed mainly to
  the weight imbalance instead (§4a).
- `dynamics.py`/`utils.py` swapped to company-provided UR5e physics files
  2026-08-20 (see `ModelReview.md`-side writeup for the full compatibility
  investigation) — DH/mass/COM are byte-identical to the previous values
  (so `gravity()`/the distill model's `gravity_torque` feature are
  numerically unaffected, confirmed directly), but the inertia tensors are
  now fully populated (previously zero for links 1-5) which changes
  `mass_matrix()` and therefore `dynamics.py`'s simulated `target_current`
  for every RL candidate. Verified the swap doesn't crash/NaN, but **no
  training run has happened with the new physics to check whether the more
  complete inertia model changes what the agent learns** — both real runs
  in §5 used it (it landed before them), so there's no before/after
  comparison to point to either.
- Leave-one-file-out validation of the distill model itself is still
  unverified (`ModelReview.md` §7 item 4) — any RL result inherits that
  uncertainty.
- No real-robot or fresh-URSim validation of an optimized script from this
  pipeline (`send.py --script ...optimized.script`) — everything so far is
  the distill model's prediction, one layer removed from ground truth.
- `path` mode is unexercised beyond the offline PPO smoke test — no real
  training run or baseline-vs-optimized comparison yet, and its
  `PATH_SCORE_WEIGHT=20.0` hasn't been rechecked against a trained policy
  the way `SCORE_WEIGHT` just was (§4a) — likely has the same kind of drift.
- PPO hyperparameters are stable-baselines3 defaults throughout — untuned.

## Changelog

- **2026-08-20** — Recalibrated `SCORE_WEIGHT` against a *trained* policy's
  actual score/cycle_time distribution instead of only the pre-training
  random-action baseline (§4a). New tool `case 2/tune_reward_weights.py`:
  samples random actions through the live `GapEnv`/`PathEnv` (or reuses a
  completed run's `episodes.csv`, new below), prints the current weight's
  real balance ratio and a recommended weight for a target ratio, and saves
  two plots (`weight_sensitivity.png`, `score_vs_cycle.png` — see §6).
  Checked against `results/2026-08-20_12-57-02_rla_params/`'s real 20,480
  episodes: `SCORE_WEIGHT=2500` weighted to 0.231:1 (score:cycle), not
  ~1:1 — cycle_time dominated the whole run. First retargeted at 2:1
  (favoring vibration reduction): `SCORE_WEIGHT` 2500.0 → 21700.0; revised
  same day back to an explicit **1:1** (score and cycle_time weighted
  equally, no favoring either): `SCORE_WEIGHT` → **10800.0** (tool's exact
  recommendation 10838.5, rounded). `PATH_SCORE_WEIGHT` left
  unchanged/unchecked (§7). `train_rla.py`'s
  `log_training_run` now also saves `episodes.csv` (full per-episode
  timestep/reward/score/cycle_time — previously only aggregated into
  `log.json`), which is what makes the post-hoc check possible without
  retraining.
- **2026-08-20** — Ported the `qdd_lag4/16/64` lag-tap feature machinery
  into this branch's `train_distillation_model.py` (`PerJointPositionModel`/
  `PerJointTreeModel`: 8 features → 11) to match `apostolosDistillModel`'s
  version — required to load the `PerJointTreeModel` pickle copied in as
  the new `models/distill.pkl`, which was crashing `train_rla.py` with
  `ValueError: X has 8 features, but HistGradientBoostingRegressor is
  expecting 11 features`. Verified against the actual pickle (not just a
  freshly-trained stand-in): loads and predicts correctly post-fix. See
  §5/§7 for how this interacts with the weight-recalibration finding above
  — both real training runs in §5 predate one or both of these fixes.
- **2026-08-20** — Swapped `case 2/dynamics.py`/`case 2/utils.py` for
  company-provided UR5e physics files (more complete inertia tensors —
  previously zero for links 1-5, now fully populated from
  `Universal_Robots_ROS2_Description`). DH/mass/COM are byte-identical to
  the previous values (verified numerically: `gravity()`/`gravity_batch()`
  bit-identical across 500 random poses), so the distill model's
  `gravity_torque` feature is unaffected and needed no refit. `mass_matrix()`
  does change (inertia now contributes), which changes `dynamics.py`'s
  simulated `target_current` for every RL candidate — the actual physics
  `train_rla.py` trains against, not just a feature. Also added a missing
  `gravity_batch()` (this project's own vectorized addition, absent from
  the company files) to the new `utils.py` before swapping. Both real
  training runs in §5 happened after this swap; no before/after comparison
  exists yet (§7). Also added periodic progress printing to
  `_TrainCallback`/`train_ppo` (`--print-every`, default 5s) — training
  previously printed nothing at all between the start and end of a
  20,000-step run (`PPO(..., verbose=0)`), which looked indistinguishable
  from hanging.
- **2026-08-19** — Gold-tier gap fix: candidates now score a real
  post-stop **settling window** (§2a), not the motion itself, matching the
  case brief's `peak`/`rms` "for t in the settling window" requirement.
  Added `_MoveEnv._score_settled` (appends `SETTLE_S=0.75s` — measured from
  `bronze_tier/segment_stats.csv`'s real `i1`→`i2` windows — of held-still
  destination position, scores only the appended rows) and routed
  `_candidate()` through it, so `GapEnv.score()`/`PathEnv.score()` picked up
  the fix with no changes of their own; fixed `PathEnv.baseline()`
  separately (it built its frame directly, bypassing `_candidate()`) so
  baseline-vs-optimized stays an apples-to-apples comparison. Verified
  offline against `data/test-4.csv`: correct settle-row count, `cycle_time`
  unaffected, both envs step/train without exceptions. The real training
  run in §5 predates this fix and needs re-running before its numbers are
  trusted again.
- **2026-08-19** — Created this file. Documents the `CurrentGapMetric` →
  `PositionGapMetric` wiring, the measured `SCORE_WEIGHT`/`PATH_SCORE_WEIGHT`
  recalibration (including the `PathEnv.baseline()` vs. `.score()`
  measurement mistake caught and fixed), the offline verification
  methodology, the real 20,000-step training run and its score-flat/
  cycle-time-down finding, the new `ur_style` plot styling, and the new
  `baseline_vs_optimized.png` plot. See `ModelReview.md`'s Changelog for the
  same-day distill-model-side entries this depends on.
