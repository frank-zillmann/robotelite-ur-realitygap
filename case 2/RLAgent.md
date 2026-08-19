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

This is consistent with, not contrary to, `ModelReview.md`'s documented
limitation: the distill model has no lag/history features, so it can't
represent the settle-window ring — the actual thing "vibration" refers to
physically. If the model's score barely changes with the trajectory shape in
the dimension that matters, the agent has nothing to climb on that axis and
will optimize the term it *can* move (speed). Worth treating as a real,
data-backed prediction of what adding lag features (`ModelReview.md` §7
item 2) should fix, not just a documentation caveat.

**Not yet done**: no run against a real robot or a fresh live URSim
connection from this session; no comparison of `params` vs `path` mode
results; no PPO hyperparameter tuning (default `MlpPolicy` throughout); no
check of whether training for longer than 20k steps keeps improving cycle
time or plateaus.

## 6. Plots each run produces

| file | from | shows |
|---|---|---|
| `training_curve.png` | `train_rla.py`'s `log_training_run` | score / cycle_time / reward vs. timestep, raw (faint) + rolling mean + best-so-far reference line, one subplot each |
| `baseline_vs_optimized.png` | `run.py`'s `run_params`/`run_path` (new, 2026-08-19) | mean score and cycle_time, baseline (script's original vel/acc) vs. the trained agent's choice, side by side with %-change titles |

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

- No fresh training run since the settling-window fix (§2a) — the real run
  recorded in §5 predates it and its `score` numbers are not the settling-
  window numbers the case brief asks for. Re-running training and
  `run.py`'s `baseline_vs_optimized.png` is the immediate next step.
- No lag/history features in the distill model → the agent can't be
  expected to reduce vibration meaningfully yet (§5), and §2a's fix doesn't
  change this: appending a held-still tail gives the metric real settling-
  window *rows* to score, but the distill model still predicts each row
  from that row's own instantaneous features, with no memory of the
  approach that preceded it — so it still can't represent a decaying ring,
  it can now just be *asked* about the right time window. This is the
  highest-priority fix, and it lives in `train_distillation_model.py`, not
  here — see `ModelReview.md` §7 item 2.
- Leave-one-file-out validation of the distill model itself is still
  unverified (`ModelReview.md` §7 item 4) — any RL result inherits that
  uncertainty.
- No real-robot or fresh-URSim validation of an optimized script from this
  pipeline (`send.py --script ...optimized.script`) — everything so far is
  the distill model's prediction, one layer removed from ground truth.
- `path` mode is unexercised beyond the offline PPO smoke test — no real
  training run or baseline-vs-optimized comparison yet.
- PPO hyperparameters are stable-baselines3 defaults throughout — untuned.

## Changelog

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
