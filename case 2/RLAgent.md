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
| score reduction | RMS over the move's rows | max over the move's rows (peak, not average) |

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
- **Scoring a candidate** (`_candidate`/`evaluate`): places the move's
  geometry at the agent's chosen speed profile, asks `dynamics.Dynamics` for
  the commanded trajectory (target_q/qd/current), asks the **distill model**
  to predict `actual_q` for it, then the **metric** scores the predicted gap.
  Nothing here touches a real robot — every score during training is a
  prediction from the distill model, evaluated on a synthetic candidate
  trajectory built by `dynamics.py`, not a measurement.
- **Reward**: `-OBJECTIVE(score, cycle_time)` (params) or
  `-PATH_OBJECTIVE(max_score, cycle_time)` (path) — see §4.

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

- No lag/history features in the distill model → the agent can't be
  expected to reduce vibration meaningfully yet (§5). This is the
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

- **2026-08-19** — Created this file. Documents the `CurrentGapMetric` →
  `PositionGapMetric` wiring, the measured `SCORE_WEIGHT`/`PATH_SCORE_WEIGHT`
  recalibration (including the `PathEnv.baseline()` vs. `.score()`
  measurement mistake caught and fixed), the offline verification
  methodology, the real 20,000-step training run and its score-flat/
  cycle-time-down finding, the new `ur_style` plot styling, and the new
  `baseline_vs_optimized.png` plot. See `ModelReview.md`'s Changelog for the
  same-day distill-model-side entries this depends on.
