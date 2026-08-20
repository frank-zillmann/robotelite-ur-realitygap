# RL Agent Review — `train_rla.py` / `run.py`

Living document, same rules as `ModelReview.md`. **Update this file every time
`train_rla.py`, `run.py`, or the objective/metric/environment they use
changes** — add a dated entry to the Changelog at the bottom, and edit the
sections above it so they always describe the *current* state.

This file is the RL-specific counterpart to `ModelReview.md`, which stays
focused on the distillation model itself (`train_distillation_model.py`,
`metrics.py`). Don't duplicate that content here — link to it.

## 0. Plain-language summary — for slides

The rest of this document is the working log. This section is the pitch:
what the RL policy actually does, and why it was switched from `PPO` to
`SAC` (§8), without requiring the reader to already know what either of
those is.

**One sentence**: for each move the robot needs to make, an agent picks how
fast to go and how hard to accelerate, trying to get there quickly without
shaking or overshooting when it arrives — trained entirely against a
simulated prediction of the robot, never the real machine.

### The problem, in an analogy

Picture packing many boxes on a conveyor belt as fast as possible without
anything rattling loose inside. For each box, you choose a packing speed:
too fast and things shift (vibration/overshoot), too slow and you waste
time. Critically, **box 7's result doesn't depend on how you packed box
6** — every box is its own, independent decision, scored purely on its own
outcome. That's exactly this problem's shape: every robot move gets its
own speed decision, with no sequence or memory across moves. This matters
a lot for the algorithm question below.

### How the agent decides, and how it knows if a choice was good

For every move, the agent is shown how far the joint needs to travel, which
joint it is, and how that same move went at its *original* commanded
speed (a baseline for comparison). From that, it picks two numbers: how
fast to go, and how hard to accelerate.

Nothing here touches a real robot. Each candidate speed is scored by a
simulated pipeline: a physics model predicts how the robot would move, the
distilled model (`ModelReview.md`) predicts how far that lands from
perfect (the "reality gap"), and that combines with how long the move took
into one score — weighted so speed and smoothness matter about equally
(§4a). Because it's all a calculation, not a real robot moving, this loop
can run thousands of times, cheaply and safely, to learn from.

### Why the first algorithm (PPO) was swapped out — the studying analogy

The original version used `PPO`, a well-known RL algorithm built for
problems that unfold over *many* steps, where doing well at step 5 depends
on what happened at steps 1-4 — think a video game or a chess match. PPO
studies a batch of practice attempts once, then throws them away before
trying new ones, because in a changing, multi-step world an old lesson
might not apply anymore by the time it's revisited.

This project's problem isn't like that: every move is scored in one shot,
and — since the score comes from a fixed calculation, not a noisy real
robot — **the same choice always gets the exact same score**. There's no
"the world changed since I last tried this" risk. That means a different
family of algorithm, `SAC`, can be used instead: it keeps every attempt
it's ever made in a big notebook (a "replay buffer") and re-reads through
it many times as it gets smarter, instead of discarding each attempt after
one use.

**The analogy**: `PPO` is a student who does one practice exam, marks it,
then throws it away before starting the next — every practice question is
only ever seen once. `SAC` is a student who keeps every practice exam
they've ever done and reviews the whole stack again each night — far more
learning squeezed out of the same amount of practice. Since this project's
"practice answers" never go stale (deterministic scoring), there's no
downside to `SAC`'s approach here the way there might be in a noisier,
ever-changing problem.

### Why we expect this to be better — honestly, not yet measured

The earlier `PPO` run is a real, visible symptom worth showing: the agent
got noticeably faster (cycle time dropped a lot) but barely reduced
vibration at all — a sign it wasn't extracting much signal from its
practice. Switching to `SAC` is a direct, reasoned bet on fixing that,
based on how well the algorithm's design matches this problem's shape —
**not a proven result yet**, since a full training run with `SAC` hasn't
happened. The next concrete step is exactly that: rerun the same training
command and compare the two learning-curve plots side by side (§8).

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
- **Action bounds** (`GapEnv`): `VEL_BOUNDS = (1, 179)` deg/s,
  `ACC_BOUNDS = (1, 200)` deg/s² (updated 2026-08-20 — see Changelog; were
  `(20, 180)`/`(40, 600)`). The vel ceiling is kept just under the user's
  robot's own measured joint speed limit (191 deg/s), not a generic `~180
  deg/s = π rad/s` figure — high enough to stay clear of where `movej`
  would clamp the action to something other than what was chosen, but not
  pinned right against the limit. The acc ceiling was cut 600 → 200 after
  the user reported the real robot feels jerky at the old range: 600 deg/s²
  reaches full `VEL_BOUNDS` speed in ~0.3s (a very sudden torque
  application) vs. ~0.9s at 200 — and a direct `GapEnv.score()` sweep (real
  distill model, a real move) showed score wasn't still climbing by
  250-600 deg/s² either, so there was no measured reason to keep the
  ceiling that high.
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
  `-PATH_OBJECTIVE(max_score, cycle_time)` (path) — see §4/§4a for the
  weights (`SCORE_WEIGHT`/`PATH_SCORE_WEIGHT`) and how they were picked.

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
predict; see `ModelReview.md` §1). Two-line change, one `metric =
PositionGapMetric()` each in `train_rla.py main()` and `run.py main()`
(line numbers not cited here — both files have grown enough since this was
wired that a specific line number would already be stale; grep for
`PositionGapMetric()` instead of trusting a number in this doc). Full
reasoning for *why* position over current lives in `ModelReview.md` §1 —
not repeated here.

One thing worth restating here specifically: `PositionGapMetric.per_row()`
returns `sum(|actual_q - target_q|)` across all 6 joints, in **radians**.
`_aggregate()` (RMS, `GapEnv`) or `.max()` (`PathEnv`) reduce that per-row
array to the single `score` the objective uses — these two reductions are
exactly the case brief's `rms`/`peak` formulas, applied to this metric's
per-row output (see `metrics.py`'s own docstring).

## 4. The objective weights — measured, not assumed

**This section is the original, 2026-08-19 calibration — superseded for
`GapEnv`/`SCORE_WEIGHT` by §4a's 2026-08-20 mean-matched recalibration,
which is itself superseded by §4b's same-day std-matched recalibration. The
code's actual current value is `SCORE_WEIGHT = 80000.0`, not the `2500.0`
this section's table shows below; kept here as the historical record of how
that starting point was reached. `PATH_SCORE_WEIGHT = 20.0` is still
current — neither §4a nor §4b touched `PathEnv`.**

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

## 4b. Mean-matching wasn't enough either — std-matched recalibration (2026-08-20)

A real SAC run was trained with §4a's `SCORE_WEIGHT=10800`
(`results/2026-08-20_14-00-58_rla_params/`, 20,000 episodes). Result:
cycle_time converged fast (~3000 steps) to a good plateau, but **score
stayed essentially flat again** — the exact same symptom §4a was supposed
to fix, even though `tune_reward_weights.py` confirms the *mean* weighted
ratio at that weight is ~1.18:1, right on the 1:1 target.

The reason: **mean-matching balances the two terms' average size, not how
much each one can actually move as the action changes** — and it's the
*change* (the gradient) that PPO/SAC actually climb, not the average.
Measured directly with an unbiased random sample (n=3000, `env.action_space.
sample()` + `env.step()` across the full action range, not a trained
policy's narrower visited range):

```
mean score: 0.0001788 rad   std: 0.0000187
mean cycle: 2.8752 s        std: 1.4890
score range: 0.000123 - 0.000205 rad
cycle range: 0.744 - 8.852 s
```

cycle_time's range/std is roughly **12x** wider than score's. A
mean-balanced reward can still be dominated end-to-end by whichever term
has the bigger spread — which is exactly cycle_time here, and exactly why
score kept flatlining even after §4a.

**Fix**: `tune_reward_weights.py`'s `analyze()` now computes a *second*
recommendation using standard deviation instead of mean:
`recommended_std = target_ratio * cycle_weight * std(cycle_time) / std(score)`.
This is the number that should track "does the agent get comparable
learning signal from both terms," not "are the two terms similar size on
average."

**A methodological trap found and fixed while doing this**: the first
attempt computed the std-matched number from `--episodes
results/2026-08-20_14-00-58_rla_params/episodes.csv` (the completed run's
own data) and got **8,460** — badly wrong. A *converged* policy's episodes
are not a fair sample of the action space: the policy has already learned
to avoid the high-cycle-time region, so its own data has a much narrower
spread (5th–95th percentile cycle_time: 0.75s–1.99s) than what's actually
reachable (0.585s–8.625s doing pure random sampling). Using a trained
policy's own visited range to recalibrate the reward it was trained under
is circular — it will always look more balanced than it is. The unbiased
random sample above (not filtered through any trained policy) gives the
real number: **79,626**, ~9x higher than the circular estimate.

`SCORE_WEIGHT` changed **10800.0 → 80000.0** (rounded from 79,626, same
rounding convention as prior calibrations). `tune_reward_weights.py` now
prints a `[warn]` whenever `--episodes` is used, explaining this trap with
the exact numbers, and its module docstring documents the same finding so
it isn't rediscovered as a "mistake" later. `PATH_SCORE_WEIGHT` is
unaffected — no path-mode training run exists to check it against.

**Not yet verified by a training run.** Same honesty standard as every
other weight change in this doc: the actual test is whether a fresh SAC
run with `SCORE_WEIGHT=80000` finally moves score, not just cycle_time.
That run hasn't happened yet in this environment (no live URSim access
here) — it's the user's next step.

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
results; no hyperparameter tuning for any algorithm (default `MlpPolicy`
throughout); no check of whether training for longer than 20k steps keeps
improving cycle time or plateaus; no run yet with the recalibrated
`SCORE_WEIGHT=10800`; no run yet with the new default algorithm (`SAC`,
§8) — every real run on record used `PPO`.

## 6. Plots each run produces

| file | from | shows |
|---|---|---|
| `training_curve.png` | `train_rla.py`'s `log_training_run` | score / cycle_time / reward vs. timestep, three subplots, each raw (faint) + rolling mean; score and cycle_time additionally get a best-so-far reference line, reward doesn't (there's no single "best reward" line drawn — read it off the score/cycle_time panels instead) |
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
- `path` mode is unexercised beyond the offline smoke test — no real
  training run or baseline-vs-optimized comparison yet, and its
  `PATH_SCORE_WEIGHT=20.0` hasn't been rechecked against a trained policy
  the way `SCORE_WEIGHT` just was (§4a) — likely has the same kind of drift.
- Agent hyperparameters are stable-baselines3 defaults throughout, for
  every algorithm (`PPO`/`SAC`/`TD3`) — untuned.
- **No real training run yet with `SAC`/`TD3`** (§8, new default) — the
  off-policy recommendation is a structural argument from the environment's
  shape, not yet a measured result. Needs a real run against live URSim,
  compared against the `PPO` baseline already on record
  (`results/2026-08-20_12-57-02_rla_params/`), ideally with the same
  `seed`/scripts/`SCORE_WEIGHT=10800` held fixed so the only thing that
  differs is the algorithm.

## 8. Algorithm choice: PPO → SAC/TD3, off-policy (2026-08-20)

**Why**: §1/§2 already establish that `GapEnv`/`PathEnv` are one-step
episodes (`reset()` picks a move, `step(action)` scores exactly one
candidate and terminates — no multi-step credit assignment, ever) and §2's
"Scoring a candidate" note that nothing during training touches a real
robot. Put together: every reward is **fully deterministic** for a fixed
`(move, action)` — `dynamics.py`'s physics, the distill model's
`.predict()`, and the metric's aggregation have zero randomness. `PPO` is
**on-policy**: it can only train on the batch of transitions it just
collected, then discards them — every environment/model query is used
exactly once. Nothing about this environment makes a past `(move, action,
reward)` sample go stale (no real robot noise, no sequential state to drift
out from under it), so an **off-policy** algorithm's replay buffer — every
sample reused many times across training — is a strictly better fit and
should be far more sample-efficient here. `SAC` and `TD3` are both
off-policy continuous-action actor-critics already available via
`stable-baselines3` (no new dependency), sharing `PPO`'s
`.learn()`/callback interface.

**What changed**: `train_ppo` generalized to `train_agent(env, algo, ...)`,
`algo` selected from `ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}`.
`train_rla.py`'s new `--algo` flag (`ppo`/`sac`/`td3`) **defaults to
`sac`** — running `train_rla.py` with no extra flags now trains SAC, not
PPO; pass `--algo ppo` to reproduce the original baseline. `log.json` and
`_TrainCallback`'s docstring now record which algorithm produced a run.

**Verified** (offline, no live robot — same limitation as everywhere else
in this file): built `GapEnv` on a locally-labelled recording
(`data/test-1.csv` run through `augment`/`add_score` the same way
`build_dataset` does, minus the live URSim collection step), ran
`train_agent(env, "sac", steps=64, ...)` and `train_agent(env, "td3",
steps=64, ...)` — both completed 64/64 episodes with no exceptions,
`_TrainCallback`'s `self.locals` populated the same `dones`/`rewards`/
`infos` shape `PPO` uses (no callback changes needed), and both produced
records structurally identical to `PPO`'s (`timestep`/`reward`/`score`/
`cycle_time`). This confirms the integration works mechanically — it does
**not** confirm SAC/TD3 actually train a *better* policy than PPO, which
needs a real training run (§7).

**A real bug this surfaced, caught and fixed the same day**: `run.py`
hardcoded `PPO.load(path)` at its two agent-loading call sites
(`search_agent`/`agent_paths`) — a leftover from when `PPO` was the only
algorithm this project ever saved. Loading a `SAC`-saved `agent_params.zip`
with `PPO.load()` doesn't silently misbehave, it raises immediately
(`TypeError: SACPolicy() got multiple values for keyword argument
'use_sde'`, verified directly), so this would have broken `run.py` for
anyone using the new `--algo sac` default the moment they ran it. Fixed
with a new `train_rla.load_agent(path)` helper: tries each `ALGOS` class's
`.load()` in turn, keeps the first success — safe because a mismatched
class raises a clear exception rather than loading with the wrong
architecture (verified in both directions: `PPO.load()` on a `SAC` zip,
and `TD3.load()` on a `SAC` zip, both fail loudly; `SAC.load()` on its own
zip succeeds). `run.py` now imports and uses `load_agent` instead of
`PPO.load` at both call sites. Verified end-to-end: trained one agent per
algorithm, confirmed `load_agent` detects each correctly and `.predict()`
works on all three.

## 8a. PPO and SAC in plain terms — for Q&A (2026-08-20)

§0 has the analogy (student who discards each practice exam vs. one who
keeps and re-studies them all) and §8 has the technical justification for
*this* environment. This section is a level below the analogy and a level
above §8's code-specific detail — what each algorithm actually is, for
fielding a question that goes past the analogy.

**PPO (Proximal Policy Optimization)** — OpenAI, 2017; the field's default
general-purpose choice.
- **On-policy**: only ever learns from data its *current* policy just
  collected; each batch gets used for one small update, then is discarded.
- **How it updates**: an actor-critic policy-gradient method — one network
  picks the action, a second scores how good the situation was, used to
  steady the update. The "proximal" part *clips* how far one update can
  move the policy, so a single bad batch can't wreck training — the reason
  it's a safe default for noisy or multi-step environments.
- **Fits best**: environments where the reward is noisy and/or steps depend
  on each other (games, multi-step robot trajectories) — not this project's
  shape (§1/§2, §8).

**SAC (Soft Actor-Critic)** — Berkeley, 2018; the standard off-policy choice
for continuous actions.
- **Off-policy**: keeps every transition it's ever seen in a replay buffer
  and reuses old data repeatedly, not just once.
- **How it updates**: an actor-critic like PPO, but learns two Q-value
  networks ("twin critics," to avoid overestimating an action's value) plus
  a stochastic policy, and adds an **entropy bonus** — extra reward for
  staying somewhat random — so it keeps exploring instead of committing to
  one guess too early.
- **Fits best**: environments where samples are slow/expensive to collect
  and old samples never go stale — this project's case exactly, since every
  `(move, action)` score is a deterministic calculation (§8), not a noisy
  real robot reading.

**One-liner**: PPO learns once from each fresh batch and moves on; SAC
keeps a notebook of everything it's tried and re-studies the whole thing —
so on a fixed, deterministic environment like this one, SAC gets more
learning out of the same number of samples.

**Likely questions**:
- *"Why not start with SAC?"* — PPO is RL tooling's usual first choice; this
  project started there, and switched once this environment's specific
  shape (one-step, deterministic, no sequencing) made off-policy reuse a
  clear win, not from a general "SAC beats PPO" claim.
- *"Is SAC always better?"* — No. With real robot noise or true multi-step
  credit assignment, reused old data can go stale or misleading — PPO's
  discard-and-move-on approach is the safer default there. That risk is
  absent here because every score is an offline calculation, not a live
  reading.
- *"Have you actually confirmed SAC trains a better policy here?"* — The
  switch is confirmed working end-to-end (loads, trains, saves, `run.py`
  compatible — §8). A real side-by-side comparison of trained policy
  quality is the next step (§7) — this is a reasoned bet, not yet a proven
  result, and worth saying plainly if asked.
- *"What's a replay buffer?"* — A log of every `(state, action, reward,
  next state)` the agent has observed; off-policy algorithms sample from it
  at random to keep training, instead of needing brand-new data every step.
- *"What does the entropy bonus in SAC do?"* — Rewards the policy for
  staying a bit random early on, so it explores more of the action space
  before settling, rather than converging to a mediocre answer too fast.
- *"Same library for both?"* — Yes, both are `stable-baselines3` with the
  same `.learn()`/`.predict()`/`.save()`/`.load()` interface — switching was
  a one-flag change (`--algo`), not a rewrite (§8).

## Changelog

- **2026-08-20** — Added §8a: a standalone "what PPO/SAC actually are" primer
  (on-policy vs. off-policy, actor-critic, PPO's clipped update, SAC's twin
  critics + entropy bonus) plus a likely-questions list, for presentation
  Q&A — a level below §8's code-specific justification and a level above
  §0's analogy.
- **2026-08-20** — `GapEnv` action bounds recalibrated from user-supplied
  hardware info, not just measurement: `VEL_BOUNDS` widened `(20, 180) ->
  (1, 179)` deg/s (ceiling set just under the user's robot's own measured
  191 deg/s joint speed limit, floor dropped near zero to allow very
  gentle moves too); `ACC_BOUNDS` narrowed `(40, 600) -> (1, 200)` deg/s²
  after the user reported the real robot feels jerky at the old ceiling —
  checked via ramp-time (600 deg/s² reaches full speed in ~0.3s vs. ~0.9s
  at 200) and a direct `GapEnv.score()` sweep (real distill model, a real
  move: score was not still climbing by 250-600 deg/s², so nothing measured
  argued for keeping the higher ceiling). §2 updated to match. Any
  already-trained agent must be retrained against the new bounds — the
  action space itself is unchanged ([-1,1]^2), but what a given action now
  maps to physically is different. Not yet verified by a training run.
- **2026-08-20** — Added §4b: a real SAC run with §4a's mean-matched
  `SCORE_WEIGHT=10800` still left score flat, root-caused to mean-matching
  balancing average size, not gradient — cycle_time's std/range across the
  action space is ~12x score's. Recalibrated to a *std*-matched weight via
  `tune_reward_weights.py`'s new second recommendation
  (`recommended_std = target_ratio * cycle_weight * std(cycle_time) /
  std(score)`); `SCORE_WEIGHT` changed **10800.0 → 80000.0**. Also found and
  fixed a circularity trap: computing the std-matched number from a
  *converged* run's own `episodes.csv` gave 8,460 (badly understated,
  because a trained policy's visited action range is narrower than the full
  space) vs. 79,626 from unbiased random sampling — `tune_reward_weights.py`
  now warns on `--episodes` use and documents this in its module docstring.
  §4's top note and the `SCORE_WEIGHT` comment block in `train_rla.py` both
  updated to point at §4b instead of §4a as current. Not yet verified by a
  training run of its own.
- **2026-08-20** — Full consistency audit against the actual current
  `train_rla.py`/`run.py` (line-by-line read of both, not spot checks).
  Found and fixed: (1) §3's `train_rla.py:546`/`run.py:213` citations were
  stale — actual current lines are 738/262, and both files have grown
  enough since this was written that any specific line number would drift
  again soon, so replaced with a "grep for `PositionGapMetric()`" pointer
  instead of a number that will just go stale again; (2) §6's
  `training_curve.png` description claimed all three subplots (score,
  cycle_time, reward) get a best-so-far reference line — checked
  `log_training_run`'s plotting code directly: only score and cycle_time
  do, reward doesn't, description corrected; (3) §4's weight table (the
  2026-08-19 calibration) could read as still-current since §4a's
  recalibration doesn't literally restate "this table is now stale" —
  added an explicit note that the code's real value is
  `SCORE_WEIGHT=10800.0`, not the `2500.0` shown there. Also fixed an
  unrelated docstring formatting glitch in `train_rla.py`'s
  `_TrainCallback` (an orphaned line break splitting "terminal output"
  across two lines mid-sentence, left over from an earlier edit).
- **2026-08-20** — Added §0, a plain-language, slide-ready summary (matching
  `ModelReview.md` §0's convention) explaining what the RL policy does and
  why `PPO` was swapped for `SAC` (§8), via a packing/studying analogy —
  written for a non-RL-expert audience, explicit that the "why it's better"
  case is a reasoned prediction from the environment's shape, not yet a
  measured result. Also fixed two lingering PPO-only mentions in §5's
  inline "not yet done" note that predated the algorithm swap (§8).
- **2026-08-20** — Switched the default RL algorithm from `PPO` (on-policy)
  to `SAC` (off-policy) — see §8. `GapEnv`/`PathEnv` are one-step,
  deterministic contextual bandits; PPO's on-policy machinery discards
  every sample after one use, while an off-policy replay buffer reuses
  samples many times with nothing about this environment making them go
  stale. Generalized `train_ppo` → `train_agent(env, algo, ...)`
  (`ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}`); new `--algo` flag on
  `train_rla.py`, default `sac`, `--algo ppo` reproduces the original
  baseline. `log.json`/`_TrainCallback` now record which algorithm produced
  a run. Verified offline (no live robot) that `SAC`/`TD3` integrate
  mechanically — same callback interface, same record shape as `PPO`, no
  exceptions over 64 steps each — but this is a structural argument for why
  off-policy should train a better policy here, not yet a measured result;
  needs a real run against live URSim compared against the `PPO` baseline
  already on record before trusting it. This also caught a real bug:
  `run.py` hardcoded `PPO.load(path)` at both its agent-loading call sites,
  which raises immediately on a `SAC`-saved `agent_params.zip` (verified —
  not a silent misload) — would have broken `run.py` for anyone using the
  new default the moment they ran it. Fixed with `train_rla.load_agent(path)`
  (tries each `ALGOS` class's `.load()`, keeps the first success — safe
  since a mismatched class fails loudly, verified in both directions);
  `run.py` updated to use it. Verified end-to-end for all three algorithms.
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
