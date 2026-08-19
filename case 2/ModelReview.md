# Model Review — `train_distillation_model.py`

Living document. **Update this file every time `train_distillation_model.py`,
`metrics.py`, or their feature/split/evaluation logic changes** — add a dated
entry to the Changelog at the bottom, and edit the sections above it so they
always describe the *current* state (don't leave the body describing an old
version — that's what the changelog is for).

## 1. What we predict

**Position**, not current. `PerJointPositionModel.predicts() -> ["actual_q"]`
— the actual joint angle (rad) a real UR10e ends up at, given a commanded
trajectory. Internally it predicts the *residual* `actual_q - target_q` (the
tracking gap) and adds `target_q` back before returning, so the public
interface still hands back an absolute angle like any other `DistillModel`.

Current (`actual_current`) is no longer predicted by anything in this file.
The old current-predicting `LinearModel` was removed entirely (see
Changelog). `train_rla.py`/`run.py` still default to
`metrics.CurrentGapMetric` and expect `actual_current` — they have not been
updated yet and will not work with a model saved from this file until they
switch to `metrics.PositionGapMetric`.

## 2. Model: `PerJointPositionModel`

One independent least-squares linear fit **per joint** — six separate
`np.linalg.lstsq` calls, not one shared-slope model with a joint indicator.

**Why per-joint, not pooled.** An earlier baseline (`LinearModel`, since
removed) pooled all six joints into one fit with a one-hot joint block (shared
slopes, per-joint intercept only). Measured on the *current* channel, held out
by file:

| joint | pooled (shared-slope) R² | per-joint R² |
|---|---|---|
| base | 0.86 | 0.86 |
| shoulder | 0.98 | 0.98 |
| elbow | 0.98 | 0.98 |
| wrist1 | 0.86 | 0.91 |
| wrist2 | 0.43 | 0.60 |
| wrist3 | 0.07 | 0.26 |

Per-joint never loses and sometimes gains a lot, because the low-current wrist
joints' true sensitivity to the inputs differs in scale *and sign* from the
big joints (e.g. one joint's fitted slope on `qd` was +1.26, another's was
-0.06) — a pooled fit can only offer one compromise slope, tuned mostly by
whichever joints dominate the loss (the big ones).

**Target: residual, not raw position.** Fits `actual_q - target_q`, not raw
`actual_q`. Reasons: (1) the residual is near-zero-mean and on a consistent,
small scale across joints (rad of *error*, not rad of arbitrary joint angle);
(2) it's what a distillation model conceptually should learn — a correction
on top of the commanded trajectory, not the trajectory itself; (3) it's what
made the evaluation-metric bug in §4 detectable in the first place.

**Intercept.** Each joint's fit has a real bias term (no one-hot to be
collinear with, since each joint is now its own independent fit) — gives a
joint's average static bias (e.g. gravity sag at its typical poses) somewhere
to go instead of leaking into the other coefficients.

## 3. Feature engineering

`FEATURE_NAMES = ["target_current", "qd", "qdd", "pos", "vel", "acc", "bias"]`
— seven features per joint per row:

| feature | source | why it's here |
|---|---|---|
| `target_current` | recorded/commanded current | proxy for commanded torque — the physical driver of elastic deflection (torque -> position error), already logged so free to use |
| `qd` | `target_qd` (commanded velocity) | tracking lag and settle-window ring both scale with how fast the joint is being driven |
| `qdd` | `np.gradient(qd, dt)` | commanded acceleration — same reasoning, and a proxy for the torque term `M(q)qdd` |
| `pos` | `target_q` | commanded angle — needed for any pose-dependent effect (gravity torque, inertia change with pose) even though this baseline doesn't yet compute those directly |
| `vel`, `acc` | the raw `movej` register values (not the realized `qd`/`qdd`) | the *commanded profile shape*, independent of what was actually achieved |
| `bias` | constant 1 | per-joint intercept, see §2 |

This is the same feature basis the old current-predicting baseline used,
carried over for the position target largely unchanged — it has **not** been
re-justified from scratch for position the way the split/metric/model-shape
decisions above were. That re-justification is open work; see §6.

**What's deliberately not in here yet, with the evidence for adding it**
(from `bronze_exploration.py`'s `segment_stats.csv`, 2497 real moves):

- **Direction of travel / gravity torque.** Same joint, same speed range,
  opposite direction: shoulder(+) mean peak overshoot 5.5 mrad vs.
  shoulder(-) 1.6 mrad — a 3.4x swing from direction alone, bigger than
  anything vel/acc explain (`corr(vel, overshoot) = -0.27`,
  `corr(acc, overshoot) = -0.33`, both weak). `utils.UR10e.gravity(q)` already
  exists and is unused by this model.
- **Lag/history features.** The settle-window "ring" (a damped oscillation
  after the commanded motion stops, ~0.3-0.5 s to decay,
  `bronze_tier/trajectories/worst_move_test-4_shoulder.png`) has memory — a
  memoryless per-row linear model structurally cannot reproduce it. Recent
  `qd`/`qdd`/jerk history, or a decaying-oscillation term, is the fix.
~~Feature scaling~~ — checked and it's a non-issue for this model: plain OLS
(`lstsq`, no regularization) is invariant to per-column rescaling, it just
rescales the corresponding coefficient — RMSE/R² come out identical either
way. Only matters if we add regularization (ridge/lasso) or move to a
gradient-based/nonlinear method (§7).

## 4. Evaluation metric: R² against "no gap," not "the mean"

`_compute_metrics`'s R² is `1 - ss_res/ss_tot` with `ss_tot` measured against
**target** (`sum((actual - target)^2)`, i.e. the trivial "predict no gap"
baseline), not the textbook "predict the mean of actual" baseline.

This was a real bug caught during this work, not a style choice: for
`actual_q`, "predict the mean" is a nearly worthless null model — position
spans radians over a move while the gap is millirad-scale, so a model that
just copies `target_q` into `actual_q` (i.e. learned nothing) scored the same
**R²=1.0000** the real fitted model did. Scoring against target instead
measures the thing actually being modeled: how much of the tracking gap the
model explains, relative to assuming there is none. Applies to any channel
(also changed `actual_current`'s numbers, less drastically, when that still
existed).

**If you're comparing against `old_position_model.py`'s numbers** (an
earlier branch's shared-slope `LinearModel`, held-out R²≈1.000 for every
joint) and see today's R² look much worse for similar or even better RMSE —
that's this fix, not a regression:

- Old's held-out R² (`main()`, `err_j = X[mask]@coef - y_j`, `y = actual_q`):
  `ss_tot = sum((y_j - y_j.mean())**2)` — variance of raw `actual_q` around
  its own mean, huge because it's dominated by the trajectory's own radian-
  scale swing, not the millirad gap. Same pathology as above.
- RMSE is still a fair, apples-to-apples comparison: old computes
  `sqrt(mean(err**2))` on raw-`actual_q` error, in radians, converted to
  degrees only for its plot (`np.degrees(rmse)`); new computes the same
  quantity via the residual-reconstruction path, which is algebraically
  identical (`target_q` cancels — see the earlier Q&A on this in the
  session). New is slightly better on every joint (e.g. shoulder
  0.058°→0.057°, elbow 0.017°→0.015°) — consistent with the per-joint-vs-
  shared finding in §2, now confirmed for position too.
- One real secondary difference, unrelated to R² and too small to explain
  the gap: old's row-level split was originally computed once over a design
  matrix stacked as joint-blocked chunks per recording, then sliced per
  joint via `argmax` on the one-hot columns — since each joint's block
  started at a different offset, the modulo split held out a different,
  phase-shifted set of time-samples per joint. **Fixed** in
  `old_position_model.py` (`_held_out_mask`, added 2026-08-19): computes the
  mask once per recording from a running global row counter and reuses it
  across that recording's six joint blocks, so all six joints now hold out
  the same underlying rows — verified directly (before: joints disagreed on
  which rows were held out; after: 100% agreement, same ~20% fraction).
  Matches `_row_split_eval`'s semantics.

## 5. Train/test split: row-level, matches `original_train.py`

Trains on **all 7 recordings** (`DEFAULT_CSVS = data/test-{1..7}.csv`).
`_row_split_eval()`:

1. Pools every row of every joint (`model._design(recordings)`).
2. Marks a deterministic fraction as "test" — every `round(1/holdout)`-th row
  by index (`holdout=0.2` -> every 5th row), no RNG.
3. Fits a **throwaway** model on the remaining 80% just to report held-out
  RMSE/R².
4. The real, saved model is then **refit on 100% of the rows** — the
  held-out numbers describe a fit that gets discarded, not the shipped model.

This intentionally replaces an earlier, stricter file-level split (whole
recordings held out, never used to fit the saved model) — chosen because the
project wants to match `original_train.py`'s methodology specifically.

**Known caveat, by design choice, not oversight**: because held-out rows sit
*inside* the same continuous trajectories the fit sees (robot streams at
~128 Hz, so adjacent rows are ~7.8 ms apart and highly autocorrelated), this
measures interpolation within seen trajectories more than generalization to
an unseen run. Measured effect on this data: row-level held-out R²=0.437 vs.
the stricter file-level split's 0.432 — a small optimistic bias in this case,
but worth stating whenever these numbers are reported.

## 6. Current results (last run: 2026-08-19, `results/2026-08-19_13-00-35/`)

7 files, 20% row holdout (216,099 held-out rows/joint).

| joint | held-out RMSE (deg) | held-out R² |
|---|---|---|
| overall | 0.0242 | 0.437 |
| base | 0.0076 | 0.725 |
| shoulder | 0.0566 | 0.388 |
| elbow | 0.0153 | 0.462 |
| wrist1 | 0.0053 | 0.913 |
| wrist2 | 0.0035 | 0.833 |
| wrist3 | 0.0019 | 0.022 |

In-sample (100%-refit model, scored on its own training data) matches the
held-out numbers almost exactly (e.g. overall R²=0.437 vs 0.437) — the
expected sanity check for this much data, not evidence of good
generalization (see §5's caveat).

Reading these: shoulder and elbow — the joints with the largest settle-window
ring — fit worst despite being the "easy," high-signal joints on the current
channel. That's the linear/no-history model failing on exactly the nonlinear,
memory-dependent effect it can't represent (§3). wrist3's R²≈0.02 is likely
near the noise floor — its gap is tiny (RMSE 0.002°) and may be dominated by
encoder/measurement noise rather than a systematic, learnable effect (worth
an FFT/autocorrelation check on its residual before spending feature-
engineering effort there).

**Plots each run produces** (`results/<datetime>/`), styled with `ur_style.py`:

| file | shows |
|---|---|
| `residuals_actual_q.png` | pooled residual histogram + all 6 joints' distributions overlaid |
| `residuals_per_joint_actual_q.png` | same residuals, one histogram per joint, own x-scale |
| `per_joint_metrics_actual_q.png` | held-out RMSE (deg) and R² per joint, side by side in one figure |
| `coefficients.png` | fitted weights per joint, one bar chart per joint (6 subplots) — comparable in spirit to an earlier branch's single-model coefficients plot, but split per-joint since that's this model's actual structure |

`coefficients.png` is produced by `_plot_coefficients`, which calls
`model.coefficients()` — an optional `DistillModel` hook (default `None`).
`PerJointPositionModel` implements it; a future non-linear regressor doesn't
have to, and the plot is silently skipped (printed, not an error) rather than
shown empty or wrong — there's no reason to plot linear weights for a model
that doesn't have any.

## 7. Not yet done, in priority order

1. **Gravity torque feature** — `utils.UR10e.gravity(q)` already exists,
  unused. Cheapest, most evidence-backed: direction alone swings overshoot
  3.4x (shoulder+ vs shoulder-, bronze-tier data) while vel/acc barely
  correlate (~0.3) — gravity torque is the physical quantity direction
  stands in for.
2. **Lag/history features for the settle-window ring** — shoulder (R²=0.39)
  and elbow (R²=0.46) are the weakest joints that aren't just noise, and
  they're exactly the joints with the biggest ring. No per-row feature fixes
  this; needs recent `qd`/`qdd`/jerk at a few lags, or a decaying-oscillation
  term. Still fits with plain `lstsq`.
3. **A non-linear regressor** — after, not instead of, 1-2: it can't see
  history it isn't given, so swapping the regressor alone just buys a
  fancier memoryless model.
4. `dynamics.Dynamics.frame()` doesn't emit an `actual_q` placeholder column
  (only `actual_current`) — relevant once this model needs to run inside
  `train_rla.py`'s candidate scoring, not for offline training/evaluation.
5. `train_rla.py`/`run.py` still wired to `CurrentGapMetric`/`actual_current`
  — need to switch to `PositionGapMetric` before they'll run against a model
  from this file.

## Changelog

- **2026-08-19** — Fixed the split phase-alignment inconsistency in
  `old_position_model.py` (`_held_out_mask`, replacing the old inline
  `arange(len(y)) % step`): each recording's held-out mask is now computed
  once and reused across all six joint blocks, so every joint holds out the
  same underlying rows, matching `_row_split_eval`'s semantics. Verified via
  a direct row-identity check (before: joints disagreed; after: 100%
  agreement). Separately discovered while testing: `old_position_model.py`'s
  `main()` can't currently run end-to-end on this branch — it imports
  `LinearModel` from `train_distillation_model`, which no longer has that
  class since it was removed earlier this session. Not fixed yet (pending a
  decision on whether the self-import should just point at
  `old_position_model` instead); the mask fix itself was verified without
  running `main()`.
- **2026-08-19** — Documentation only, no code changed: added §4's
  comparison-against-`old_position_model.py` explanation, after the user
  asked why their old branch's held-out R² was ≈1.000 while the current
  numbers are much lower despite similar/better RMSE. Confirmed by reading
  both files: it's the same R²-baseline fix already in §4 (old scores
  against variance of raw `actual_q`, new against the tracking gap), not a
  model regression. Also found and documented a secondary, non-explanatory
  difference: old's per-joint held-out split is phase-shifted across joints
  (an artifact of slicing a joint-blocked stacked matrix), which
  `_row_split_eval` avoids by construction.
- **2026-08-19** — Added `_plot_coefficients` (per-joint weight bar charts,
  `coefficients.png`) and consolidated the old separate RMSE/R² bar charts
  into one combined `_plot_per_joint_metrics` (`per_joint_metrics_*.png`,
  side by side), matching the layout of two comparison plots from an earlier
  branch's baseline so the two can sit next to each other. Added the optional
  `DistillModel.coefficients()` hook (`None` default) so the coefficients
  plot only renders for a model that actually has linear weights to show.
  Corrected the "feature scaling" open item (§3/§7): checked, and it's a
  non-issue for plain OLS.
- **2026-08-19** — Split methodology switched to row-level holdout on all 7
  files, matching `original_train.py` (`DEFAULT_CSVS`/`DEFAULT_HOLDOUT`,
  `_row_split_eval`, refit-on-100%-before-save). Replaces the earlier
  file-level `DEFAULT_TRAIN_CSVS`/`DEFAULT_TEST_CSVS` split. Fixed a stale
  module docstring left over from that change.
- **2026-08-19** — Removed `LinearModel` (current-predicting, shared-slope)
  entirely; `PerJointPositionModel` is now the only model in `MODELS`, and
  the CLI default. Current-channel prediction is out of scope going forward.
- **2026-08-19** — Added UR brand plotting style (`ur_style.py`) across every
  plot in this file and `analysis.py`. Position-valued plots (residual
  histograms, per-joint RMSE, the comparison-plot RMSE row) convert rad to
  degrees for display only — `log.json`/`runs_summary.csv` stay in radians.
- **2026-08-19** — Added `PerJointPositionModel` (position, per-joint,
  residual target) alongside the then-still-present `LinearModel`. Fixed the
  R²-vs-mean bug in `_compute_metrics` (§4) while verifying it.
- **2026-08-19** — `metrics.py`: added `PositionGapMetric` alongside
  `CurrentGapMetric`. Not yet wired into `train_rla.py`/`run.py`.
