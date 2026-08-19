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

## 2. Models: `PerJointPositionModel` and `PerJointTreeModel`

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

**`PerJointTreeModel`** (added 2026-08-19): a second `DistillModel`,
`--model tree_per_joint`, one `sklearn.ensemble.HistGradientBoostingRegressor`
per joint instead of `lstsq`. Same residual target convention as
`PerJointPositionModel` — matters *more* for a tree than for the linear
model: a tree approximates functions as piecewise-constant regions, so
making it reproduce "output ≈ target_q" (a continuous ~±π-range near-identity
mapping) before it has capacity left for the millirad-scale correction would
waste most of its splits on the trivial part.

Initially shipped with the pre-gravity 7-feature set (a scope decision, not
evidence-driven) so the first linear-vs-tree comparison wasn't
apples-to-apples on the joints gravity helped. **Updated same day** to
inherit the parent's full 8-feature set (including `gravity_torque`) —
see §6 for the result, which was not what the missing-feature hypothesis
predicted: gravity moved the tree model's overall R² by +0.0015 (0.882 →
0.883), an order of magnitude less than its effect on the linear model
(+0.008), and did **not** close the wrist1/wrist2 gap to linear. A plausible
read: the tree can already approximate some of the gravity-driven,
pose-dependent pattern from the raw `pos`/`qd`/`qdd` features via nonlinear
splits, without needing to be handed the physics formula explicitly — the
value of hand-engineering a physics feature is much higher for a model with
no capacity to discover nonlinear pose-dependence on its own (linear) than
for one that can partially find it anyway (trees). Worth stating plainly:
this is exactly why the missing-feature hypothesis needed to be *tested*, not
assumed — it turned out to be wrong.

To support two model shapes without duplicating ~40 lines of feature-building
logic, `_row_features`/`_design`/`predict` were refactored to key off
`self.FEATURE_NAMES` (a `{name: array}` dict, then `column_stack` in
`FEATURE_NAMES` order) rather than a hard-coded column list — `_design` also
only pays for `_gravity_block`'s FK pass when `"gravity_torque"` is actually
in `FEATURE_NAMES`, so the tree model doesn't compute a feature it won't use.
Both classes' `_row_features`/`_design`/`predicts`/`bounds` are now shared
unconditionally; only `fit`/`predict`/`params`/`coefficients` differ.

**No feature scaling needed for the tree model**: `HistGradientBoostingRegressor`
splits are threshold-based on one feature at a time and invariant to
monotonic per-column transforms — this is *not* the same class of model as a
true gradient-descent-trained one (e.g. an MLP), which would need it. Don't
conflate "gradient-*boosted*" with "gradient-descent-optimized" — this
project hasn't used the latter and doesn't need scaling here.

**A real bug the refactor caught and fixed**: `_row_split_eval` (the
held-out diagnostic) used to hard-code `np.linalg.lstsq` for its throwaway
fit regardless of `--model` — so before the fix, the tree model's reported
held-out RMSE/R² would have silently just been a *linear* refit's accuracy
on the same 7 features, not the tree's. Fixed by extracting a
`_row_split_fit_predict(X_train, y_train, X_test)` hook onto
`PerJointPositionModel` (default: the old `lstsq` logic) that
`PerJointTreeModel` overrides to fit+predict a fresh
`HistGradientBoostingRegressor` instead. Also fixed in the same pass:
`bounds()` checked `self.coefs is None` to decide if the model was fit —
always `True` for the tree model (which never sets `self.coefs`, only
`self.models`), so `bounds()` would have always incorrectly returned `None`.
Changed the check to `self.vel_range is None`, the attribute `bounds()`
actually returns, which both models set identically.

`HistGradientBoostingRegressor(random_state=_RANDOM_STATE)` — a fixed seed
is necessary, not just tidy: its auto-triggered early stopping (active here,
n≫10,000) internally carves out its own validation split, which is
non-deterministic without a seed, undermining this project's otherwise
deterministic ("no RNG") split methodology (§5).

## 3. Feature engineering

`FEATURE_NAMES = ["target_current", "qd", "qdd", "pos", "gravity_torque", "vel", "acc", "bias"]`
— eight features per joint per row:

| feature | source | why it's here |
|---|---|---|
| `target_current` | recorded/commanded current | proxy for commanded torque — the physical driver of elastic deflection (torque -> position error), already logged so free to use |
| `qd` | `target_qd` (commanded velocity) | tracking lag and settle-window ring both scale with how fast the joint is being driven |
| `qdd` | `np.gradient(qd, dt)` | commanded acceleration — same reasoning, and a proxy for the torque term `M(q)qdd` |
| `pos` | `target_q` | commanded angle — the residual's reference point, and the input `gravity_torque` (below) is computed from; other pose-dependent physics (inertia at this pose) still isn't computed, see §7 |
| `gravity_torque` | `utils.UR10e.gravity(target_q)[joint]` | direct physical driver of static deflection — see below |
| `vel`, `acc` | the raw `movej` register values (not the realized `qd`/`qdd`) | the *commanded profile shape*, independent of what was actually achieved |
| `bias` | constant 1 | per-joint intercept, see §2 |

This is the same feature basis the old current-predicting baseline used
(plus `gravity_torque`), carried over for the position target largely
unchanged — it has **not** been re-justified from scratch for position the
way the split/metric/model-shape decisions above were. That re-justification
is open work; see §6.

**`gravity_torque`** (added 2026-08-19): `utils.UR10e.gravity(q)` returns all
six joints' torques from one full-pose call — a joint's own `pos` alone
isn't enough, since gravity torque on any joint depends on the whole
kinematic chain's configuration. Computed once per recording (not once per
joint) via the new `utils.UR10e.gravity_batch` and sliced per joint;
`_gravity_block` in this file wraps that call. **Performance note**: the
per-row `gravity()` method is not vectorized and is ~150x too slow to call
in a loop over the ~1e6 rows a training run covers (verified: would have
taken minutes, called 2-3x per run) — `gravity_batch` vectorizes the same
FK/Jacobian math with numpy (verified bit-identical output against the
per-row method on random poses, 7.5s for 1.08M rows).

Justification: direction of travel alone (same joint, same speed range,
opposite direction) swung measured peak overshoot 3.4x (shoulder+ 5.5 mrad
vs. shoulder- 1.6 mrad, `bronze_exploration.py`'s `segment_stats.csv`) —
bigger than anything vel/acc explain (`corr(vel, overshoot)=-0.27`,
`corr(acc, overshoot)=-0.33`, both weak). Gravity torque is the physical
quantity direction was standing in for.

**Result**: held-out R² 0.437 → 0.445 overall. Moved almost exactly where
physics predicts and nowhere else: shoulder +0.0095 (0.388→0.398), wrist2
+0.0067 (0.833→0.840), elbow +0.0005, wrist1 +0.0008 — and **base and
wrist3 essentially unchanged** (base 0.7246→0.7246 to 4 decimals). That's
not noise: the base joint rotates about the vertical axis, so gravity does
essentially zero work as it rotates — the feature correctly contributed
nothing there. wrist3 was already at the noise floor (§6) and stayed there.

**A coefficient-scale trap, seen concretely in `coefficients.png`**:
`gravity_torque`'s fitted weight rounds to `+0.0000` in the plot for every
joint, including shoulder/elbow where it measurably helped R² above. Not a
bug — gravity torque is tens of Nm while the fit target is millirad-scale,
so its coefficient (rad of position error per Nm) is naturally tiny in raw
units even though the feature carries real signal (a large-magnitude
feature times a small-looking coefficient still moves the prediction). This
is the concrete case for showing *standardized* coefficients
(`coef_j × std(feature_j)`, comparable across features regardless of native
units) instead of raw ones — proposed, not yet implemented; see §7.

**Still not in here, with the evidence for adding it:**

- **Lag/history features.** The settle-window "ring" (a damped oscillation
  after the commanded motion stops, ~0.3-0.5 s to decay,
  `bronze_tier/trajectories/worst_move_test-4_shoulder.png`) has memory — a
  memoryless per-row linear model structurally cannot reproduce it, no
  matter what pose-dependent (static) feature you add. Recent `qd`/`qdd`/
  jerk history, or a decaying-oscillation term, is the fix — this is also
  why shoulder/elbow's R² is still the weakest among the "real" (non-noise-
  floor) joints even after `gravity_torque`: gravity is a static term and
  has nothing to say about a dynamic, history-dependent effect.
~~Feature scaling~~ — checked and it's a non-issue for this model: plain OLS
(`lstsq`, no regularization) is invariant to per-column rescaling, it just
rescales the corresponding coefficient — RMSE/R² come out identical either
way. Only matters if we add regularization (ridge/lasso) or move to a
gradient-based/nonlinear method (§7). (Don't confuse this with the
coefficient-*display* issue above — that's about interpreting the plot, not
about fit quality.)

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

## 6. Current results (last run: 2026-08-19, `results/2026-08-19_13-59-27/`)

7 files, 20% row holdout (216,099 held-out rows/joint). With `gravity_torque`
(previous run without it in parentheses):

| joint | held-out RMSE (deg) | held-out R² |
|---|---|---|
| overall | 0.0242 | 0.445 (was 0.437) |
| base | 0.0076 | 0.725 (unchanged) |
| shoulder | 0.0561 | 0.398 (was 0.388) |
| elbow | 0.0153 | 0.462 (was 0.462) |
| wrist1 | 0.0053 | 0.914 (was 0.913) |
| wrist2 | 0.0035 | 0.840 (was 0.833) |
| wrist3 | 0.0019 | 0.022 (unchanged) |

In-sample (100%-refit model, scored on its own training data) matches the
held-out numbers almost exactly (e.g. overall R²=0.445 vs 0.445) — the
expected sanity check for this much data, not evidence of good
generalization (see §5's caveat).

Reading these: shoulder and elbow — the joints with the largest settle-window
ring — are still the weakest of the non-noise-floor joints even after adding
the (static) gravity feature, exactly as §3 predicts: the ring is a dynamic,
history-dependent effect a static pose feature can't touch. wrist3's R²≈0.02
is likely near the noise floor — its gap is tiny (RMSE 0.002°) and may be
dominated by encoder/measurement noise rather than a systematic, learnable
effect (worth an FFT/autocorrelation check on its residual before spending
feature-engineering effort there; also consistent with it not moving at all
when `gravity_torque` was added).

**Tree vs. linear, both with the same 8 features** (`results/2026-08-19_15-03-35/`
tree, `.../2026-08-19_14-30-54/` linear; both 20% row holdout, same split).
The tree column here supersedes the first (no-gravity) comparison — kept
inline below since the *change* from adding gravity to the tree model is
itself the interesting result:

| joint | linear R² | tree R² (no gravity) | tree R² (w/ gravity) | linear RMSE (deg) | tree RMSE (deg) |
|---|---|---|---|---|---|
| overall | 0.445 | 0.882 | **0.883** | 0.0242 | **0.0111** |
| base | 0.725 | 0.734 | 0.734 | 0.0076 | 0.0075 |
| shoulder | 0.398 | 0.913 | **0.914** | 0.0561 | **0.0213** |
| elbow | 0.462 | 0.630 | **0.637** | 0.0153 | **0.0127** |
| wrist1 | **0.914** | 0.833 | 0.834 | **0.0053** | 0.0074 |
| wrist2 | **0.840** | 0.796 | 0.798 | **0.0035** | 0.0039 |
| wrist3 | 0.022 | 0.148 | 0.147 | 0.0019 | **0.0017** |

Not a clean sweep, and worth reading honestly: the tree model is a large win
on base/shoulder/elbow/wrist3 (shoulder R² more than doubles, from 0.40 to
0.91, without any lag/history features — trees pick up some of the
instantaneous nonlinear interaction among `qd`/`qdd`/`target_current`/`pos`
a linear model structurally can't, even though neither model has been given
the temporal history the ring actually needs), but **linear still wins on
wrist1 and wrist2** even now that both models see identical features — so
this is a genuine regressor difference on those two joints, not a
missing-feature artifact (the original, untested hypothesis from before
gravity was added to the tree model). Read together with how little gravity
moved the tree overall (+0.0015 vs. linear's +0.008, see §2), the likely
story: wrist1/wrist2's residual is close to a smooth, near-linear function
of the inputs, which a linear model represents natively and a
piecewise-constant tree (with untuned default hyperparameters) approximates
less efficiently — while shoulder/elbow/base's residual is nonlinear enough
that the tree's flexibility wins outright. Different joints may simply want
different model shapes; not tested here (would mean per-joint model
selection, not just a per-joint *fit*, which both models already do).
`models/distill_tree.pkl` saved alongside `models/distill.pkl` (still the
default/shipped model — `--model` default unchanged).

**Caveat on all of the above, not yet resolved**: these are still row-level
held-out numbers (§5), and the leakage concern applies *more* to the tree
model than the linear one. `HistGradientBoostingRegressor` has far more
capacity than a 7-parameter-per-joint linear fit, so it can exploit
"this held-out row's neighbors are in the training set" more effectively —
the tree's R²=0.88 could be sitting on more leakage-driven optimism than the
linear model's (whose row-level-vs-file-level gap was already measured and
small, §5). Not yet checked for the tree model. The leave-one-file-out CV
diagnostic (§7) would answer this and is now higher priority than before.

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

1. ~~Gravity torque feature~~ — done, see §3/§6 (2026-08-19).
2. **Lag/history features for the settle-window ring** — shoulder (R²=0.40)
  and elbow (R²=0.46) are the weakest joints that aren't just noise, and
  they're exactly the joints with the biggest ring. No per-row feature fixes
  this; needs recent `qd`/`qdd`/jerk at a few lags, or a decaying-oscillation
  term. Still fits with plain `lstsq`.
3. **Standardized coefficients in `coefficients.png`** — cheap, and now
  demonstrated as a real gap (§3): `gravity_torque`'s raw weight rounds to
  `0.0000` in the plot despite measurably improving R², because it's in
  Nm-scale units against a millirad-scale target. Show `coef_j × std(feature_j)`
  instead so bar heights are comparable across features regardless of native
  units; doesn't change the fit, only the display.
4. **Leave-one-file-out CV diagnostic** — 7 cheap refits (42 params), gives
  an honest file-level generalization number alongside the row-level holdout
  (§5's caveat); doesn't change what gets shipped.
5. ~~A non-linear regressor~~ — done, `PerJointTreeModel`, see §2/§6
  (2026-08-19), **out of this list's stated order** (done before item 2, at
  the user's explicit direction). The predicted caveat held: the tree model
  still can't see history it isn't given, so it doesn't fix the settle-
  window ring — shoulder/elbow both improved a lot over linear but item 2
  (lag features) is still the more direct fix for the ring specifically, and
  is untested on top of the tree model.
6. `dynamics.Dynamics.frame()` doesn't emit an `actual_q` placeholder column
  (only `actual_current`) — relevant once this model needs to run inside
  `train_rla.py`'s candidate scoring, not for offline training/evaluation.
7. `train_rla.py`/`run.py` still wired to `CurrentGapMetric`/`actual_current`
  — need to switch to `PositionGapMetric` before they'll run against a model
  from this file.

## Changelog

- **2026-08-19** — Added `gravity_torque` to `PerJointTreeModel` (removed its
  `FEATURE_NAMES` override so it inherits the parent's full 8-feature set;
  fixed its `predict()` override, which had no gravity-conditional logic
  since it predated the feature, to match the parent's pattern). Tested the
  hypothesis from the previous entry ("linear's wrist1/wrist2 edge might
  just be a missing feature") and found it **false**: gravity moved the
  tree's overall R² by only +0.0015 (0.882→0.883, vs. linear's +0.008) and
  didn't close the wrist1/wrist2 gap at all. Documented likely explanation
  in §2/§6: trees can partially approximate pose-dependent nonlinearity from
  raw features without being handed the physics formula, so the feature is
  worth much less to them than to a linear model with no such capacity.
- **2026-08-19** — Added `PerJointTreeModel` (`--model tree_per_joint`,
  `sklearn.ensemble.HistGradientBoostingRegressor` per joint), deliberately
  using the pre-gravity 7-feature set (user's call, not evidence-driven).
  Refactored `_row_features`/`_design`/`predict` to key off
  `self.FEATURE_NAMES` so both models share the same feature-building code
  without duplication. Fixed two real bugs surfaced by adding a second model
  class: (1) `_row_split_eval` hard-coded a linear `lstsq` refit for its
  held-out diagnostic regardless of `--model` — extracted a
  `_row_split_fit_predict` hook so each model's own fitting method is used;
  (2) `bounds()` checked `self.coefs is None`, which the tree model never
  sets, so it always returned `None` post-fit — changed the check to
  `self.vel_range is None`. Verified: reran the linear model post-refactor
  and confirmed identical numbers to the pre-refactor run (0.445 R²,
  bit-for-bit per-joint match); confirmed `model.bounds()` returns real
  values (not `None`) on the fitted tree model; confirmed `coefficients.png`
  is skipped (not errored) for the tree model. Results: held-out R² 0.445
  (linear) vs. 0.882 (tree) overall — but not a clean win, linear is still
  better on wrist1/wrist2 (the joints `gravity_torque` helped, which the
  tree model doesn't have) — see §2/§6 for the full per-joint table and the
  apples-to-apples caveat.
- **2026-08-19** — Added the `gravity_torque` feature to `PerJointPositionModel`
  (`FEATURE_NAMES` now 8 entries). Added `utils.UR10e.gravity_batch` (and
  its `_dh_batch`/`_frames_batch`/`_point_jacobian_batch` helpers) after
  discovering the naive per-row `gravity()` loop was ~150x too slow for
  training-scale data (would have taken minutes, called 2-3x per run);
  verified the batched version is bit-identical to the per-row one on random
  poses before wiring it in, then confirmed a full training run completes in
  ~41s. Held-out R² 0.437→0.445 overall, moving almost exactly where physics
  predicts (shoulder/wrist2 up, base/wrist3 unchanged — base rotates about
  the vertical axis, so gravity does ~zero work there, and wrist3 is already
  noise-floor). Also found and documented a coefficient-display gap while
  reviewing `coefficients.png`: `gravity_torque`'s raw-unit weight rounds to
  `0.0000` despite the real R² gain, because it's Nm-scale against a
  millirad-scale target — added "standardized coefficients" to §7 as the
  proposed (not yet implemented) fix.
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
