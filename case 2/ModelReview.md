# Model Review — `train_distillation_model.py`

Living document. **Update this file every time `train_distillation_model.py`,
`metrics.py`, or their feature/split/evaluation logic changes** — add a dated
entry to the Changelog at the bottom, and edit the sections above it so they
always describe the *current* state (don't leave the body describing an old
version — that's what the changelog is for).

## 0. Plain-language summary — for slides

The rest of this document is the working log. This section is the pitch:
what the best model on the current real-UR5e data (`PerJointTreeModel` with
gravity + lag, hyperparameters tuned via file-grouped CV — see §6c) does and
why per-joint fitting, gravity, history, and boosting each earned their
place, without requiring the reader to know what gradient boosting or R² is.

**Not what `models/distill.pkl` currently holds.** That file (the one
`run.py`/`train_rla.py` load by default) was never repointed to this model —
it still holds an old UR10e-era pickle that, as of 2026-08-20, **fails to
load at all** under the installed scikit-learn (an old-version pickle,
incompatible internals). The actual best model described here is saved
separately at `models/distill_tree_tuned.pkl`. Repointing `distill.pkl` is a
deliberate follow-up, not done as part of this note — see §7.

**One sentence**: for each joint, the model predicts how far off the real
robot's position will end up from what was commanded, as a function of how
that joint is being driven *right now and a moment ago*, plus how hard
gravity is pulling on it at the current arm pose — and adds that correction
on top of the commanded trajectory.

### The basic trick: predict the error, not the position

A commanded trajectory (`target_q`) already tells you almost everything
about where the joint will be — the real robot follows it closely. So
instead of asking the model to predict the whole position from scratch, we
only ask it to predict the small leftover gap:

```
predicted_position(t) = commanded_position(t) + predicted_error(t)
```

`predicted_error(t)` is millirad-scale (a small fraction of a degree);
`commanded_position(t)` can swing over the whole trajectory (radians). This
is the difference between asking someone to describe a photo from scratch
versus asking them to mark up what's wrong with a photo that's already 99%
right — the second question is easier and the answers are more comparable
across very different joints. (§2 in the sections below calls this the
*residual*.)

### One model per joint, not one model for the whole arm

The base, shoulder, elbow and the three wrist joints don't behave alike —
different gearing, different mass driven, different typical speeds. A single
formula tuned to fit all six at once ends up compromising, dominated by
whichever joints have the biggest numbers (the base/shoulder/elbow). So the
model is really six small, independent models — one per joint, each free to
learn that joint's own sensitivity to the inputs — packaged behind one
interface (§2).

### Why gradient-boosted trees, not a straight-line formula

The first version of this model was linear — one fixed formula per joint,
e.g. "position error grows by this much per unit of commanded velocity, plus
this much per unit of gravity torque, plus …," all added together with fixed
weights. That's easy to read off as a table of coefficients, but it assumes
every input affects the error independently and proportionally — no
"only matters when moving fast *and* against gravity at the same time" kind
of effect.

Gradient-boosted trees drop that assumption: instead of one formula, the
model is a sequence of simple decision-tree rules ("if commanded velocity is
above X *and* gravity torque is above Y, add Z"), each correcting what the
previous rules got wrong. That lets it pick up on interactions between the
inputs automatically, without anyone having to guess and hand-write the
interaction terms. Switching from the straight-line formula to trees, with
the exact same inputs, nearly doubled the explained error variance on the
original synthetic UR10e dataset trees were first tried on (§6) — but that
gain didn't carry over for free to the real UR5e data this model actually
ships against: with default hyperparameters, the tree *lost* to the linear
model there (§6b). It took hyperparameter tuning, not just switching to
trees, to recover and exceed the linear model's number (§6c). The
trees-beat-a-straight-line story below is real on the current data, but it's
as much a tuning story as a model-family one.

### Why gravity torque helps: it's a real, physical, missing input

Holding the arm out against gravity takes real motor torque, and torque
translates into a small elastic "sag" — the same way a shelf bracket flexes
more under a heavy load than a light one. That sag *is* part of the position
error we're trying to predict. The model wasn't being told about it directly
before — it only saw joint angles and commanded speeds, which correlate with
gravity loading only loosely.

The physical quantity is the manipulator's **gravity torque vector**, from
standard rigid-body dynamics:

```
M(q)·q̈ + C(q, q̇)·q̇ + g(q) = τ
```

`g(q)` — torque needed at each joint just to hold the current pose `q`
against gravity, with no motion at all — is exactly the missing input. It
depends on the whole arm's pose at once (how the elbow is bent changes the
load felt at the shoulder), not just one joint's own angle, so it's computed
once from the full 6-joint pose and then split out per joint
(`utils.UR5e.gravity`, §3). Feeding this in means the model doesn't have to
*rediscover* gravity's effect from position/velocity data alone — it's
handed the answer directly, the same way an engineer would rather look up a
formula than reverse-engineer it from a scatter plot.

**Where it helped, and where it didn't, is itself evidence it's real
physics and not a lucky correlation**: it moved shoulder and wrist2 (joints
that visibly sag under load) and left base and wrist3 essentially untouched
— base rotates about the vertical axis, so gravity does ~zero work turning
it, exactly as the physics predicts (§3).

### Why "lag" (history) features help: the arm keeps moving after the command stops

After a fast move, a real joint doesn't stop instantly — it overshoots
slightly and rings, like a diving board settling after someone steps off it,
for a few tenths of a second. That ringing is *why* the position error at
this instant depends on what the joint was doing a moment ago, not just what
it's doing right now. A model that only ever looks at the current instant
has no way to tell "just started, moving fast" apart from "already stopped,
still ringing from a moment ago" — both can look identical in an
instantaneous snapshot.

The fix is to hand the model a short memory: alongside the commanded
acceleration *right now*, also give it the same signal from a few moments
earlier:

```
qdd_lag_k(t) = commanded_acceleration(t − k samples),  k = 4, 16, 64
             ≈ commanded_acceleration 31 / 125 / 500 ms ago
```

Three snapshots spread across roughly the ring's decay time (~0.3–0.5s) —
not a full continuous history, just enough for the model to notice "there
was a hard stop recently" and how hard. This is the model's *only* source of
memory; every other input is still a single instant in time.

**Result, and why it's convincing**: on the original ablation this was
tested on (old synthetic UR10e data, §3/§6), this was the single biggest
jump of any one addition. Shoulder — the joint with the most visible ringing
— gained the most of any joint on both models, and the gains lined up with
which joints actually have a visible ring, not a blanket improvement across
the board. On the real UR5e data below, gravity and lag are still there
(both baked into every model past the first row of the table), but the
biggest *aggregate* lever turned out to be something else entirely — see
after the table.

### Putting it together — the net effect (real UR5e data, file-level heldout on unseen recordings)

| step | overall R² (0 = no better than assuming no error, 1 = perfect) |
|---|---|
| linear, pooled across joints, no gravity/lag | 0.764 |
| + fit per joint instead of pooled | 0.765 |
| + gravity torque | 0.775 |
| + gradient-boosted trees (+lag), **default hyperparameters** | 0.689 — a *regression*, see §6b |
| + gradient-boosted trees (+lag), **CV-tuned hyperparameters** (**current best**) | **0.820** |

(R² here specifically measures how much of the *tracking gap* — the
millirad-scale error, not the whole trajectory — the model explains; see §4
for why that framing matters and how it differs from a more naive score.
Table numbers: §6c/Changelog 2026-08-20.)

**The honest reading of this table**: per-joint fitting and gravity each
give small, real, monotonic gains (0.764 → 0.765 → 0.775) — consistent with
the physics reasoning above. Switching to trees does *not* automatically
help — with untuned defaults it's a net loss (0.775 → 0.689), because
`HistGradientBoostingRegressor`'s default early stopping validates against
randomly-split, highly-autocorrelated rows instead of genuinely unseen
recordings (§6c). Tuning the tree via file-grouped cross-validation is what
actually delivers the win (0.689 → 0.820) — the single biggest lever in this
table, bigger than gravity, per-joint fitting, or switching model families
by themselves. **The takeaway for a slide: on this dataset, "use a fancier
model" without tuning it properly is not obviously better than a well-built
linear baseline — and might be worse.**

Gravity and lag are each still *targeted* wins, independent of the tuning
story above — each closes a specific, physically-understood gap (gravity:
static sag on the joints that actually carry load against gravity; lag: the
settle-window ring on the joints that visibly exhibit one) rather than
moving every joint a little; see §3 for the joint-by-joint evidence for
each. See §6c for the current model's full per-joint breakdown (§6/§6b are
earlier, now-superseded stages of this same story — kept for the record, not
as the current numbers) and `model_architecture.png` for a diagram of the
whole per-joint pipeline (the architecture is unchanged by hyperparameter
tuning — same 11 features, same per-joint independent-regressor shape;
only what's fitted inside the `Regressor` box differs).

## 1. What we predict

**Position**, not current. `PerJointPositionModel.predicts() -> ["actual_q"]`
— the actual joint angle (rad) a real UR5e ends up at, given a commanded
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

**`PerJointPositionModelNoGravity`** (added 2026-08-19, `--model
linear_per_joint_no_gravity`): the linear model's pre-gravity 7-feature set,
as a permanent, reproducible ablation baseline rather than a historical
number to trust. Only overrides `FEATURE_NAMES`; everything else is
inherited unchanged, which is the whole point of the `FEATURE_NAMES`-driven
refactor above — dropping a feature needs no other code changes. Reran under
the current code and got an exact match to the originally-recorded
pre-gravity numbers (overall R²=0.4368, per-joint bit-for-bit) — confirms
the later split/metric/tree refactors didn't silently change the linear
model's behavior.

## 3. Feature engineering

`FEATURE_NAMES = ["target_current", "qd", "qdd", "qdd_lag4", "qdd_lag16", "qdd_lag64", "pos", "gravity_torque", "vel", "acc", "bias"]`
— eleven features per joint per row:

| feature | source | why it's here |
|---|---|---|
| `target_current` | recorded/commanded current | proxy for commanded torque — the physical driver of elastic deflection (torque -> position error), already logged so free to use |
| `qd` | `target_qd` (commanded velocity) | tracking lag and settle-window ring both scale with how fast the joint is being driven |
| `qdd` | `np.gradient(qd, dt)` | commanded acceleration — same reasoning, and a proxy for the torque term `M(q)qdd` |
| `qdd_lag4`, `qdd_lag16`, `qdd_lag64` | `qdd` held 4/16/64 samples in the past (~31/125/500 ms at ~128 Hz), causal, per-recording — see below | history taps aimed at the settle-window ring — see below |
| `pos` | `target_q` | commanded angle — the residual's reference point, and the input `gravity_torque` (below) is computed from; other pose-dependent physics (inertia at this pose) still isn't computed, see §7 |
| `gravity_torque` | `utils.UR5e.gravity(target_q)[joint]` | direct physical driver of static deflection — see below |
| `vel`, `acc` | the raw `movej` register values (not the realized `qd`/`qdd`) | the *commanded profile shape*, independent of what was actually achieved |
| `bias` | constant 1 | per-joint intercept, see §2 |

This is the same feature basis the old current-predicting baseline used
(plus `gravity_torque` and the `qdd_lag*` taps), carried over for the
position target largely unchanged — it has **not** been re-justified from
scratch for position the way the split/metric/model-shape decisions above
were. That re-justification is open work; see §6.

**`qdd_lag4`/`qdd_lag16`/`qdd_lag64`** (added 2026-08-19): three FIR-style
taps of the commanded acceleration `qdd`, held 4/16/64 samples in the past
(`train_distillation_model._lag_array`, `PerJointPositionModel.LAG_SAMPLES`).
This is this project's first per-row feature with *memory* — the settle-
window ring (a damped oscillation after the commanded motion stops, ~0.3-0.5s
to decay, `bronze_tier/trajectories/worst_move_test-4_shoulder.png`) is a
dynamic, history-dependent effect that no purely instantaneous feature
(static pose, current velocity/acceleration/current) can explain, no matter
how many of those are added — see the "not yet done" item this replaces,
carried in this file's §7 since gravity was added.

Design constraints, both load-bearing (see `_lag_array`/`_lag_block`
docstrings):

- **Causal only.** Row `i`'s lag features look only at rows before `i`.
- **Per-recording, computed before pooling.** Lagging must happen on one
  recording's own `(n, N_JOINTS)` `qdd` array, *before* `_design` stacks rows
  across recordings — lagging the pooled/concatenated array would leak the
  tail of one file into the start of the next.
- **Commanded signal only, never `actual_q`.** `predict()` runs on sim
  trajectories where `actual_*` reads 0.0/is unknown ahead of time, so the
  history has to come from the same `target_qd`-derived quantities every
  other feature already uses, or the feature would be unusable at inference
  time (this is also why `qdd`, not `actual_q`'s derivative, is what gets
  lagged).
- **3 log-spaced taps, not one per sample.** Covering the ring's ~0.3-0.5s
  decay at ~128 Hz densely (one tap per sample) would mean ~40-64 nearly
  collinear columns for a smooth signal; 3 taps log-spaced across that range
  (4/16/64 samples) is a coarse, cheap first cut, not a fitted model of the
  oscillation itself — see below for what a fitted decaying-oscillation term
  would look like instead.

**Result** (held-out row-level, `results/2026-08-19_20-53-55/` linear,
`results/2026-08-19_20-55-23/` tree; both vs. the previous gravity-only run):

| joint | linear R² (w/o lag) | linear R² (w/ lag) | tree R² (w/o lag) | tree R² (w/ lag) |
|---|---|---|---|---|
| overall | 0.445 | **0.613** | 0.883 | **0.921** |
| base | 0.725 | 0.737 | 0.734 | 0.787 |
| shoulder | 0.398 | **0.597** | 0.914 | **0.951** |
| elbow | 0.462 | 0.487 | 0.637 | **0.704** |
| wrist1 | 0.914 | 0.917 | 0.834 | 0.840 |
| wrist2 | 0.840 | 0.841 | 0.798 | 0.807 |
| wrist3 | 0.022 | 0.028 | 0.147 | 0.171 |

Moved almost exactly where the hypothesis predicted: **shoulder — the joint
with the biggest observed ring — gained the most** on both models (linear
+0.20, tree +0.04, already near its ceiling), elbow gained meaningfully on
the tree (+0.07), and wrist1/wrist2/base (small rings, per §3's earlier
gravity discussion) moved only slightly, consistent with a real dynamic
effect being captured rather than noise. wrist3 stayed at its noise floor
(§6), as expected. Full current numbers in §6.

**`gravity_torque`** (added 2026-08-19): `utils.UR5e.gravity(q)` returns all
six joints' torques from one full-pose call — a joint's own `pos` alone
isn't enough, since gravity torque on any joint depends on the whole
kinematic chain's configuration. Computed once per recording (not once per
joint) via the new `utils.UR5e.gravity_batch` and sliced per joint;
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

~~Lag/history features~~ — done, see the `qdd_lag*` writeup above (2026-08-19).

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

## 6. UR10e-era results (superseded — see §6b for real UR5e numbers)

**These numbers describe the old UR10e placeholder data** (see the
Changelog's "Switched from UR10e to UR5e" entry and `UR5eMigration.md`).
Kept below for historical reference only — §6b has the current, real-UR5e,
file-level-heldout comparison across all three models, which is what should
be quoted or trusted now.

Last run on UR10e data: 2026-08-19, `results/2026-08-19_20-53-55/` linear,
`results/2026-08-19_20-55-23/` tree.

7 files, 20% row holdout (216,099 held-out rows/joint). With `gravity_torque`
+ the `qdd_lag4/16/64` history taps (previous gravity-only run without lag in
parentheses):

| joint | held-out RMSE (deg) | held-out R² |
|---|---|---|
| overall | 0.0201 | **0.613** (was 0.445) |
| base | 0.0075 | 0.737 (was 0.725) |
| shoulder | 0.0460 | **0.597** (was 0.398) |
| elbow | 0.0150 | 0.487 (was 0.462) |
| wrist1 | 0.0052 | 0.917 (was 0.914) |
| wrist2 | 0.0035 | 0.841 (was 0.840) |
| wrist3 | 0.0019 | 0.028 (was 0.022) |

In-sample (100%-refit model, scored on its own training data) matches the
held-out numbers almost exactly, as before (see §5's caveat).

Reading these: shoulder — the joint with the largest observed settle-window
ring — gained the most of any joint (R² 0.398→0.597, the biggest single-joint
jump this file has recorded), exactly what §3's lag-feature hypothesis
predicted. elbow gained more modestly (0.462→0.487); wrist1/wrist2/base
(small rings) barely moved, and wrist3 stayed at its noise floor — all
consistent with the taps capturing a real dynamic effect rather than fitting
noise. Shoulder and elbow are still the weakest non-noise-floor joints,
though: 3 coarse FIR taps are a first cut, not a fitted model of the ring
(see §3's design-constraints note) — a decaying-oscillation basis term is the
likely next step if more of this gap is worth closing.

**Tree vs. linear, both with the same 11 features** (`results/2026-08-19_20-55-23/`
tree, `results/2026-08-19_20-53-55/` linear; both 20% row holdout, same split).
Previous (pre-lag, gravity-only) numbers in parentheses:

| joint | linear R² | tree R² | linear RMSE (deg) | tree RMSE (deg) |
|---|---|---|---|---|
| overall | 0.613 (0.445) | **0.921** (0.883) | 0.0201 (0.0242) | **0.0091** (0.0111) |
| base | 0.737 (0.725) | **0.787** (0.734) | 0.0075 (0.0076) | 0.0067 (0.0075) |
| shoulder | 0.597 (0.398) | **0.951** (0.914) | 0.0460 (0.0561) | **0.0160** (0.0213) |
| elbow | 0.487 (0.462) | **0.704** (0.637) | 0.0150 (0.0153) | **0.0114** (0.0127) |
| wrist1 | **0.917** (0.914) | 0.840 (0.834) | **0.0052** (0.0053) | 0.0072 (0.0074) |
| wrist2 | **0.841** (0.840) | 0.807 (0.798) | **0.0035** (0.0035) | 0.0038 (0.0039) |
| wrist3 | 0.028 (0.022) | 0.171 (0.147) | 0.0019 (0.0019) | **0.0017** (0.0017) |

Both models improved on every joint after adding the lag taps — the tree's
overall R² is now 0.921, and its gains are concentrated on the same joints
the linear model's are (shoulder +0.037, elbow +0.067, base +0.053), which is
itself evidence the taps carry real signal rather than the tree just having
more capacity to overfit them: if the taps were noise, a
`HistGradientBoostingRegressor` would be the model most likely to exploit
that, and it wouldn't move in the same joint-specific pattern the physically-
motivated hypothesis predicted. **Linear still wins on wrist1 and wrist2**
even with identical features on both models — a genuine regressor difference
on those two joints, not a missing-feature artifact (as already established
before lag was added, see the entry below). Read together with how little gravity
moved the tree overall (+0.0015 vs. linear's +0.008, see §2), the likely
story: wrist1/wrist2's residual is close to a smooth, near-linear function
of the inputs, which a linear model represents natively and a
piecewise-constant tree (with untuned default hyperparameters) approximates
less efficiently — while shoulder/elbow/base's residual is nonlinear enough
that the tree's flexibility wins outright. Different joints may simply want
different model shapes; not tested here (would mean per-joint model
selection, not just a per-joint *fit*, which both models already do).
`models/distill_tree.pkl` saved alongside `models/distill.pkl`. **Update
(2026-08-19, later the same day): `PerJointTreeModel` is now the shipped
default** — retrained with `--model tree_per_joint --out models/distill.pkl`
(`results/2026-08-19_21-06-24/`, overall R²=0.9195, essentially identical to
the `distill_tree.pkl` numbers above; the ~0.001 difference is run-to-run/
environment noise, not a code or feature change). `--model`'s CLI *default*
flag is still `linear_per_joint` (nothing forces `tree_per_joint` from the
command line) — this is about which pickle currently sits at the path
`run.py`/`train_rla.py` load by default, not the CLI flag's default value.

**Caveat on all of the above, not yet resolved**: these are still row-level
held-out numbers (§5), and the leakage concern applies *more* to the tree
model than the linear one. `HistGradientBoostingRegressor` has far more
capacity than an 11-parameter-per-joint linear fit, so it can exploit
"this held-out row's neighbors are in the training set" more effectively —
the tree's R²=0.92 could be sitting on more leakage-driven optimism than the
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

## 6b. Real UR5e data + file-level heldout comparison (2026-08-19)

`data/` was populated with real UR5e recordings (101 files: `T01`–`T08`,
`T11` fast/medium/slow × 2 reps, plus `reach_extend_*`, `short_moves_*`,
`workspace_sweep_*`, `wrist_sweep_*` sweeps) and a `data/heldout/`
subdirectory (`T09`, `T10`, 12 files) that is **never trained on** — the
first genuinely unseen-trajectory test this project has had. Two code
changes were needed before this data could be used at all:

- `DEFAULT_CSVS` globbed `data/test-*.csv` (the old UR10e naming) — no file
  matches that pattern anymore, so the default run trained on zero
  recordings. Changed to glob `data/*.csv`; since `glob.glob` is
  non-recursive this still never picks up `data/heldout/*.csv`.
- Added `DEFAULT_HELDOUT_CSVS` (`data/heldout/*.csv`) and a `--heldout-csvs`
  flag to `main()`: after the model is refit on 100% of `--csvs`, it's also
  scored with `_evaluate_model`/`_compute_metrics` on the heldout
  recordings — files the model never saw in any form, unlike
  `_row_split_eval`'s row-level split (rows from the *same* trajectories the
  model trained on, §5's documented caveat). Results land in
  `log.json`'s new `heldout_files`/`heldout_metrics` keys, a `heldout/`
  plot subfolder per run, and a dedicated `runs_summary_heldout.csv` /
  `comparison_plot_heldout.png` so the three models' unseen-trajectory
  numbers accumulate in one place. This is the leave-one-file-out-style
  diagnostic §7 item 4 had been asking for — done via a fixed held-out file
  set rather than full leave-one-file-out CV (cheaper, and the recordings
  already come pre-split into a training pool and a held-out pool).

All three registered models were retrained on the 101 training files and
scored on the 12 heldout files (original run folders since deleted from
disk — see `results/runs_summary_heldout.csv` for the preserved numbers, or
rerun `python train_distillation_model.py --model <name>` for a fresh,
numerically-equivalent folder: no-gravity/gravity reproduced at
`results/2026-08-20_10-40-55/`/`results/2026-08-20_10-44-06/`; the untuned
tree's exact config is no longer the default post-§6c, see §6c):

| joint | linear (no gravity) R² | linear (+gravity) R² | tree (+gravity+lag) R² |
|---|---|---|---|
| overall | 0.765 | **0.775** | 0.689 |
| base | 0.883 | **0.886** | 0.772 |
| shoulder | 0.620 | **0.644** | 0.594 |
| elbow | 0.679 | **0.693** | 0.634 |
| wrist1 | 0.520 | **0.527** | 0.492 |
| wrist2 | 0.359 | **0.360** | 0.316 |
| wrist3 | 0.917 | **0.920** | 0.806 |

**Two findings, one expected and one not:**

1. **Gravity still helps, consistently** — `linear (+gravity)` beats
   `linear (no gravity)` on every single joint, matching §3's original
   physics-based reasoning (small but real gains, largest on
   shoulder/elbow/wrist3).
2. **The tree model loses to *both* linear models, on every joint, on
   both splits** — the opposite of the UR10e-era §6 result, where the tree
   won by a wide margin (0.921 vs. 0.613 overall) on a row-level split. On
   this real UR5e data the tree is behind even on its own row-level split
   (overall R²=0.777 vs. linear-gravity's 0.881, preserved in
   `results/runs_summary.csv`'s row-level numbers for that run — its
   `log.json` folder no longer exists on disk), not just the file-level one
   (0.689 vs. 0.775).

   **This turned out not to be the leakage story §6/§7 item 4 predicted**:
   if the tree's row-level number were mostly leakage-driven optimism, its
   row→file R² drop should be *larger* than the linear models' (more
   capacity to exploit "this held-out row's neighbors are in the training
   set," per §6's original concern). Measured, it's the opposite — tree
   drops 0.777→0.689 (−0.089), linear-gravity drops 0.881→0.775 (−0.107),
   linear-no-gravity drops 0.877→0.765 (−0.112). The tree's row-level
   number was *not* especially inflated by leakage relative to linear's;
   it's just a worse fit to this dataset's residual on both splits. A
   plausible explanation: `HistGradientBoostingRegressor`'s untuned
   defaults (particularly its internal early-stopping validation split,
   sized for the ~1e6-row UR10e-era dataset) may not suit this dataset's
   size/noise characteristics — this hasn't been tested, and is the
   natural next step before writing the tree model off, rather than
   concluding trees are categorically worse here. **Practical read for
   now**: on this real-UR5e dataset, with default hyperparameters, the
   linear+gravity model is the better choice — the opposite conclusion
   from the UR10e-era recommendation, and not yet fully explained.
   **Update:** the untuned-defaults hypothesis was tested and confirmed —
   see §6c, where a tuned tree overtakes linear+gravity on every joint.

`models/distill.pkl` (the path `run.py`/`train_rla.py` load by default)
has **not** been repointed as part of this — the three models are saved
separately as `models/distill_linear_no_gravity.pkl`,
`models/distill_linear_gravity.pkl`, `models/distill_tree.pkl`, and
`models/distill.pkl` still holds the old UR10e-trained tree model from the
prior Changelog entry. Repointing it to `distill_linear_gravity.pkl` (this
comparison's winner) is a follow-up decision, not made here.

## 6c. Tree hyperparameter tuning + Random Forest (2026-08-20)

Followed up on §6b's open hypothesis (untuned `HistGradientBoostingRegressor`
defaults, sized for the old ~1e6-row UR10e dataset, don't suit the smaller/
noisier real-UR5e one) and added a bagging alternative.

**Root cause, more specifically than §6b could say:** `PerJointTreeModel`
set nothing but `random_state=0` — full sklearn defaults. Worse than just
"untuned," `HistGradientBoostingRegressor`'s default `early_stopping="auto"`
(active above 10k samples, true here) validates against a *random split of
the pooled training rows*. Rows are ~8ms apart and highly autocorrelated
within a recording (more so with the `qdd_lag*` taps in play), so that
internal validation signal is systematically easier/more optimistic than
this project's real generalization target — an unseen *recording*
(`data/heldout/*.csv`). The tree's stopping criterion and its actual
evaluation criterion were measuring different things.

**Fix:** `PerJointPositionModel._design` now returns a third element per
joint, `groups` (the source-recording index for every row), so a
hyperparameter search can use `sklearn.model_selection.GroupKFold` — CV
folds split by whole recording, matching the file-level heldout evaluation
methodology instead of the leaky row-level one. New standalone script
`tune_tree_hyperparams.py` runs a per-joint `RandomizedSearchCV` with
`GroupKFold` for `tree_per_joint` and `forest_per_joint`.

**Tree search** (all 101 training files, 25 candidates × 5 folds/joint,
`early_stopping=False` fixed so `max_iter` is tuned explicitly instead of
picked by the mismatched internal split): all six joints' searches
independently converged on the *same* hyperparameters —
`max_iter=300, learning_rate=0.2, max_leaf_nodes=63, min_samples_leaf=20,
l2_regularization=1.0` — with file-grouped CV R² of 0.87–0.96 per joint
(`results/tuning_tree_per_joint_2026-08-20_08-27-23.json`). `max_iter` and
`max_leaf_nodes` landed on the *edge* of the search grid, so there may be
more headroom with a wider grid — not chased further here. These became
`PerJointTreeModel.HGB_KWARGS`.

**Forest search — abandoned, replaced with a cheap smoke test:** the
equivalent full-dataset `RandomForestRegressor` search (started at
`n_estimators` up to 500, `max_depth` including unbounded) took ~17 minutes
for the *first* joint alone — unbounded-depth trees are expensive to grow on
~370k rows, and the estimated ~1.5–2 hours for all six joints wasn't worth
it relative to the gain expected (forests are far less hyperparameter-
sensitive than boosting — no learning-rate/iteration-count interaction to
get wrong). Replaced with a 20-file, 3-candidates×3-folds smoke test
(`results/tuning_forest_per_joint_2026-08-20_09-42-12.json`): `max_depth`
capped at 16, `min_samples_leaf=5`, `max_features=0.5` plateaued for the
higher-signal joints (base/shoulder/elbow, R² 0.86–0.92 on that small
sample); wrist joints were inconclusive on only 20 files. These, plus a
fixed `n_estimators=200`, became `PerJointForestModel.RF_KWARGS`. A proper
file-grouped search (smaller candidate/fold count, or a row-count cap) is
the natural follow-up if more headroom is worth the wait.

**Result — both changes flip the §6b conclusion.** Retrained on the 101
training files, scored on the same 12 never-trained-on heldout files:

| joint | linear (+gravity) R² | tree, untuned (§6b) | tree, tuned | forest |
|---|---|---|---|---|
| overall | 0.775 | 0.689 | **0.820** | 0.809 |
| base | 0.886 | 0.772 | **0.915** | 0.911 |
| shoulder | 0.644 | 0.594 | **0.745** | 0.698 |
| elbow | 0.693 | 0.634 | 0.726 | **0.740** |
| wrist1 | 0.527 | 0.492 | **0.584** | 0.584 |
| wrist2 | 0.360 | 0.316 | 0.385 | **0.400** |
| wrist3 | 0.920 | 0.806 | **0.931** | **0.931** |

(original run folders `results/2026-08-20_10-18-44/`/`results/2026-08-20_10-22-42/`
since deleted from disk; re-run and reproduced at
`results/2026-08-20_10-45-56/` tuned tree, `results/2026-08-20_10-56-46/`
forest — tree R² came back 0.816 on the re-run, not 0.820: `HistGradientBoostingRegressor`
isn't perfectly bit-reproducible run-to-run even with a fixed `random_state`,
due to floating-point summation order in its multi-threaded histogram
building; forest reproduced exactly, since `RandomForestRegressor` has no
such nondeterminism. Table below keeps the original numbers; expect ~0.004
R² of run-to-run noise on the tree specifically.) Both tuned tree-based
models now beat linear+gravity on **every
joint** — confirming §6b's hypothesis: the tree family was never worse than
linear here, the boosted model's defaults just didn't fit this dataset's
size/noise and its early-stopping validation split was measuring the wrong
thing. The tuned HGB tree is the best model overall and the model of choice
if only one is shipped; forest is close behind and, being far less
hyperparameter-sensitive, is the safer choice if this dataset changes size/
composition again before there's time to re-tune. Neither `models/distill.pkl`
nor the §6b "which model to ship" follow-up was changed here — this section
is a hyperparameter/model-family comparison, not a repointing decision.

## 7. Not yet done, in priority order

1. ~~Gravity torque feature~~ — done, see §3/§6 (2026-08-19).
2. ~~Lag/history features for the settle-window ring~~ — done, `qdd_lag4/16/64`
  FIR taps, see §3/§6 (2026-08-19). Closed most of the gap on shoulder
  (R² 0.40→0.60 linear, 0.91→0.95 tree) and some on elbow (0.46→0.49 linear,
  0.64→0.70 tree); shoulder/elbow are still the weakest non-noise-floor
  joints, so a fitted decaying-oscillation term (rather than 3 fixed FIR
  taps) is the natural next step if more of this gap is worth closing —
  demoted to a new item 8 below rather than reopening this one, since the
  cheap version is done and shipped.
3. **Standardized coefficients in `coefficients.png`** — cheap, and now
  demonstrated as a real gap (§3): `gravity_torque`'s raw weight rounds to
  `0.0000` in the plot despite measurably improving R², because it's in
  Nm-scale units against a millirad-scale target. Show `coef_j × std(feature_j)`
  instead so bar heights are comparable across features regardless of native
  units; doesn't change the fit, only the display.
4. ~~Leave-one-file-out CV diagnostic~~ — done differently than originally
  scoped: rather than full leave-one-file-out CV, added a fixed
  `data/heldout/*.csv` file set (`--heldout-csvs`) never trained on, scored
  after the model's refit on 100% of the training files. Gives the same
  honest file-level generalization number alongside the row-level holdout
  (§5's caveat) at lower cost, since the recordings already arrived
  pre-split into a training pool and a held-out pool. See §6b for the
  result — it overturned the tree-vs-linear recommendation from §6.
5. ~~A non-linear regressor~~ — done, `PerJointTreeModel`, see §2/§6
  (2026-08-19), **out of this list's stated order** (done before item 2, at
  the user's explicit direction). The predicted caveat held at the time: the
  tree model couldn't see history it wasn't given, so gravity alone didn't
  fix the settle-window ring. Since resolved by item 2 above — the lag taps
  were added on top of the tree model too (both models share
  `FEATURE_NAMES`), and both improved together (§6).
6. ~~`dynamics.Dynamics.frame()` doesn't emit an `actual_q` placeholder
  column~~ — checked (2026-08-19) and it's **not actually needed**: verified
  directly that `utils.set_block` creates brand-new columns correctly (not
  just overwrites existing ones), so `train_rla.py`'s `evaluate()` (which
  calls `model.predict(frame)` then `set_block(frame, base, ...)`) already
  works on a `Dynamics.frame()` output missing `actual_q*` entirely —
  `model.predict()` only reads `target_*` columns, never `actual_q`, so
  nothing needs it to pre-exist. This item had been carried forward in this
  list without re-verifying; corrected now. Optional, not required: adding
  an `actual_q{j} = 0.0` placeholder to `frame()` anyway would make the
  candidate frame's schema match a real recording's before the model runs,
  for readability — but it fixes nothing that's actually broken.
7. `train_rla.py`/`run.py` still wired to `CurrentGapMetric`/`actual_current`
  — need to switch to `PositionGapMetric` before they'll run against a model
  from this file. **Also needs `SCORE_WEIGHT`/`CYCLE_WEIGHT` (and
  `PATH_SCORE_WEIGHT`/`PATH_CYCLE_WEIGHT`) reconsidered when this happens**:
  `CurrentGapMetric`'s score is in amps (order ~1-10 after aggregation),
  comparable to `cycle_time` in seconds — `PositionGapMetric`'s score is in
  radians (order ~0.01), 100-500x smaller. Swapping the metric class without
  rescaling/reweighting would make `OBJECTIVE ≈ cycle_time` alone — the
  agent would optimize almost purely for speed, silently dropping the
  vibration-minimization term instead of balancing it.
8. **A fitted decaying-oscillation term for the ring**, in place of (or
  alongside) the fixed `qdd_lag*` FIR taps — item 2's taps closed most, not
  all, of shoulder/elbow's gap to the other joints (§6). Would need the
  ring's natural frequency/decay estimated first (FFT or autocorrelation on
  the residual, already useful for confirming wrist3 is noise-floor per §6),
  then a feature like `e^-t'/τ · sin(ωt')` keyed off time since the commanded
  move stopped. Still fits with plain `lstsq` if linear-in-the-fitted-basis;
  lower priority than items 3/4/7 since the FIR taps already captured most of
  the effect cheaply.
9. **Repoint (or fix) `models/distill.pkl`** — the file `run.py`/`train_rla.py`
  load by default was never updated past an old UR10e-era pickle, and as of
  2026-08-20 it **fails to load at all** (`ModuleNotFoundError: No module
  named '_loss'`) under the installed scikit-learn (1.9.0) — an old pickle's
  internal module layout no longer matches. Anyone running the pipeline
  end-to-end right now hits this. Fix: either repoint it at
  `models/distill_tree_tuned.pkl` (the current best model, §6c) via a fresh
  `train_distillation_model.py --model tree_per_joint --out models/distill.pkl`,
  or retrain/re-save whichever model should actually ship. Not done here —
  §0 only documents the discrepancy.

## Changelog

- **2026-08-20** — Every `results/<timestamp>/` folder from earlier today
  and 2026-08-19 (no-gravity, +gravity, untuned tree, tuned tree, forest,
  first pooled-linear run) was found deleted from disk mid-session — outside
  any command run here; `runs_summary.csv`/`runs_summary_heldout.csv` still
  had every row, only the per-run folders (plots/log.json/pickle) were gone.
  Re-ran the four presentation models (`train_pooled_linear.py`;
  `train_distillation_model.py --model {linear_per_joint_no_gravity,
  linear_per_joint,tree_per_joint}`) and `--model forest_per_joint`, landing
  in fresh folders (`results/2026-08-20_10-39-13/` pooled,
  `.../10-40-55/` no-gravity, `.../10-44-06/` gravity, `.../10-45-56/` tuned
  tree, `.../10-56-46/` forest) with numbers matching the deleted originals
  to within floating-point noise (`HistGradientBoostingRegressor`'s
  multi-threaded histogram building isn't perfectly bit-reproducible run to
  run even with a fixed `random_state` — tree R² came back 0.816 vs. the
  original 0.820; every other model reproduced exactly). Updated every
  `results/<old-timestamp>/` path reference in §6b/§6c/below to either the
  new folder or a note that the folder is gone but the number is preserved
  in the summary CSVs. **The untuned tree's exact original run has no
  replacement folder** — its config (default `HistGradientBoostingRegressor`
  hyperparameters) is no longer what `PerJointTreeModel` builds post-§6c, so
  reproducing that folder would require temporarily reverting `HGB_KWARGS`;
  not done, since the number is already fully preserved in §6b's text and
  `runs_summary(_heldout).csv`.
- **2026-08-20** — Brought §0 (the plain-language, slide-ready summary) back
  in sync with the real-UR5e results: it had still been describing the old
  synthetic-UR10e R² progression (0.44→0.88→0.92→0.92) and calling
  `models/distill.pkl` "the shipped model" — both stale, and the second one
  actively wrong (`distill.pkl` was never repointed off an old UR10e-era
  pickle and, checked directly, now **fails to load** under the installed
  scikit-learn 1.9.0; see new §7 item 9). Rewrote the opening framing, the
  "why gradient-boosted trees" paragraph, and the closing net-effect table to
  the current real-UR5e, file-level-heldout numbers (pooled linear 0.764 →
  per-joint 0.765 → +gravity 0.775 → tree/default-hyperparameters 0.689,
  a regression → tree/CV-tuned 0.820), with an explicit "tuning mattered more
  than model family" takeaway that the old table's clean monotonic story
  didn't have room for. Also updated `plot_model_diagram.py`'s regressor box
  (`linear least-squares OR gradient-boosted trees` → `... OR random forest`,
  now that `PerJointForestModel` exists) and regenerated
  `model_architecture.png`; noted in both the script's docstring and §0 that
  `PooledLinearModel` is deliberately not shown in that diagram, since it
  shares one fit across all six joints instead of the "×6 independent"
  architecture the diagram depicts.
- **2026-08-20** — Added `PooledLinearModel` (`train_distillation_model.py`)
  for a presentation comparison: one shared-slope `lstsq` fit across all six
  joints' stacked rows (6-column joint one-hot standing in for a per-joint
  intercept), same no-gravity/no-lag 6-feature set as
  `PerJointPositionModelNoGravity`, isolating the per-joint-vs-pooled
  variable specifically. Mirrors this project's original `LinearModel`
  (removed in `062a315`, which predicted `actual_current` on the old UR10e
  data) retargeted at `actual_q` on the current real-UR5e data. Not
  registered in `MODELS`/`--model` — `main()`'s row-level diagnostic assumes
  a per-joint-keyed design, which a single stacked-across-joints fit doesn't
  produce — so it's trained via a new standalone script,
  `train_pooled_linear.py`, with its own row-level held-out helper
  (`_pooled_row_split_eval`) that reconstructs per-joint metrics from the
  flat design so it still plugs into the existing `_compute_metrics`/
  `log_run`. Result (original folder `results/2026-08-20_10-36-15/` since
  deleted from disk; re-run and reproduced exactly at
  `results/2026-08-20_10-39-13/`, file-level heldout): overall R²=0.764,
  essentially tied with `PerJointPositionModelNoGravity`'s
  0.765 and slightly behind it on every joint but wrist3 — a much smaller
  per-joint-fitting gap on the position target than the original current-target
  pooled-vs-per-joint comparison in §2 (R² 0.43→0.60 for wrist2) found, but
  directionally the same (pooling never wins).
- **2026-08-20** — Followed up §6b's open "untuned tree defaults" hypothesis
  and added a Random Forest alternative — see §6c for the full writeup.
  `PerJointPositionModel._design` now returns a per-row recording-group id
  (`{joint: (X, y, groups)}`, was `(X, y)`) so a hyperparameter search can use
  file-grouped CV (`GroupKFold`) instead of the leaky row-level split
  `HistGradientBoostingRegressor`'s own default early stopping was
  implicitly using. New standalone script `tune_tree_hyperparams.py` runs a
  per-joint `RandomizedSearchCV` for `tree_per_joint`/`forest_per_joint`.
  `PerJointTreeModel` gained a tuned `HGB_KWARGS` class attribute
  (`early_stopping=False`, plus CV-selected `max_iter`/`learning_rate`/
  `max_leaf_nodes`/`min_samples_leaf`/`l2_regularization`) threaded into both
  its `HistGradientBoostingRegressor` call sites. Added `PerJointForestModel`
  (`RandomForestRegressor` per joint, registered as `forest_per_joint` in
  `MODELS`) with a smoke-tested `RF_KWARGS` — its own full file-grouped
  search was abandoned as too slow for this dataset (~17 min for a single
  joint), see §6c for why and what replaced it. Both changes reverse §6b's
  conclusion: tuned tree (heldout R² 0.820) and forest (0.809) now beat
  linear+gravity (0.775) on every joint, where the untuned tree (0.689) had
  lost to it on every joint. `models/distill.pkl` was **not** repointed —
  saved separately as `models/distill_tree_tuned.pkl`/`distill_forest.pkl`,
  pending a ship decision.
- **2026-08-19** — Real UR5e recordings landed in `data/` (101 training
  files) plus a never-trained-on `data/heldout/` (12 files, `T09`/`T10`) —
  see §6b for the full writeup. Fixed `DEFAULT_CSVS`, which globbed the old
  `data/test-*.csv` naming and matched nothing on the new files (`T01_*`,
  `reach_extend_*`, etc.), to glob `data/*.csv` instead (still excludes
  `data/heldout/*.csv` since `glob.glob` is non-recursive). Added
  `DEFAULT_HELDOUT_CSVS`/`--heldout-csvs` to `train_distillation_model.py`:
  after the model refits on 100% of `--csvs`, it's also scored on the
  heldout files via the existing `_evaluate_model`/`_compute_metrics` (no
  new eval code needed), logged into `log.json`'s new
  `heldout_files`/`heldout_metrics` keys, plotted into a `run_dir/heldout/`
  subfolder, and appended to a new `runs_summary_heldout.csv`/
  `comparison_plot_heldout.png` (`_update_summary` gained an optional
  `suffix` param for this). This is §7 item 4's leave-one-file-out
  diagnostic, done via a fixed held-out set instead of full CV since the
  data already arrived pre-split. Retrained all three registered models
  (`linear_per_joint_no_gravity`, `linear_per_joint`, `tree_per_joint`) on
  the real data and compared them on the heldout files — the tree model,
  the UR10e-era winner by a wide margin, now **loses to both linear models
  on every joint, on both the row-level and file-level splits**; the
  row→file R² drop is if anything *smaller* for the tree (−0.089) than for
  linear (−0.107), so this isn't the leakage-optimism story §6 had
  predicted — more likely the tree's untuned
  `HistGradientBoostingRegressor` defaults don't suit this dataset, not yet
  investigated. `models/distill.pkl` (the default pipeline path) was
  **not** repointed — the three new models are saved separately
  (`models/distill_linear_no_gravity.pkl`, `distill_linear_gravity.pkl`,
  `distill_tree.pkl`) pending a decision on which one ships.
- **2026-08-19** — Switched from UR10e to UR5e (the user's real robot) on
  this branch, bringing over the physics fix already applied on
  `apostolosRLA` (see `UR5eMigration.md`, copied over from that branch, for
  the full writeup and before/after parameter tables). Renamed `utils.UR10e`→
  `utils.UR5e`, `dynamics.UR10eDynamics`→`UR5eDynamics`, and this file's
  `_UR10E`→`_UR5E` singleton (the `gravity_torque` feature's underlying
  `gravity_batch` call). Replaced `utils.py`'s DH/mass/COM/inertia constants
  with UR5e's published values (Universal Robots' own DH-parameters page,
  fetched live, cross-checked against the still-present UR10e numbers before
  trusting it). Changed `DEFAULT_CSVS` from a hardcoded `range(1,8)` to a
  glob over `data/test-*.csv`. **This directly affects `gravity_torque`**:
  that feature is *computed* from `utils.py`'s hardcoded robot geometry
  (forward kinematics/dynamics), not read from the recording CSV — so even
  though every other feature (`target_current`, `qd`, `qdd`, `pos`, `vel`,
  `acc`, the `qdd_lag*` taps) is pulled straight from whatever data you feed
  `fit()`, `gravity_torque` was silently computed as "what a UR10e-shaped arm
  would feel at these joint angles" until this fix, even on real UR5
  recordings. **`data/test-*.csv` still holds UR10e recordings as of this
  entry** — every result in §6/§8/§0 above still describes the UR10e-trained
  model; retrain once real UR5 data replaces them (delete the old UR10e
  files first — the new glob-based `DEFAULT_CSVS` would otherwise pool both
  robots' data into one fit).
- **2026-08-19** — Added §0, a plain-language, slide-ready summary of the
  shipped model (trees + gravity + lag) for a non-ML-expert engineering
  audience: the residual trick, why per-joint, why trees over a linear
  formula, why gravity and lag each help (with the manipulator-dynamics
  `M(q)q̈ + C(q,q̇)q̇ + g(q) = τ` equation and the `qdd_lag_k` definition),
  and a step-by-step R² table (0.44 → 0.88 → 0.92 → 0.92) tying each addition
  to a specific, physically-understood gap it closed. Also corrected a stale
  claim in §2/§6: `models/distill.pkl` (the path `run.py`/`train_rla.py` load
  by default) had said "linear model, still the default" but was retrained
  by the user directly (outside this file's usual training-run trail) with
  `--model tree_per_joint --out models/distill.pkl` — `PerJointTreeModel`
  (gravity + lag) is now the shipped default (`results/2026-08-19_21-06-24/`,
  overall R²=0.9195, matching the session's separate `distill_tree.pkl` run
  to within run-to-run noise). Also regenerated `model_architecture.png`
  (`plot_model_diagram.py`) to show all 11 features, including the new
  `qdd_lag4/16/64` taps (highlighted the same way `gravity_torque` already
  was, as a hand-engineered addition rather than a raw-logged signal) — the
  diagram had been left showing the old 8-feature version after the lag
  features were added earlier the same day.
- **2026-08-19** — Added `qdd_lag4`/`qdd_lag16`/`qdd_lag64` (causal FIR taps
  of commanded acceleration at 4/16/64 samples in the past, ~31/125/500 ms)
  to `PerJointPositionModel.FEATURE_NAMES` — this project's first feature
  with memory, aimed at the settle-window ring a purely instantaneous
  per-row model structurally can't reproduce (§3). Added `_lag_array` (causal,
  edge-holds the first `k` rows) and `PerJointPositionModel._lag_block`/
  `_lag_cols_for_joint` (mirrors `_gravity_block`'s "only pay for what's
  used" pattern), computed per-recording *before* `_design` pools rows across
  recordings so lags never cross a file boundary, and built only from
  `target_qd`-derived history (never `actual_q`, unavailable at `predict()`
  time on a sim trajectory). `PerJointTreeModel` inherits the new features
  unchanged (deliberately not overridden, same reasoning as gravity).
  `PerJointPositionModelNoGravity` deliberately does **not** get them — it's
  pinned to the historical pre-gravity 7-feature baseline; reran it and
  confirmed its numbers are bit-for-bit unchanged (R²=0.4368 overall),
  confirming the lag-feature plumbing doesn't leak into models that don't
  request it. Results (held-out row-level, `results/2026-08-19_20-53-55/`
  linear, `results/2026-08-19_20-55-23/` tree): overall R² 0.445→0.613
  (linear), 0.883→0.921 (tree); shoulder — the joint with the biggest
  observed ring — gained the most on both (0.398→0.597 linear,
  0.914→0.951 tree), matching the hypothesis. Added item 8 to §7 (a fitted
  decaying-oscillation term) as the natural next step if more of
  shoulder/elbow's remaining gap is worth closing.
- **2026-08-19** — Corrected §7: the "`dynamics.Dynamics.frame()` needs an
  `actual_q` placeholder" item had been carried forward unverified since
  early in the session (a version of it was actually checked and ruled out
  much earlier, but the conclusion never made it back into this file).
  Re-verified directly (`set_block` creates new columns correctly) and
  marked it not required. Also added the `SCORE_WEIGHT`/`CYCLE_WEIGHT`
  scale-mismatch warning to item 7 (`PositionGapMetric`'s score is ~100-500x
  smaller than `CurrentGapMetric`'s, relative to `cycle_time` — a naive
  metric swap would silently break the RL objective's balance) — this was
  flagged in passing back when the metric was first added but never written
  down as a concrete blocker until now.
- **2026-08-19** — Added `PerJointPositionModelNoGravity` (`--model
  linear_per_joint_no_gravity`) as a permanent, reproducible ablation
  baseline (7-feature `FEATURE_NAMES` override, nothing else) rather than
  relying on an older recorded number. Reran it and got an exact match to
  the historical pre-gravity result (R²=0.4368 overall, per-joint
  bit-for-bit) — confirms later refactors (the `_row_split_fit_predict`
  hook, the `bounds()` fix, the tree model) didn't change the linear model's
  behavior.
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
