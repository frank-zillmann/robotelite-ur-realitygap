# Case 2 — Reality Gap Modeling & RL Motion Optimization

Working memory for this folder. Read this first, then `README.md` (setup +
run commands) and `REVIEW.md` (deep pipeline walkthrough, file-by-file notes,
findings from actually running it, tier-by-tier task breakdown) for depth.
This file stays short — it's a status board and change log, not a copy of
those two docs.

## Maintaining this file (read this before editing case 2 code)

Whenever a change is made to anything in `case 2/` — a swapped-in model, a
new metric, a bug fix, a retrained agent worth noting, a decision about
scope — add a dated entry to the **Change Log** below before finishing that
piece of work. Keep entries short: what changed, why, and which file. If the
change affects tier status (e.g. Silver's model now beats the baseline) or
invalidates a "known issue" below, update those sections too, don't just log
it and leave the stale state in place.

## What this case is

Bridge the gap between URSim (zero tracking error, by construction) and a
real UR10e (overshoot + ringdown after each move, worse at higher
speed/acceleration). Three layers: characterize the gap from real RTDE
recordings (Bronze), learn to predict it (Silver), then use RL to pick
speed/acceleration that moves fast while minimizing vibration, validated
against the learned model and ideally real hardware (Gold/Diamond). Full
brief: `../2026-robotelite-sdu-dk-6_ur.pdf` (or the case index PDF at repo
root).

## Repo map (this folder)

| File | Role |
|---|---|
| `data/test-*.csv` (+`.script`) | real UR10e RTDE recordings, varied vel/acc/blend — Bronze dataset |
| `analysis.py` | `Recording` loader + per-joint stats/plots (current channel) |
| `plot_target_actual.py` | quick-look target-vs-actual plotter for any channel (`q`, `qd`, `current`, `TCP_pose`, `TCP_speed`) |
| `channel_gap_bar_chart.py` | peak+RMS \|actual - target\| per channel/component, 3 views under `bronze_tier/channel_gap/` (full run, settle-window-only, worst-3-by-component) |
| `common.py` | `segments()` — splits a recording into per-`movej` segments |
| `pca_feature_analysis.py` | PCA (numpy SVD) over each output's candidate feature set — scree/loadings/importance/PC1-PC2 plots per output, informs which features `DistillModel` should use, not run by the main pipeline |
| `record.py` / `send.py` | passive RTDE logger / push a script or path and record it |
| `dynamics.py` | `Dynamics` interface + `UR10eDynamics` physics model (torque/current from a candidate motion) |
| `train_distillation_model.py` | `DistillModel` interface + `LinearModel` baseline — the gap model (Silver) |
| `metrics.py` | `EvaluationMetric` interface + `CurrentGapMetric` baseline — what the RL agent optimizes |
| `preprocess.py` | `Preprocess` interface, currently a no-op — feature scaling hook |
| `train_rla.py` | `GapEnv`/`PathEnv` (Gym) + PPO training — the RL agent (Gold) |
| `run.py` | apply trained agent to a held-out script, write optimized script/path |
| `models/`, `scripts/`, `results/` | trained artifacts, URScript motions, run outputs |

## Current status (tiers)

- **Bronze**: not yet done in this pipeline run — `analysis.py`/
  `plot_target_actual.py` exist and work on `data/test-*.csv`, but no
  writeup/plots characterizing which joints/speeds vibrate most yet.
- **Silver**: `LinearModel` predicts `actual_current`, `actual_TCP_speed`,
  and (re-added 2026-08-15, see Change Log) `actual_q` — not `actual_qd` or
  `actual_TCP_pose` (dropped 2026-08-15 once `target_selection_analysis.py`
  showed why, see `DISTILL_MODEL.md`). Three independent least-squares fits
  sharing no coefficients: `actual_current` and `actual_q` share the
  original per-joint feature set (`target_current, qd, qdd, pos, vel, acc`,
  joint one-hot) but fit different targets; `actual_TCP_speed` uses a
  per-axis feature set (`target_TCP_pose, target_TCP_speed,
  target_TCP_accel, vel, acc`, axis one-hot over x/y/z/rx/ry/rz —
  `target_TCP_pose` is still an *input* feature, just not a fitted output).
  Run-level held-out split still applies (Known issue 3). Current baseline
  (`results/2026-08-15_22-28-17/log.json`): held-out `actual_current` RMSE
  0.680A/R² 0.977 (unchanged by any retargeting); `actual_TCP_speed` RMSE
  0.0193/R² 0.9992 (pooled across m/s and rad/s); `actual_q` RMSE 0.0004
  rad/R² 1.0000 — **read that R² as confirmation of the identity-map
  problem, not a good fit; see `DISTILL_MODEL.md`'s dedicated section before
  citing it anywhere**. wrist2/wrist3 current R² is still weak (0.43/0.07,
  unchanged). **`CurrentGapMetric`, `train_rla.py`, and `run.py` still only
  read `actual_current`** — they ignore `actual_TCP_speed` and `actual_q`
  entirely for now; a vibration-aware `EvaluationMetric` (Known issue 1) is
  still needed before the RL side can use either, but at least
  `actual_current` scoring is meaningful again (it wasn't while the model
  predicted q/qd instead, see below).

  **Why not `actual_q`/`actual_qd`** (`target_selection_analysis.py`,
  `bronze_tier/target_selection/`): settling-window (`common.segments`'s
  `i1:i2`) tracking error as a percent of each channel's own full-scale
  range, pooled over all 7 real recordings — `q` 0.004% RMS / 0.233% peak,
  `qd` 0.122% RMS / 3.408% peak, TCP translation 0.024%/0.650%, TCP rotation
  0.011%/0.345%, vs. `current` 1.045%/10.268% and TCP linear speed
  0.441%/8.206%. Position-type channels (q, TCP pose) are uniformly under 1%
  peak; current and TCP *speed* carry the real, learnable signal.
  `q_qd_identity_weights.png` shows why a linear model can't help there
  either: fit anyway, it puts 99.9% of its |weight| on the `pos` feature
  (for `actual_q`) or 99.4% on `qd` (for `actual_qd`) — i.e. it learns
  `actual ≈ target`, the identity map, because there's essentially no gap
  signal to learn.
- **Gold**: baseline pipeline runs end to end (PPO trains, `run.py` produces
  an optimized script), but scores against `CurrentGapMetric`, not the PDF's
  peak-overshoot/RMS-position spec.
- **Diamond**: not started. No real-hardware validation run yet (`send.py`
  runs so far were against URSim, which has no gap by construction).

## Known issues / prerequisites (see REVIEW.md §6 for full detail)

1. **Metric mismatch** — `CurrentGapMetric` (`metrics.py`) measures current
   error, not the PDF's peak-overshoot / RMS-position-error metrics. Implement
   those first; everything downstream (Gold, Diamond) is scored against
   whatever metric is in place.
2. **Distill model doesn't predict position** — investigated 2026-08-15 on
   `apostolosDistillModel`; conclusion changed three times the same day (see
   Change Log). First tried predicting `actual_q`/`actual_qd`
   (`LinearModel.predicts() == ["actual_q", "actual_qd"]`), which broke
   `CurrentGapMetric`/`run.py`/`train_rla.py` (they only read
   `actual_current`, which that version no longer predicted). Then
   `target_selection_analysis.py` showed *why* q/qd wasn't a good choice
   anyway — the tracking gap there is tiny (<1% peak of full-scale, even
   inside the settling window) and a linear model fit to it just learns the
   identity map. Next tried adding `actual_TCP_pose` alongside
   `actual_current`/`actual_TCP_speed`, then dropped `actual_TCP_pose` too
   once the same check showed its gap is equally negligible (translation
   and rotation both <1% peak). Landed on **`actual_current` +
   `actual_TCP_speed`**: current is the original, real signal (and what the
   RL side already reads); TCP speed is the one TCP-space channel with a
   real gap (linear: 8.2% peak, comparable to current's 10.3%) —
   `target_TCP_pose` is kept only as an *input* feature for that fit, not
   an output. **Still open**: `CurrentGapMetric`/`run.py`/`train_rla.py`
   don't read `actual_TCP_speed` at all — a vibration-aware
   `EvaluationMetric` (issue 1) is needed to actually use it for anything
   beyond Bronze-tier characterization.
3. ~~**Row-level held-out split**~~ — resolved 2026-08-15 (for real this
   time — verified by running the script): `train_distillation_model.py`
   now defaults to a fixed file-level split, no flag needed —
   `DEFAULT_TRAIN_CSVS`/`DEFAULT_TEST_CSVS` = `test-1,2,3,6.csv` /
   `test-4,5,7.csv`, applied the same way regardless of `--model` or the
   active `Preprocess`. `--train-csvs`/`--test-csvs` override it (the two
   must not overlap — the script exits if they do). The pairing isn't
   arbitrary: `{2,4}` are both acc sweeps at vel=100, `{3,5}` both vel
   sweeps at acc=100, `{6,7}` both wide random vel/acc combos (`test-1` is a
   standalone low-range grid) — holding out one file per pair keeps the test
   set in the same regime as training, so the reported number is a
   generalization check, not an extrapolation test. `log.json` now reports
   `held_out_metrics` (on the test files) and `in_sample_metrics` (on the
   train files, sanity-check only) separately, with per-joint RMSE/R² for
   each. See REVIEW.md §9.2.
4. **URSim validation is meaningless** — `send.py --robot-ip 127.0.0.1`
   always shows zero gap. Before/after comparisons need real hardware or the
   distill model's injected predictions.

## Other branches with related work

`origin/DistillModel` has commits not on this branch (`apostolos`), including
a CNN-based `DistillModel` refactor (`d4e0d97 add CNNModel + large refactor
for DL based DistillModels`). Worth checking before starting Silver work from
scratch, to avoid duplicating a teammate's model.

## Change Log

- **2026-08-16**: Added `BRONZE_ANALYSIS.md` (new file, case 2 root) --
  reference doc for the Bronze-tier analysis: the exact math behind every
  gap metric (`bronze_exploration.py`'s per-move RMS-position-error/
  peak-overshoot, `channel_gap_bar_chart.py`'s peak/RMS-as-%-of-full-scale
  and the quaternion/geodesic special case for TCP orientation), what each
  plot folder under `bronze_tier/` shows, and the headline numbers pulled
  from `log.json`/`channel_gap_summary.json`. Documentation only, no code
  changed. Also expanded `REVIEW.md` §4 with three new subsections written
  up from a Q&A walkthrough of the RL/deployment side that wasn't
  previously documented: **"There is no real-time inference anywhere in
  this pipeline"** (synthesizes the already-documented one-step-bandit fact
  with the already-documented `run.py`-bakes-a-static-file fact into one
  argument -- PPO's justification here is amortizing inference across a
  script's moves, not sequential credit assignment, since nothing in this
  pipeline calls `agent.predict()` anywhere near the robot at motion time);
  **"Why `run.py` averages `vel`/`acc` across moves"** (it's not a modeling
  choice -- `movej` scripts have exactly one shared `vel`/`acc` variable
  pair, `utils.get_param`/`set_param` read/write a single regex match, so
  N per-move agent opinions have to collapse into one number; path mode
  doesn't have this problem since it writes a per-move plan into its own
  CSV rows); **"`movej` vs `servoj`: who decides the trajectory"** (params
  mode hands the controller an endpoint + speed limits and its onboard
  planner decides every intermediate sample in real time --
  `dynamics.trapezoidal()` is this repo's own offline *model* of that
  planner, not the real thing, and `duration_range_by_file` already shows
  them disagreeing; path mode computes every setpoint in Python ahead of
  time and reduces the controller to a `servoj` tracking loop with no
  planning role left). None of this changes any code or model behavior --
  documentation catch-up only.
- **2026-08-16**: Extended `channel_gap_bar_chart.py` (same day, later) into 3
  views under `bronze_tier/channel_gap/` instead of one flat chart --
  `full_run/`, `settling_window/`, `worst_channels_by_component/` -- so they're
  easier to find/compare than the old single-file output (which is now
  deleted by the script on each run, replaced by these). Each bar now shows
  **peak and RMS** side by side, not peak alone -- RMS distinguishes "this
  channel rings once, badly" (peak high, RMS low) from "this channel is off
  all the time" (both high), which the original peak-only chart couldn't.
  `settling_window/` is the same chart restricted to rows inside
  `common.segments`' `i1:i2` (post-move ringing only, motion itself
  excluded) -- **reuses `full_run`'s full-scale-range denominator** rather
  than recomputing one from the (much narrower, near-destination) settle-only
  data, so the two views' percentages stay comparable; using a settle-window
  range would have silently inflated the settle-window percentages for no
  real reason. `worst_channels_by_component/` drills into the 3 channels
  with the worst full-run peak %FS (currently TCP speed linear, TCP speed
  angular, `qd`) with one chart per channel, peak+RMS by joint/axis instead
  of pooled -- e.g. `qd`'s gap turned out concentrated in shoulder/elbow
  (10-11% peak) with wrist2/wrist3 negligible (<1%), which the channel-level
  number alone couldn't say. (TCP orientation is deliberately excluded as a
  breakdown candidate -- its geodesic gap isn't a per-axis quantity, see next
  paragraph.) **All bar charts now use a log y-axis**: peak/RMS %FS spans
  ~4 orders of magnitude across channels (22% down to 0.004%), which on a
  linear axis crushed the small channels' peak/RMS labels into each other
  (illegible) and made them invisible -- same log-scale choice
  `target_selection_analysis.py` made for this same kind of cross-channel %FS
  comparison, see the entry below. Verified by rendering and reading every
  output PNG.
- **2026-08-16**: Added `channel_gap_bar_chart.py` -- one bar chart, one bar per
  channel (`q`, `qd`, `current`, TCP position, TCP orientation, TCP speed split
  linear/angular -- 7 bars, TCP pose/speed split so the angle-valued halves can
  be shown in degrees without forcing the linear halves into the same unit),
  showing the single largest \|actual - target\| seen anywhere in
  `data/test-*.csv` (whole run, not just the settle window). Bar height is that
  max as a percentage of the channel's own full-scale range (pooled
  target+actual, whole dataset) -- units/scales differ too much across
  channels (A, mm, deg, m/s) to plot the raw numbers on one linear axis, so %
  is the shared axis and the actual value (native unit) + that percentage are
  printed above each bar instead. **TCP orientation needed its own code
  path** (`orientation_gap_stats`, geodesic/quaternion angle, not per-axis
  subtraction): `TCP_pose3..5` is a rotation *vector*, a 2-to-1
  representation (angle-pi about axis n == angle-pi about -n), so it can flip
  sign between adjacent samples with no real motion -- naive subtraction
  produced a bogus ~360 degree spike (`test-6` row 138273, real gap 0.085
  degrees by geodesic distance) that would otherwise have dominated the whole
  chart. Its % denominator is the fixed 180 degrees max-possible-rotation
  bound, not a pooled data range (that bound doesn't vary by recording, so no
  full-scale computation is needed or meaningful there). Verified numbers
  (`bronze_tier/channel_gap_summary.json`): worst channel is TCP linear speed
  (0.66 m/s, 22.4% of range, `test-6`), then TCP angular speed (57.6 deg/s,
  12.6%), `qd` (35.0 deg/s, 10.9%), current (6.72 A, 9.2%) -- all in the same
  order as Known issue 2's settling-window numbers; TCP position (20.5 mm,
  0.78%), TCP orientation (1.21 deg, 0.67%), and `q` (1.28 deg, 0.28%) are an
  order of magnitude smaller, consistent with the identity-map finding there.
  Read-only analysis script -- doesn't touch the model/RL pipeline.
- **2026-08-16**: Added `pca_feature_analysis.py` — PCA (plain numpy SVD, no
  sklearn dependency — it's listed in `requirements.txt` but wasn't actually
  installed in the environment this ran in) over the candidate feature set
  for each of the five `actual_*` outputs (current, q, qd, TCP pose, TCP
  speed), to help decide what a `DistillModel` should actually take as
  input. current/q/qd share one joint-space candidate matrix
  (`target_current, qd, qdd, pos, vel, acc` + joint one-hot, same as
  `LinearModel._row_features`, plus two physics features from `utils.UR10e`
  — gravity torque, diagonal mass-matrix term — that the docstring calls out
  as cheap candidates but `LinearModel` doesn't use); TCP pose/TCP speed
  share one TCP-space matrix (`target_TCP_pose, target_TCP_speed,
  target_TCP_accel, vel, acc` + axis one-hot, same as
  `LinearModel._tcp_row_features`). Rows subsampled to 3000/file, evenly
  spaced (`UR10e.gravity`+`mass_matrix` cost ~1.6ms/row combined, timed;
  full-resolution over all 7 files is >1e6 rows, ~30 min just for physics).
  Per output: scree plot (variance explained + cumulative, components needed
  for 95%), loadings heatmap (which raw features load onto which
  components), a feature-importance ranking (`|loading| · |corr(PC, actual
  value)|`, summed over components — folds the *target* back in, since
  current/q/qd share the same X but should rank features differently), and a
  PC1-PC2 scatter colored by the actual value. Outputs ->
  `data_analysis/{current,q,qd,tcp_pose,tcp_speed}/` +
  `data_analysis/summary.json`+`summary_n_components.png`. Ran at the
  default 3000 rows/file: **cross-checks the existing
  `target_selection_analysis.py` finding independently** — q's top-ranked
  feature is `pos` and qd's is `qd` (both literally the channel's own
  commanded target — the identity-map signature, Known issue 2), while
  current's top feature is `target_current` and TCP pose/speed's are their
  own `target_TCP_pose`/`target_TCP_speed`. All five need 9 of their
  11-14 candidate features to reach 95% variance — the candidate set isn't
  very redundant, so cutting features for dimensionality's sake isn't
  well-motivated; the gravity/mass-matrix physics features place 2nd/3rd for
  `current` and `q`, ahead of most one-hot/kinematic terms, suggesting
  they'd be worth trying in `LinearModel` (not done here — this script is
  read-only analysis, doesn't touch `train_distillation_model.py`).
- **2026-08-15**: `LinearModel` re-added `actual_q` as a predicted output
  (fourth retargeting today) — `predicts()` now returns `["actual_current",
  "actual_TCP_speed", "actual_q"]`. Reuses `_row_features`/`_design_current`'s
  exact feature construction (new shared `_design_row(recordings,
  actual_attr)` helper, called with `"actual_current"` or `"actual_q"` to
  avoid duplicating the design-matrix loop), fit as its own coefficient
  vector `coef_q`, no sharing with the current fit. Done not because the
  evidence against predicting position changed (it hasn't — see
  `DISTILL_MODEL.md`, unchanged), but because Gold's PDF-literal criterion
  ("outperforms... on the vibration metrics") needs *some* predicted
  `actual_q` to score RL candidates against once a position-based
  `EvaluationMetric` is wired in (Known issue 1, still open — this change
  doesn't wire it in, just unblocks the model side). Verified:
  `results/2026-08-15_22-28-17/log.json` — held-out RMSE 0.0004 rad, R²
  1.0000, every joint, flat across joints in a way `actual_current`'s
  per-joint spread (R² 0.07-0.98) is not — the flatness itself is a signature
  of the identity-map fit, documented in detail in `DISTILL_MODEL.md`'s new
  "Why `actual_q` is back in" section. **This number should not be presented
  without that caveat.** `actual_current`/`actual_TCP_speed` numbers
  unchanged, confirming the new fit doesn't perturb the others.
- **2026-08-15**: `LinearModel` dropped `actual_TCP_pose` as a predicted
  output (third retargeting today) — now predicts just `actual_current` +
  `actual_TCP_speed`. `target_selection_analysis.py`'s numbers already
  showed TCP position/orientation gap is as negligible as joint q (<1%
  peak %FS), so fitting it was the same identity-map problem as q/qd, just
  not yet called out. `target_TCP_pose` is still read as an *input* feature
  for the TCP-speed fit (`_design_tcp`/`_tcp_row_features`), just not
  fitted as an output anymore. `_evaluate_model`/plotting code needed no
  changes (loops over `model.predicts()` generically); `COMPONENT_NAMES`
  dropped its `actual_TCP_pose` entry. Verified: `results/2026-08-15_16-12-17/
  log.json` — `actual_current`/`actual_TCP_speed` numbers unchanged from the
  3-channel run (0.680A/R² 0.977 and 0.0193/R² 0.9992 respectively), as
  expected since dropping a separate fit doesn't touch the other two.
  `target_selection_analysis.py` also gained **`plot_gap_summary`** — a
  single headline bar chart (`bronze_tier/target_selection/gap_summary.png`)
  putting all 7 candidate channels' settling-window RMS/peak %FS on one
  log-scale axis, color-coded by whether `LinearModel` actually predicts
  that channel (green) or not (grey) — meant to be the one slide-ready plot,
  vs. `gap_by_channel.png`'s native-unit small multiples (kept, as backup/
  detail). One honest wrinkle visible in it: `qd`'s peak (3.4%) is not
  obviously below the selected channels' — the RMS numbers are the cleaner
  separator (current/TCP-speed all ≥0.1%, q/qd/TCP-pose all <0.13% except
  qd sitting right at the boundary); worth a caveat if this plot goes in
  front of the company, not just a clean "always separated" story.
- **2026-08-15**: `LinearModel` (`train_distillation_model.py`) retargeted a
  second time, this time to `actual_current` + `actual_TCP_pose` +
  `actual_TCP_speed` (previous entry below retargeted it to `actual_q`/
  `actual_qd`, same day — that turned out not to be a good target, see Known
  issue 2). Three independent least-squares fits: `_design_current`/
  `_row_features` (unchanged from the original baseline, per-joint) for
  `actual_current`; new `_design_tcp`/`_tcp_row_features` (per-axis:
  x/y/z/rx/ry/rz, feature set `target_TCP_pose, target_TCP_speed,
  target_TCP_accel, vel, acc` + axis one-hot) for the two TCP channels.
  `analysis.Recording` gained `target_TCP_pose`/`actual_TCP_pose`/
  `target_TCP_speed`/`actual_TCP_speed` arrays (columns already existed in
  the CSVs). `_compute_metrics`/`_plot_per_joint_rmse`/`_plot_per_joint_r2`
  now label components via a new `COMPONENT_NAMES` dict (joint names for
  `actual_current`, Cartesian/orientation axis names for the TCP channels)
  instead of hardcoding `JOINT_NAMES` — no other change needed since they
  already looped generically over `model.predicts()`. Ran
  `train_distillation_model.py --out models/distill.pkl`:
  `actual_current` held-out RMSE 0.680A/R² 0.977 (identical to the original
  baseline, as expected — same features, same fit), `actual_TCP_pose` RMSE
  0.0121/R² 0.9999, `actual_TCP_speed` RMSE 0.0193/R² 0.9992 (pooled; see
  `results/2026-08-15_15-58-21/log.json` for per-axis numbers). Added
  **`target_selection_analysis.py`** (new file) — settling-window gap
  comparison across all 5 candidate channels (current, q, qd, TCP
  pose split translation/rotation, TCP speed split linear/angular) as %
  of each channel's own full-scale range, plus a plot of what a linear
  model actually learns if fit to `actual_q`/`actual_qd` anyway (it puts
  ~100% of its weight on the identity feature). Outputs in
  `bronze_tier/target_selection/` (`gap_by_channel.png`,
  `q_qd_identity_weights.png`, `log.json`) — this is the evidence for why
  q/qd were dropped in favor of current/TCP-speed. See Known issue 2 for
  the numbers.
- **2026-08-15**: `LinearModel` (`train_distillation_model.py`) retargeted
  from `actual_current` to `actual_q`/`actual_qd` — `predicts()` now returns
  `["actual_q", "actual_qd"]`, fit as two separate least-squares solves
  (`coef_q`, `coef_qd`) against the *same* feature vector the old
  current-predicting baseline used, so results stay comparable
  apples-to-apples. `analysis.Recording` gained an `actual_qd` array
  (column already existed in the CSVs, just wasn't loaded). No changes to
  `_evaluate_model`/plotting/`runs_summary.csv`/`augment()` — all already
  looped generically over `model.predicts()`. Ran
  `train_distillation_model.py --out models/distill.pkl` on the default
  file-level split: held-out RMSE 0.0004 rad / R² 1.0000 on `actual_q`,
  RMSE 0.0108 rad/s / R² 0.9995 on `actual_qd` (see Silver status above for
  the caveat on what that R² does and doesn't show). **Not done**:
  `metrics.py`/`train_rla.py`/`run.py` still assume `actual_current` — see
  Known issue 2's new consequence. `models/distill.pkl` and
  `results/2026-08-15_15-35-35/` now reflect this model; earlier
  `actual_current`-predicting versions are only in older `results/<dt>/`
  folders.
- **2026-08-15**: `train_distillation_model.py` now defaults to a fixed
  file-level train/test split — `model.fit()` only sees
  `data/test-1,2,3,6.csv`, every printed/plotted/logged metric comes from
  predicting `data/test-4,5,7.csv` (never seen by `fit()`), regardless of
  `--model` or the active `Preprocess`. Replaces the old row-level
  `--holdout` split (adjacent-row leakage made that number optimistic).
  `--train-csvs`/`--test-csvs` override the defaults; a `MODELS` registry
  (`{"linear": LinearModel}`) backs a new `--model` flag so future
  `DistillModel` subclasses get the same split for free. `log_run()` now logs
  `held_out_metrics` (test files) and `in_sample_metrics` (train files, sanity
  check only) instead of the old `full_data_metrics`/`holdout_fraction`. The
  saved model is fit on the training files only — it is not refit on
  everything afterward, so the reported numbers describe the exact deployed
  model. Verified by running it: held-out RMSE 0.680A/R² 0.977 (pooled),
  in-sample RMSE 0.670A/R² 0.973 — see updated Silver status above.
  `README.md`/`REVIEW.md` §9.2 updated to match. **Note**: while auditing this
  area, found the entry below (also dated 2026-08-15) describing `LinearModel`
  switched to predict `actual_q`/`actual_qd` does not match the code on this
  branch — `predicts()` still returns `["actual_current"]`. Left the entry
  below as-is (historical record) but corrected Known issue 2 above; don't
  trust that entry's claims without re-checking the code.
- **2026-08-15**: `LinearModel` (`train_distillation_model.py`) now predicts
  `actual_q`/`actual_qd` instead of `actual_current` — see Known issue 2.
  `analysis.Recording` gained an `actual_qd` array to support fitting it.
  `bronze_exploration.py` also gained a `presentation_plots_baseline/` output
  (4-5 curated plots in degrees/%/seconds instead of mrad, rerunnable against
  optimized-trajectory recordings via `--presentation-name`) and a
  `duration_s` column in `segment_stats.csv`. Flagged, not yet fixed:
  `CurrentGapMetric` needs replacing with a position-based metric before the
  RL agent (`train_rla.py`/`run.py`) is retrained against this model.
  — Later same day: added `--test-csvs` to `train_distillation_model.py` for
  a genuine run-level held-out split (Known issue 3); `models/distill.pkl`
  is now trained on `test-2,3,6,7` and evaluated on `test-1,4,5`
  (`results/2026-08-15_14-37-58/`). R² is still ≈1.0 either way — that's a
  separate, still-open problem (RMSE-on-`actual_q` barely reacts to the
  ring), not a leakage artifact.
- **2026-08-13**: Created this file. No code changes yet this session —
  reviewed the case PDF (`../2026-robotelite-sdu-dk-6_ur.pdf`) against
  `README.md`/`REVIEW.md` to confirm current status above.
