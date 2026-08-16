# Bronze-Tier Analysis

How the reality gap was characterized from the recorded UR10e data, what the
two analysis scripts compute, the math behind every gap metric, and what
each plot under `bronze_tier/` shows. See `CLAUDE.md`'s Change Log for the
dated narrative of how this analysis evolved; this file is the reference for
what the finished metrics/plots mean.

## Data

`data/test-1.csv` … `test-7.csv`, each with a matching `.script` — genuine
real-hardware RTDE recordings (not URSim, which has zero gap by
construction), one row per RTDE tick, `target_*` (commanded) and `actual_*`
(measured) columns per channel, plus the `vel`/`acc` register values the
running script commanded at that instant (see `.script` files — these step
or randomize `vel`/`acc` over the course of a recording, then use
`write_output_float_register` to log the current value into the CSV).

| File | Sweep |
|---|---|
| `test-1` | vel and acc stepped together, 100→10, fixed two-point move |
| `test-2` | acc stepped 100→10, vel fixed at 100 |
| `test-3` | vel stepped 100→10, acc fixed at 100 |
| `test-4`/`test-5` | same as test-2/test-3, with `sleep(0.75)` between moves |
| `test-6`/`test-7` | random vel/acc draws over a richer 5-point path |

Pooling every move across all 7 files gives a dense `(vel, acc) →
vibration` dataset with no extra recording needed. `test-1/2/3` are the
cleanest single-variable sweeps.

Two scripts produce everything under `bronze_tier/`:

- **`bronze_exploration.py`** — segments each recording into individual
  `movej` moves and computes **per-move** metrics (peak overshoot, RMS
  position error) in **joint space**.
- **`channel_gap_bar_chart.py`** — pools every row of every recording and
  computes **peak/RMS gap as a percentage of full-scale range**, per
  **channel** (q, qd, current, TCP position/orientation/speed), not per
  move.

They answer different questions: `bronze_exploration.py` asks *"how much
does a given move overshoot, and does that depend on vel/acc?"*;
`channel_gap_bar_chart.py` asks *"which of the seven measurable channels
actually carries a real actual-vs-target gap at all, and where in that
channel is it concentrated?"* — the second question is what motivated
Silver's choice of what to predict (see `CLAUDE.md` Known issue 2).

## Segmenting a recording into moves

Both scripts rely on `common.segments()`. A `movej` line moves the robot
from wherever it is to one waypoint; the controller reports which script
line is currently executing (`script_control_line`), so a segment is the
run of rows belonging to one `movej` line, bounded by the next `movej`
line (or the recording's end). Within a segment:

- **`i0`** — the move begins (the row where this `movej`'s line first
  appears).
- **`i1`** — motion stops: the last row where any joint's *commanded*
  speed (`target_qd`) is still above `0.05 rad/s`, plus one.
- **`i2`** — the next `movej` begins (or the recording ends).

So `[i0:i1]` is the **motion window** (the trapezoidal speed profile
actually running) and `[i1:i2]` is the **settle window** — the real robot
is still moving after the commanded profile has nominally finished, and
that post-move ringing is exactly what the case brief's vibration metrics
are meant to capture. `dist`/`start`/`dest` are the widest-travel joint's
angle at `i0`/`i1`, in radians; `vel`/`acc` are the commanded register
values active at `i0`.

## Per-move metrics (`bronze_exploration.py`, `segment_stats.csv`)

Computed once per segment, on the segment's own dominant joint.

### RMS position error

Root-mean-square of actual-minus-target joint angle, over the **whole
move** (`i0:i2`, motion + settle):

```
rms_pos_err = sqrt( mean( (actual_q[i0:i2] - target_q[i0:i2])^2 ) )   [rad, reported in mrad]
```

Same formula `analysis.py`'s per-joint stats use. This captures *tracking*
error throughout the move, not just the end.

### Peak overshoot

Max absolute deviation from the **final destination** angle, restricted to
the **settle window only** (`i1:i2`):

```
peak_overshoot = max( |actual_q[i1:i2] - dest| )   [rad, reported in mrad]
```

`dest` is fixed (the commanded target's value at `i1`), so this measures
how far the real robot swings past — or oscillates around — the
destination *after* the commanded profile says it should already be there.
This is the number that answers "how much does it ring."

Both are also reported as **degrees** and as **% of that move's own
commanded travel distance** (`dist_rad`) in the presentation plots —
`error / dist_rad * 100` — so a 2° error on a 3-radian shoulder swing and a
2° error on a 0.1-radian wrist twitch aren't compared as if they were the
same thing.

## Channel gap metrics (`channel_gap_bar_chart.py`)

Where `bronze_exploration.py` looks at one channel (`q`) move-by-move,
this script looks at **every** channel — `q`, `qd`, `current`, TCP
position, TCP orientation, TCP speed (linear/angular) — pooled across
**every row of every recording**, not just settle windows (a separate
`settling_window/` view restricts to settle-window rows only, see below).

### Peak and RMS gap (per channel, per component)

For a channel with target/actual columns (e.g. `target_q0..5` /
`actual_q0..5`), pool `|actual - target|` over every row, every file, every
component (joint or axis):

```
gap[i]   = |actual[i] - target[i]|
peak_raw = max(gap)
rms_raw  = sqrt( sum(gap^2) / n )
```

### Full-scale range (the denominator)

Each channel's own **full-scale range** — the spread of values that
channel actually took on, target and actual pooled together, over the
**whole run**:

```
fs_range = max(target ∪ actual) - min(target ∪ actual)
```

Percentages always divide by this same whole-run number, even when the
gap itself is being measured only inside settle windows (`settling_window/`
view) — reusing the full-run range keeps the two views comparable; a
settle-window-only range would be much narrower (the robot is near its
destination there) and would silently inflate the settle-window
percentages for no real reason.

```
pct_peak = peak_raw / fs_range * 100
pct_rms  = rms_raw  / fs_range * 100
```

This percentage — not the raw A/mm/deg/s number — is the one axis every
channel shares despite living in completely different units, which is why
every bar chart in `channel_gap/` plots %FS, with the raw value printed
above each bar for reference.

### TCP orientation — a special case

`TCP_pose3..5` is a rotation **vector** (axis × angle), a 2-to-1
representation: rotating by angle π about axis *n* is the same physical
orientation as angle π about axis −*n*, so the logged vector can flip sign
between adjacent samples with **no real motion**. Naive per-axis
subtraction picks up spurious ~360° spikes at those flips (observed
directly: `test-6` row 138273, true gap there is 0.085°, not ~360°) — so
orientation gets its own math, the **geodesic (quaternion) angle** between
target and actual orientation:

```
q = axis_angle_to_quaternion(v)              # v = rotation vector, radians
gap[i] = 2 * arccos( |dot(q_target[i], q_actual[i])| )   # radians, in [0, π]
```

peak/RMS computed the same way (max / root-mean-square) over this angle
series. The denominator is a **fixed physical bound**, not a data range —
two orientations can never differ by more than π radians (180°) — so
`pct = gap / π * 100`, and it needs no full-run-vs-settle-window override
since that bound never depends on the data.

### Component breakdown

The channel-level number says *which channel* has the worst gap; the
`worst_channels_by_component/` view answers *where in that channel*. Same
formula, just run once per single joint/axis instead of pooled across all
of them, still divided by the channel's whole-run `fs_range` so components
stay comparable to the channel-level bar they drill into.

## What the plots show

### `bronze_tier/*.png` — top level (`bronze_exploration.py`)

- **`overshoot_vs_vel.png` / `rms_vs_vel.png`** — peak overshoot / RMS
  error vs. commanded `vel`, from the vel-sweep files (acc held fixed),
  one series per (joint, travel direction) — split by direction because a
  joint moving against gravity vs. with gravity assist can ring
  differently at the same vel/acc, which pooling both into one series
  would hide as noise.
- **`overshoot_vs_acc.png` / `rms_vs_acc.png`** — same, against `acc`
  (vel held fixed).
- **`overshoot_heatmap.png` / `rms_heatmap.png`** — mean metric over a
  binned `(vel, acc)` grid, from the randomized combo runs (`test-1`,
  `test-6`, `test-7`), the one place both parameters vary together.
- **`per_joint_overshoot.png`** — bar chart, mean peak overshoot per
  joint, pooled over every file — which joints ring the most in absolute
  terms.
- **`overshoot_by_direction.png` / `rms_by_direction.png`** — box plots
  split by (joint, travel direction), isolating the direction/gravity
  effect from the vel/acc effect the scatter plots show.

### `bronze_tier/trajectories/` — target vs. actual, one move at a time

- **`test-N_<joint>_<param>_sweep.png`** — target (dashed) vs. actual
  (solid) position curves for several settings of a sweep file's varying
  parameter, overlaid and colored by that parameter's value — makes the
  speed/acceleration trend visible directly in the trajectory shape, not
  just in a summary statistic.
- **`worst_move_*.png` / `best_move_*.png`** — the single move with the
  largest / smallest peak overshoot anywhere in the dataset, in detail:
  full trajectory for context plus a zoomed panel of the settle window in
  mrad-from-destination (the overshoot is invisible against a
  multi-radian move on a shared axis, hence the separate zoomed view).

### `bronze_tier/per_run/` and `bronze_tier/presentation_plots_baseline/`

Both use the curated bar-chart form (`plot_metric_by_joint`): one bar per
joint, labeled with **both** degrees and **% of that move's own commanded
travel distance**, rather than raw mrad. `per_run/` runs it once per
recording (`test-N_overshoot_by_joint.png` / `test-N_rms_by_joint.png`),
un-pooled so a joint that barely moves in one file isn't squashed by
another file's larger bars. `presentation_plots_baseline/` is the pooled,
slide-ready set (4-5 plots): `overshoot_by_joint.png`, `rms_by_joint.png`,
`duration_by_joint.png` (mean motion-only duration per joint — the "did it
actually get faster" companion metric, since vibration is trivial to
reduce by simply slowing down), and labeled `(vel, acc)` heatmaps with the
value printed in each cell. Meant to be regenerated unchanged against
optimized-trajectory recordings later (`--presentation-name`) for a direct
baseline-vs-optimized comparison.

### `bronze_tier/channel_gap/` (`channel_gap_bar_chart.py`)

- **`full_run/channel_gap_bar_chart.png`** — one grouped peak+RMS bar pair
  per channel (7 channels: q, qd, current, TCP position, TCP orientation,
  TCP speed linear, TCP speed angular), height = % of that channel's own
  full-scale range, log-scaled y-axis (%FS spans ~4 orders of magnitude
  across channels — linear would crush the small channels' bars/labels
  into illegibility). Raw value (native unit) + percentage printed above
  each bar. This is the headline "which channel actually has a gap" chart.
- **`settling_window/channel_gap_bar_chart.png`** — identical chart, rows
  restricted to settle-window-only (`i1:i2`), reusing `full_run`'s %FS
  denominator so the two are directly comparable.
- **`worst_channels_by_component/*.png`** — one chart per each of the 3
  channels with the worst full-run peak %FS, broken down by joint/axis
  instead of pooled (TCP orientation is excluded here — its geodesic gap
  isn't a per-axis quantity, see above).

### Headline numbers (`bronze_tier/log.json`, `channel_gap/full_run/channel_gap_summary.json`)

From `log.json` (2497 segments, 7 files): worst peak overshoot is the
**shoulder** joint, `test-4`/`test-5`, ~5.8-5.9 mrad; worst RMS error is
also the shoulder, `test-3`, ~2.3 mrad. `corr_vel_overshoot` = -0.27,
`corr_acc_overshoot` = -0.33 (weak negative — higher vel/acc *reduces*
mean overshoot in this data, counter to the naive "faster = more
vibration" assumption, computed within each (joint, direction) group and
averaged to avoid mixing speed regimes). `duration_range_by_file` shows
move duration barely changes across the swept vel/acc register (ratio
~1.0-1.35 over most files) — evidence the controller isn't simply running
a trapezoidal profile capped at the commanded vel/acc the way `dynamics.py`
assumes; something else (a lower default speed limit, a blend/safety cap)
is binding instead.

From `channel_gap_summary.json` (whole-run, all 7 files): the ranked worst
channels by peak %FS are **TCP linear speed** (0.66 m/s, 22.4%, `test-6`),
**TCP angular speed** (57.6°/s, 12.6%), **qd** (35.0°/s, 10.9%), **current**
(6.72 A, 9.2%) — all an order of magnitude above **TCP position** (20.5 mm,
0.78%), **TCP orientation** (1.21°, 0.67%), and **q** (1.28°, 0.28%). This
is the evidence behind Silver's choice of `actual_current` +
`actual_TCP_speed` as the model's predicted outputs: q/qd/TCP-pose have a
gap so small a fitted model just learns the identity map, while current
and TCP speed carry a real, learnable signal.

## Reproducing

```bash
python bronze_exploration.py          # -> bronze_tier/*.png, segment_stats.csv, log.json
python channel_gap_bar_chart.py       # -> bronze_tier/channel_gap/{full_run,settling_window,worst_channels_by_component}/
```

Both default to `data/test-*.csv`; pass `--data-glob`/`--out` to point at a
different recording set (e.g. optimized-trajectory recordings for a
before/after comparison).
