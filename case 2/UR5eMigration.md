# UR10e → UR5e Migration

What changed when this project switched from a UR10e to a UR5e (the user's
real, physically-accessible robot), the exact before/after physical
parameters, and — the main point of this file — **what to update if better
UR5e specs turn up**. Unlike `ModelReview.md` (which tracks
`train_distillation_model.py`'s feature/model/results specifically), this is
a one-time migration record, not a living document tied to a particular
file's changes; it doesn't need a changelog of its own.

## 1. What was done (2026-08-19)

| Area | Before | After |
|---|---|---|
| Simulator (`simulation environment/docker-compose.yml`) | `ROBOT_TYPE=UR10` | `ROBOT_TYPE=UR5` |
| `utils.py` | `class UR10e` | `class UR5e` (same interface, new constants — §2) |
| `dynamics.py` | `class UR10eDynamics`, `default_dynamics()` returned it | `class UR5eDynamics`, `default_dynamics()` returns it |
| `train_distillation_model.py` | `_UR10E = UR10e()` singleton for `gravity_torque` | `_UR5E = UR5e()`; also `DEFAULT_CSVS` changed from a hardcoded `range(1,8)` file list to a glob over `data/test-*.csv` |
| `data/test-*.csv` | Real UR10e recordings | **Unchanged as of this file** — still UR10e recordings; not yet swapped for real UR5 data (see §4) |
| `models/*.pkl`, `models/agent_params.zip` | Trained on UR10e data/physics | **Unchanged as of this file** — stale until retrained on real UR5 data |

Docs updated for consistency: `README.md`, `REVIEW.md`, `ModelReview.md`
(body + a dated Changelog entry, per its own living-document convention),
`pipeline.html`, `simulation environment/README.md`.

**Not changed, and didn't need to be** (confirmed generic/robot-agnostic):
`record.py`, `send.py`, `run.py`, `train_rla.py`, `analysis.py`, `common.py`,
`preprocess.py`, `metrics.py`, `ur_style.py`, RTDE column names, `--float-register`
handling, `MAX_JOINT_SPEED`/`MAX_JOINT_ACC` (generic firmware ceilings, not
per-model — see §3 if you want to tighten these), `fit_kt` (fit empirically
per-recording, adapts automatically to whichever robot's data it's given).

## 2. Physical parameters: before (UR10e) vs after (UR5e)

Both sets are Standard/Classic DH, from the same source: Universal Robots'
["DH Parameters for calculations of kinematics and
dynamics"](https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/)
(fetched live 2026-08-19 for the UR5e numbers, not from memory — cross-checked
by re-fetching the UR10e row from the same page and confirming it matched
what was already in the code, exact digit for digit, before trusting the
UR5e row from the same fetch).

### DH parameters

| joint | UR10e `a` (m) | UR5e `a` (m) | UR10e `d` (m) | UR5e `d` (m) | `alpha` (rad, same both) |
|---|---|---|---|---|---|
| 1 (base) | 0.0 | 0.0 | 0.1807 | 0.1625 | π/2 |
| 2 (shoulder) | -0.6127 | -0.425 | 0.0 | 0.0 | 0 |
| 3 (elbow) | -0.57155 | -0.3922 | 0.0 | 0.0 | 0 |
| 4 (wrist1) | 0.0 | 0.0 | 0.17415 | 0.1333 | π/2 |
| 5 (wrist2) | 0.0 | 0.0 | 0.11985 | 0.0997 | -π/2 |
| 6 (wrist3) | 0.0 | 0.0 | 0.11655 | 0.0996 | 0 |

UR5e's `a`/`d` are consistently smaller than UR10e's, as expected for a
physically smaller arm — a useful plausibility check if you ever re-fetch
these and want to sanity-check the numbers before trusting them.

### Link mass, center of mass, inertia

| link | UR10e mass (kg) | UR5e mass (kg) | UR10e COM (m) | UR5e COM (m) | UR10e inertia (diag Ixx,Iyy,Izz) | UR5e inertia (diag Ixx,Iyy,Izz) |
|---|---|---|---|---|---|---|
| 1 | 7.369 | 3.761 | [0.021, 0.000, 0.027] | [0.000, -0.02561, 0.00193] | 0.0341, 0.0353, 0.0216 | 0, 0, 0 |
| 2 | 13.051 | 8.058 | [0.380, 0.000, 0.158] | [0.2125, 0.000, 0.11336] | 0.0281, 0.7707, 0.7694 | 0, 0, 0 |
| 3 | 3.989 | 2.846 | [0.240, 0.000, 0.068] | [0.150, 0.000, 0.0265] | 0.0101, 0.3093, 0.3065 | 0, 0, 0 |
| 4 | 2.1 | 1.37 | [0.000, 0.007, 0.018] | [0.000, -0.0018, 0.01634] | 0.0030, 0.0022, 0.0026 | 0, 0, 0 |
| 5 | 1.98 | 1.3 | [0.000, 0.007, 0.018] | [0.000, 0.0018, 0.01634] | 0.0030, 0.0022, 0.0026 | 0, 0, 0 |
| 6 | 0.615 | 0.365 | [0.000, 0.000, -0.026] | [0.000, 0.000, -0.001159] | 0.0000, 0.0004, 0.0003 | 0, 0, 0.0002 |
| **total** | **29.10 kg** | **17.70 kg** | | | | |

The UR10e inertia tensors also carry small off-diagonal terms (e.g. link 1's
`Ixz = -0.0043`) — full values are in `utils.py`'s git history if you need
them for comparison; omitted here for width.

**The zero UR5e inertia is not a fetch error** — verified three separate
times against the live page. Universal Robots' own published dynamics table
only gives nonzero link inertia from UR10e upward; UR5e/UR7e's rotational
inertia is small enough relative to the point-mass (`mass`/`COM`) term that
UR itself drops it. `utils.py`'s `mass_matrix()` still comes out symmetric
and positive-definite with this (verified directly, not just assumed) — the
`Jv.T @ Jv` translational term alone carries it once `I_base = 0`.

`_G = 9.80665` (gravity, m/s²) is unchanged — robot-agnostic.

## 3. What to change if you get better/official specs

Everything above came from Universal Robots' own public support page, so
it's a real published source, not a rough estimate — but here's what's worth
revisiting if you get something more precise (an official datasheet PDF
directly from UR, a CAD export, or a robot-specific calibration):

1. **Link inertia tensors (links 1-5)** — currently all-zero per UR's own
   simplification (§2). If you obtain fuller values (e.g. from a URDF/CAD
   export with more precision than UR's own published table), replace
   `utils.py`'s `_INERTIA` array. This is the single most approximate part of
   the current physics — everything else here is UR's own stated number, this
   one is UR's own stated *simplification*.
2. **Payload** — `UR5e(payload=0.0)` is the default everywhere
   (`train_distillation_model.py`'s `_UR5E`, `dynamics.py`'s
   `UR5eDynamics.__init__`). If your UR5e carries a known tool/gripper mass at
   the flange, pass the real value — it's a point mass added at the TCP in
   `utils.py`'s kinematics, so `gravity_torque` and `mass_matrix` will be
   measurably off at anything but a bare flange otherwise (the same way the
   original UR10e code's docstring already noted `payload=0.8` as an example,
   never wired up because recordings didn't log a per-run payload).
3. **Per-joint speed/acceleration limits** — `dynamics.py`'s
   `MAX_JOINT_SPEED = np.pi` / `MAX_JOINT_ACC = 4.0 * np.pi` are one flat
   ceiling for all six joints, sourced as "a generous ceiling" rather than
   from a spec sheet (confirmed not model-specific, i.e. not changed in this
   migration). UR5e's real datasheet gives tighter, per-joint limits (the
   wrist joints typically allow higher deg/s than base/shoulder/elbow) — if
   candidate-generation bounds ever matter more precisely than "a safe
   ceiling", replace this with a 6-element array from the datasheet.
4. **Coriolis term** — `UR5e.coriolis()` exists (finite-difference of
   `mass_matrix`) but `UR5eDynamics` doesn't call it (`tau = M(q)qdd + g(q)`
   only, documented as "small here, expensive to compute" — a pre-existing
   design choice, not something this migration changed). Worth revisiting if
   a future accuracy check shows it matters more for the UR5e's dynamics
   range than it did for the UR10e's.
5. **`Kt` (torque/current constant)** — already fit empirically per-recording
   (`dynamics.fit_kt`), not hardcoded, so it doesn't need touching here — it
   automatically reflects whatever real UR5 data it's given (§4). Listed here
   only so it's clear this *isn't* a manual-update item, in case you go
   looking for it.
6. **DH convention, if you fetch from a different source than the page cited
   in §2** — this code uses Standard/Classic DH (`alpha` = *previous* joint's
   twist). Some other sources (ROS `ur_description`, some academic papers)
   use Modified DH or a different axis convention — pasting those numbers in
   directly without converting would silently produce wrong kinematics that
   still "run" without erroring. Always verify against `utils.py`'s existing
   `_frames`/`_dh` row form before trusting a new source.

**To make an update**: edit `utils.py`'s `_A`/`_D`/`_ALPHA`/`_MASS`/`_COM`/`_INERTIA`
arrays (lines ~57-81), update the citation comment above them with the new
source, then re-run the structural sanity check below before trusting the
result:

```python
from utils import UR5e, _MASS, _COM, _INERTIA
import numpy as np
assert np.all(_MASS > 0)
assert not np.any(np.isnan(np.concatenate([_MASS, _COM.ravel(), _INERTIA.ravel()])))
for i in range(6):
    assert np.allclose(_INERTIA[i], _INERTIA[i].T, atol=1e-6)
ur = UR5e()
M = ur.mass_matrix([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])
assert np.allclose(M, M.T) and np.all(np.linalg.eigvalsh(M) > 0)
```

This catches shape/transcription errors and non-physical values (asymmetric
or non-positive-definite mass matrix) — it cannot confirm the numbers
themselves are correct, only structurally sane. There's no substitute for a
citable source for that part.

## 4. Still pending — not done in this migration

- **Real UR5 recordings aren't in `data/` yet.** `data/test-*.csv` still
  holds the original UR10e recordings. `train_distillation_model.py`'s
  `DEFAULT_CSVS` now globs `data/test-*.csv`, so it'll pick up UR5
  recordings automatically once they land there under that naming — but
  don't just drop new files in alongside the old ones: the old UR10e
  recordings need deleting first (`git rm`, they're git-LFS tracked), or the
  glob will silently train on a mix of both robots' data. Confirm the new
  files are in place and valid before deleting the old ones.
- **`models/distill.pkl`, `models/distill_tree.pkl`,
  `models/agent_params.zip` are stale** — trained on UR10e data through the
  old physics. Rerun `train_distillation_model.py` and `train_rla.py` once
  real UR5 data is in place; both overwrite in place, no manual cleanup
  needed first.
- **`ModelReview.md`'s results (§6/§8) still describe the UR10e-trained
  model** — its own Changelog entry for this migration says as much; update
  the body once a UR5-trained run exists to compare against.
