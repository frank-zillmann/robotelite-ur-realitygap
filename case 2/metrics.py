"""Score a recording's rows into one per-row value the optimizer minimizes.

``EvaluationMetric`` is the interface:

    needs()      -> per-joint channel bases this metric reads
    per_row(df)  -> (n,) score, one value per row

``CurrentGapMetric`` is the original current-tracking gap. ``PositionGapMetric``
is the position-tracking gap the case brief actually specifies (PDF "Challenge":
``peak = max(|actual(t) - target(t)|)``, ``rms = sqrt(mean((actual(t) -
target(t))^2))``, both over the settling window of a move). A subclass can read
other channels (jerk, a mix); each must be a channel the recording carries or the
distill model predicts.

Neither peak nor rms is computed in this file: ``per_row`` returns one
tracking-gap value per row, and the caller reduces it over whatever window it
cares about. ``train_rla.py`` already does both reductions on a metric's
``per_row`` output: ``_aggregate`` (RMS over a move's rows) and ``PathEnv.score``
(``.max()`` over a candidate's rows) are exactly the PDF's rms/peak formulas
applied to this module's per-row gap. So swapping which channel the gap is read
from (current vs. position) is the whole change needed here; whether a caller
restricts those rows to the settling window (``Segment.i1:i2``) rather than the
full move (``i0:i2``) is a windowing choice made where the metric is consumed,
not in the metric itself.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from utils import get_block, joint_cols

SCORE_COL = "score"


class EvaluationMetric(ABC):
    """Per-row score to minimize. Implement ``needs`` and ``per_row``."""

    @abstractmethod
    def needs(self) -> list[str]:
        """Per-joint channel bases this metric reads, e.g. ``["target_current",
        "actual_current"]``. Each must be a recording channel or one the distill
        model predicts, or ``add_score`` raises.
        """

    @abstractmethod
    def per_row(self, df) -> np.ndarray:
        """Score for every row of ``df``, shape ``(n,)``."""


class CurrentGapMetric(EvaluationMetric):
    """Current-tracking gap: ``|actual_current - target_current|`` summed over joints."""

    def needs(self) -> list[str]:
        return ["target_current", "actual_current"]

    def per_row(self, df) -> np.ndarray:
        gap = np.abs(get_block(df, "actual_current") - get_block(df, "target_current"))
        return gap.sum(axis=1)


class PositionGapMetric(EvaluationMetric):
    """Position-tracking gap: ``|actual_q - target_q|`` summed over joints, rad.

    This is what the PDF's ``peak``/``rms`` are computed from: applying
    ``.max()`` to this over a candidate's rows gives peak overshoot,
    ``sqrt(mean(x**2))`` gives RMS position error (``train_rla.py``'s
    ``_aggregate`` and ``PathEnv.score`` already do exactly that — see the
    module docstring). Sums the six joints rather than keeping them separate or
    reporting only the worst joint, matching ``CurrentGapMetric``'s reduction so
    it drops into the same ``metric = ...Metric()`` call sites and the same
    ``OBJECTIVE``/``PATH_OBJECTIVE`` weighting without further changes.

    Needs ``actual_q``, which real recordings already carry (unlike
    ``actual_current``, URSim's ``actual_q`` isn't a degenerate 0.0 — it tracks
    the commanded trajectory kinematically). The gap this reads is therefore
    real on real recordings today; using it inside ``train_rla.py``'s candidate
    scoring needs the distill model to predict ``actual_q`` first (candidate
    frames from ``dynamics.Dynamics.frame`` don't emit an ``actual_q`` column
    yet, only ``actual_current`` — the model has nothing to overwrite until it
    does).
    """

    def needs(self) -> list[str]:
        return ["target_q", "actual_q"]

    def per_row(self, df) -> np.ndarray:
        gap = np.abs(get_block(df, "actual_q") - get_block(df, "target_q"))
        return gap.sum(axis=1)


def add_score(df, metric: EvaluationMetric):
    """Return ``df`` with a ``score`` column from ``metric``.

    Checks the channels the metric needs are present, so a missing column raises
    here rather than later.
    """
    missing = [c for base in metric.needs() for c in joint_cols(base) if c not in df]
    if missing:
        raise ValueError(f"metric needs columns not in the recording: {missing}")
    df = df.copy()                          # defragment: recordings are very wide
    df[SCORE_COL] = metric.per_row(df)
    return df
