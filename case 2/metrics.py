"""Score a recording's rows into one per-row value the optimizer minimizes.

``EvaluationMetric`` is the interface:

    needs()      -> per-joint channel bases this metric reads
    per_row(df)  -> (n,) score, one value per row

The default ``CurrentGapMetric`` is the current-tracking gap. A subclass can read
other channels (position error, jerk, a mix); each must be a channel the recording
carries or the distill model predicts.
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
    """Position-tracking gap: ``|actual_q - target_q|`` (rad) summed over joints.

    Same shape as ``CurrentGapMetric`` but reads position instead of current --
    pair with a ``DistillModel`` that predicts ``actual_q``. Note the scale is
    very different (~1e-3 rad vs. several A), which matters if this score is
    combined with a cycle-time term elsewhere (see train_rla.py's OBJECTIVE).
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
