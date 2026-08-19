"""Score a recording's rows into one per-row value the optimizer minimizes.

``EvaluationMetric`` is the interface:

    needs()      -> per-joint channel bases this metric reads
    per_row(df)  -> (n,) score, one value per row

The default ``GapMetric("q")`` is the position-tracking gap. A subclass can read
other channels (currents, jerk, a mix); each must be a channel the recording
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


class GapMetric(EvaluationMetric):
    """Tracking gap ``|actual_X - target_X|`` summed over joints, X = ``quantity``.

    ``"q"`` (rad) is the reality gap itself: overshoot and the ring-down after a
    stop. ``"current"`` (A) scores the torque it took instead. The quantity has to
    be one the distill model fills in (``DistillModel.predicts``).
    """

    def __init__(self, quantity: str = "q"):
        self.quantity = quantity

    def needs(self) -> list[str]:
        return [f"target_{self.quantity}", f"actual_{self.quantity}"]

    def per_row(self, df) -> np.ndarray:
        t, a = self.needs()
        return np.abs(get_block(df, a) - get_block(df, t)).sum(axis=1)


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
