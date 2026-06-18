r"""Calibration metrics for whether an arm's confidences are trustworthy.

When an arm (or model) emits a confidence for its answer, calibration asks whether
that confidence matches the empirical hit rate: among answers asserted at 0.8
confidence, are ~80% correct? This module provides the proper scoring rule (Brier
score), the Expected Calibration Error (binned gap between confidence and accuracy),
and the reliability-curve bins used to draw a reliability diagram. These let the
benchmark report not just *whether* an arm is right but whether its self-reported
confidence can be trusted -- relevant when comparing systems that abstain or rank by
score.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ReliabilityBin",
    "ReliabilityReport",
    "brier_score",
    "expected_calibration_error",
    "reliability_curve",
]


def _validate(
    probs: Sequence[float], outcomes: Sequence[bool]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError("probs and outcomes must align")
    if p.size == 0:
        raise ValueError("need at least one observation")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    return p, y


def brier_score(probs: Sequence[float], outcomes: Sequence[bool]) -> float:
    r"""Mean squared error of probabilistic predictions, :math:`\tfrac1N\sum (p_i-y_i)^2`.

    Lower is better; 0 is perfect. A strictly proper scoring rule.
    """
    p, y = _validate(probs, outcomes)
    return float(np.mean((p - y) ** 2))


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One reliability-diagram bin."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class ReliabilityReport:
    """Reliability-curve bins plus the summary calibration error."""

    bins: tuple[ReliabilityBin, ...]
    ece: float


def reliability_curve(
    probs: Sequence[float], outcomes: Sequence[bool], n_bins: int = 10
) -> ReliabilityReport:
    """Bin predictions into equal-width confidence bins and report calibration.

    Each non-empty bin reports its mean confidence vs. empirical accuracy; the ECE is
    the count-weighted mean absolute gap between the two.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    p, y = _validate(probs, outcomes)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    ece = 0.0
    n = p.size
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        mean_conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        bins.append(ReliabilityBin(float(lo), float(hi), count, mean_conf, acc))
        ece += (count / n) * abs(mean_conf - acc)
    return ReliabilityReport(bins=tuple(bins), ece=ece)


def expected_calibration_error(
    probs: Sequence[float], outcomes: Sequence[bool], n_bins: int = 10
) -> float:
    """Return the Expected Calibration Error (count-weighted |confidence - accuracy|)."""
    return reliability_curve(probs, outcomes, n_bins).ece
