r"""Isotonic (monotone) regression via the Pool-Adjacent-Violators Algorithm.

Isotonic regression is the :math:`L_2` projection of an observed sequence onto the
cone of monotone sequences. For non-decreasing fits it solves

.. math::

    \min_{\hat{y}}\; \sum_{i=1}^{n} w_i\,(y_i - \hat{y}_i)^2
        \quad\text{subject to}\quad \hat{y}_1 \le \hat{y}_2 \le \cdots \le \hat{y}_n,

and the unique minimiser is computed exactly in :math:`O(n)` by the
Pool-Adjacent-Violators Algorithm (PAVA): scan left to right and, whenever a new
block violates monotonicity against its predecessor, pool the two into one block
whose value is their weight-averaged mean; pooling cascades backwards until order is
restored. Each fitted block value is the weighted mean of the raw points it covers.

This module powers the benchmark's central claim that *compliance decreases
monotonically with causal-chain depth*. :func:`monotone_trend_test` aggregates
compliance by depth, fits the best non-increasing curve, and reports an isotonic
coefficient of determination as the monotonicity statistic (the fraction of
variance explained by the monotone fit, in ``[0, 1]``; ``1`` for data already
monotone in the constrained direction, ``0`` when the best monotone fit is flat).

References
----------
Ayer, M., Brunk, H. D., Ewing, G. M., Reid, W. T., & Silverman, E. (1955). An
empirical distribution function for sampling with incomplete information. *Annals of
Mathematical Statistics*, 26(4), 641-647.

Barlow, R. E., Bartholomew, D. J., Bremner, J. M., & Brunk, H. D. (1972).
*Statistical Inference Under Order Restrictions: The Theory and Application of
Isotonic Regression*. Wiley.

Robertson, T., Wright, F. T., & Dykstra, R. L. (1988). *Order Restricted Statistical
Inference*. Wiley.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MonotoneTrendResult",
    "isotonic_regression",
    "monotone_trend_statistic",
    "monotone_trend_test",
]

# Public numeric inputs accept plain sequences or numpy arrays interchangeably.
FloatVector = Sequence[float] | NDArray[np.float64]


def _as_weights(weights: FloatVector | None, n: int) -> NDArray[np.float64]:
    """Return validated positive weights of length ``n`` (unit weights if ``None``)."""
    if weights is None:
        return np.ones(n, dtype=np.float64)
    wv = np.asarray(weights, dtype=np.float64)
    if wv.shape != (n,):
        raise ValueError("weights must match y in shape")
    if np.any(wv <= 0.0):
        raise ValueError("weights must be strictly positive")
    return wv


def isotonic_regression(
    y: FloatVector,
    weights: FloatVector | None = None,
    *,
    increasing: bool = True,
) -> NDArray[np.float64]:
    r"""Fit the monotone least-squares regression of ``y`` using PAVA.

    Parameters
    ----------
    y
        Observed values in their natural (e.g. depth) order.
    weights
        Optional strictly positive weights :math:`w_i`. Unit weights by default.
    increasing
        If ``True`` (default) fit a non-decreasing sequence; if ``False`` fit a
        non-increasing one (handled by negating ``y``, fitting, and negating back).

    Returns
    -------
    numpy.ndarray
        The fitted values, the same length as ``y``. Guaranteed monotone in the
        requested direction, and equal to ``y`` when ``y`` is already monotone
        (so the operator is idempotent).

    Notes
    -----
    Each fitted block equals the weighted mean of the raw points it pools, so the
    solution is the exact :math:`L_2` projection onto the monotone cone (Barlow et
    al., 1972).
    """
    yv = np.asarray(y, dtype=np.float64)
    if yv.ndim != 1:
        raise ValueError("y must be one-dimensional")
    n = int(yv.size)
    if n == 0:
        raise ValueError("y must be non-empty")
    wv = _as_weights(weights, n)

    sign = 1.0 if increasing else -1.0
    target = sign * yv

    # Stack of pooled blocks: parallel lists of (block mean, block weight, count).
    means: list[float] = []
    block_weights: list[float] = []
    counts: list[int] = []
    for i in range(n):
        value = float(target[i])
        weight = float(wv[i])
        count = 1
        # Pool backwards while the previous block violates non-decreasing order.
        while means and means[-1] > value:
            prev_value = means.pop()
            prev_weight = block_weights.pop()
            prev_count = counts.pop()
            total_weight = prev_weight + weight
            value = (prev_value * prev_weight + value * weight) / total_weight
            weight = total_weight
            count += prev_count
        means.append(value)
        block_weights.append(weight)
        counts.append(count)

    fitted = np.empty(n, dtype=np.float64)
    start = 0
    for value, count in zip(means, counts, strict=True):
        fitted[start : start + count] = value
        start += count
    return sign * fitted


def monotone_trend_statistic(
    y: FloatVector,
    weights: FloatVector | None = None,
    *,
    increasing: bool = True,
) -> float:
    r"""Return the isotonic coefficient of determination of ``y``.

    The statistic is the fraction of (weighted) variance explained by the best
    monotone fit,

    .. math::

        R^2_{\mathrm{iso}} = 1 - \frac{\sum_i w_i (y_i - \hat{y}_i)^2}
                                       {\sum_i w_i (y_i - \bar{y})^2},

    where :math:`\hat{y}` is the isotonic fit and :math:`\bar{y}` the weighted mean.
    It lies in ``[0, 1]``: ``1`` when ``y`` is already monotone in the requested
    direction, ``0`` when the best monotone fit collapses to the flat mean (the data
    trend opposes the constraint). Constant data is treated as trivially monotone
    (``1.0``).

    Parameters
    ----------
    y
        Observed values in their natural order.
    weights
        Optional strictly positive weights. Unit weights by default.
    increasing
        Direction of the constraint passed to :func:`isotonic_regression`.
    """
    yv = np.asarray(y, dtype=np.float64)
    if yv.ndim != 1 or yv.size == 0:
        raise ValueError("y must be a non-empty one-dimensional sequence")
    wv = _as_weights(weights, int(yv.size))

    fitted = isotonic_regression(yv, wv, increasing=increasing)
    wmean = float(np.average(yv, weights=wv))
    ss_total = float(np.sum(wv * (yv - wmean) ** 2))
    if ss_total == 0.0:
        return 1.0
    ss_resid = float(np.sum(wv * (yv - fitted) ** 2))
    return 1.0 - ss_resid / ss_total


@dataclass(frozen=True, slots=True)
class MonotoneTrendResult:
    """Isotonic fit of compliance against depth with its monotonicity statistic."""

    depths: tuple[int, ...]
    n_by_depth: tuple[int, ...]
    fitted: tuple[float, ...]
    statistic: float
    decreasing: bool


def monotone_trend_test(
    depths: Sequence[int],
    compliance: Sequence[float],
    *,
    decreasing: bool = True,
) -> MonotoneTrendResult:
    """Test whether compliance changes monotonically with depth.

    Observations are grouped by depth, the mean compliance per depth is computed
    (weighted by the group size), and the best monotone curve is fitted across
    depths in ascending order. ``decreasing=True`` (the benchmark's hypothesis that
    compliance falls as causal-chain depth grows) fits a non-increasing curve.

    Parameters
    ----------
    depths
        Integer causal-chain depth for each observation.
    compliance
        Compliance outcome per observation (binary ``0``/``1`` or a rate).
    decreasing
        If ``True`` (default) fit a non-increasing trend in depth; otherwise fit a
        non-decreasing one.

    Returns
    -------
    MonotoneTrendResult
        The sorted unique depths, the observation count and fitted compliance at
        each depth, and the isotonic monotonicity statistic.
    """
    dv = np.asarray(depths)
    cv = np.asarray(compliance, dtype=np.float64)
    if dv.shape != cv.shape:
        raise ValueError("depths and compliance must align")
    if dv.size == 0:
        raise ValueError("need at least one observation")

    unique = np.unique(dv)
    means = np.array([float(cv[dv == d].mean()) for d in unique], dtype=np.float64)
    counts = np.array([float(np.count_nonzero(dv == d)) for d in unique], dtype=np.float64)

    increasing = not decreasing
    fitted = isotonic_regression(means, counts, increasing=increasing)
    statistic = monotone_trend_statistic(means, counts, increasing=increasing)
    return MonotoneTrendResult(
        depths=tuple(int(d) for d in unique),
        n_by_depth=tuple(int(c) for c in counts),
        fitted=tuple(float(f) for f in fitted),
        statistic=statistic,
        decreasing=decreasing,
    )
