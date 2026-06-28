r"""Split (inductive) conformal prediction intervals for regression.

Conformal prediction wraps any point predictor in a distribution-free interval that
attains *finite-sample marginal coverage* under exchangeability alone -- no parametric
noise model, no asymptotics. For a benchmark that compares retrieval arms on bounded,
non-normal error distributions this is the right tool for "predict the next score with a
guaranteed-coverage band".

This module implements the **split (inductive)** variant for regression with the
absolute-residual nonconformity score :math:`R_i = |y_i - \hat{y}_i|`. Given ``n``
calibration residuals and a miscoverage level ``alpha``, the half-width is the conformal
quantile

.. math::

    \hat{q} = R_{(k)}, \qquad k = \lceil (n + 1)(1 - \alpha) \rceil ,

i.e. the ``k``-th smallest calibration residual (equivalently, the level-``k/n``
inverse-CDF / type-1 empirical quantile of the residuals). When
:math:`k > n` -- too few calibration points for the requested level -- the finite-sample
guarantee can only be met by an unbounded band, so the half-width is ``+inf``. The
symmetric prediction interval at a test point is then
:math:`[\hat{y} - \hat{q},\ \hat{y} + \hat{q}]`, and for an exchangeable test point

.. math::

    \Pr\big(y \in [\hat{y} - \hat{q},\ \hat{y} + \hat{q}]\big)
        = \frac{\lceil (n + 1)(1 - \alpha) \rceil}{n + 1} \ge 1 - \alpha .

References
----------
.. [1] V. Vovk, A. Gammerman, and G. Shafer. *Algorithmic Learning in a Random World.*
       Springer, 2005.
.. [2] J. Lei, M. G'Sell, A. Rinaldo, R. J. Tibshirani, and L. Wasserman.
       "Distribution-Free Predictive Inference for Regression." *Journal of the American
       Statistical Association*, 113(523):1094-1111, 2018.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["ConformalInterval", "conformal_quantile", "split_conformal_interval"]


@dataclass(frozen=True, slots=True)
class ConformalInterval:
    """A symmetric split-conformal prediction interval.

    Attributes
    ----------
    point : float
        The model's point prediction at the test input.
    low, high : float
        The lower and upper interval bounds, ``point -/+ half_width``. They are
        ``-inf`` / ``+inf`` when the calibration set is too small for ``coverage``.
    half_width : float
        The conformal quantile of the calibration residuals (the band radius).
    coverage : float
        The nominal coverage level ``1 - alpha`` the interval targets.
    """

    point: float
    low: float
    high: float
    half_width: float
    coverage: float


def conformal_quantile(
    residuals: Sequence[float] | NDArray[np.float64], alpha: float = 0.1
) -> float:
    r"""Return the split-conformal quantile of calibration residuals.

    The band radius :math:`\hat{q} = R_{(k)}` with
    :math:`k = \lceil (n + 1)(1 - \alpha) \rceil`, the ``k``-th smallest residual. If
    ``k > n`` the level is not attainable in finite samples and ``+inf`` is returned.

    Parameters
    ----------
    residuals : Sequence[float]
        Nonconformity scores on the calibration set, typically
        :math:`|y_i - \hat{y}_i|`. Must be non-empty.
    alpha : float, optional
        Target miscoverage level in ``(0, 1)``; coverage is ``1 - alpha``. Default ``0.1``.

    Returns
    -------
    float
        The conformal quantile (interval half-width), or ``+inf`` if ``n`` is too small.

    Raises
    ------
    ValueError
        If ``alpha`` is not in ``(0, 1)`` or ``residuals`` is empty.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    r = np.sort(np.asarray(residuals, dtype=np.float64))
    n = int(r.size)
    if n == 0:
        raise ValueError("need at least one calibration residual")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return math.inf
    return float(r[k - 1])


def split_conformal_interval(
    calib_y_true: Sequence[float],
    calib_y_pred: Sequence[float],
    prediction: float,
    *,
    alpha: float = 0.1,
) -> ConformalInterval:
    r"""Build a symmetric split-conformal interval around a point prediction.

    Computes absolute calibration residuals :math:`|y_i - \hat{y}_i|`, takes their
    conformal quantile :math:`\hat{q}` (see :func:`conformal_quantile`), and returns the
    band :math:`[\text{prediction} - \hat{q},\ \text{prediction} + \hat{q}]`.

    Parameters
    ----------
    calib_y_true : Sequence[float]
        Observed targets on the held-out calibration split.
    calib_y_pred : Sequence[float]
        Model predictions on the same calibration split (must align with
        ``calib_y_true``).
    prediction : float
        The model's point prediction at the test input to wrap in an interval.
    alpha : float, optional
        Target miscoverage level in ``(0, 1)``; coverage is ``1 - alpha``. Default ``0.1``.

    Returns
    -------
    ConformalInterval
        The point, bounds, half-width, and nominal coverage. Bounds are infinite when
        the calibration set is too small for the requested level.

    Raises
    ------
    ValueError
        If the calibration arrays are not one-dimensional and aligned, or ``alpha`` is
        out of range, or the calibration set is empty.
    """
    y = np.asarray(calib_y_true, dtype=np.float64)
    yhat = np.asarray(calib_y_pred, dtype=np.float64)
    if y.ndim != 1 or yhat.ndim != 1:
        raise ValueError("calibration arrays must be one-dimensional")
    if y.shape != yhat.shape:
        raise ValueError("calibration truth and predictions must align")
    q = conformal_quantile(np.abs(y - yhat), alpha)
    point = float(prediction)
    return ConformalInterval(
        point=point,
        low=point - q,
        high=point + q,
        half_width=q,
        coverage=1.0 - alpha,
    )
