r"""Linear quantile regression of compliance on causal-chain depth.

Ordinary least squares fits the conditional *mean*; quantile regression fits a
conditional *quantile*, which is what a depth study actually wants -- the median
compliance trend is robust to the bimodal "complied / ignored" split, and the upper
and lower quantile slopes reveal whether the worst-case (low-quantile) compliance
collapses with depth faster than the typical case does.

For a quantile level :math:`\tau \in (0, 1)` the fit minimises the **tilted (pinball)
loss** of Koenker & Bassett (1978):

.. math::

    (\hat a, \hat b) = \arg\min_{a, b} \; \frac{1}{n}\sum_{i=1}^{n}
        \rho_\tau\!\bigl(y_i - a - b\,x_i\bigr),
    \qquad
    \rho_\tau(u) = u\,\bigl(\tau - \mathbb{1}\{u < 0\}\bigr).

Because :math:`\rho_\tau` is piecewise linear, the problem is the linear program

.. math::

    \min_{a, b, u^+, u^-} \; \tau\, \mathbf{1}^\top u^+ + (1-\tau)\,\mathbf{1}^\top u^-
    \quad\text{s.t.}\quad y_i = a + b\,x_i + u_i^+ - u_i^-,\; u^+, u^- \ge 0,

solved here exactly with :func:`scipy.optimize.linprog` (the HiGHS simplex). At
:math:`\tau = 0.5` the loss is :math:`\tfrac12|u|`, so the fit is least-absolute-
deviations (median) regression.

References
----------
Koenker, R., & Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1),
33--50. https://doi.org/10.2307/1913643
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

__all__ = ["QuantileFit", "pinball_loss", "quantile_regression", "quantile_regression_path"]


@dataclass(frozen=True, slots=True)
class QuantileFit:
    r"""A fitted line for one quantile level :math:`\tau`.

    Attributes
    ----------
    tau
        Quantile level in ``(0, 1)`` that was fitted.
    intercept
        Estimated intercept :math:`\hat a`.
    slope
        Estimated slope :math:`\hat b` (change in the :math:`\tau`-quantile of ``y``
        per unit ``x``).
    pinball_loss
        Mean tilted loss of the fitted residuals at this ``tau`` (the minimised
        objective). At ``tau = 0.5`` this equals ``0.5 * mean|residual|``.
    n_obs
        Number of observations used.
    """

    tau: float
    intercept: float
    slope: float
    pinball_loss: float
    n_obs: int

    def predict(self, x: float) -> float:
        r"""Return the fitted :math:`\tau`-quantile of ``y`` at the covariate ``x``."""
        return self.intercept + self.slope * x


def pinball_loss(residuals: Sequence[float], tau: float) -> float:
    r"""Return the mean tilted (pinball) loss of ``residuals`` at level ``tau``.

    The pinball loss of a residual :math:`u = y - \hat y` is
    :math:`\rho_\tau(u) = u\,(\tau - \mathbb{1}\{u < 0\})`, i.e. ``tau * u`` when the
    prediction is too low (``u > 0``) and ``(tau - 1) * u`` when it is too high. This
    function returns the mean over all residuals.

    Parameters
    ----------
    residuals
        Residuals :math:`y_i - \hat y_i`.
    tau
        Quantile level in ``(0, 1)``.

    Returns
    -------
    float
        Mean pinball loss.

    Raises
    ------
    ValueError
        If ``residuals`` is empty or ``tau`` is not in ``(0, 1)``.
    """
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must be in (0, 1)")
    u = np.asarray(residuals, dtype=np.float64)
    if u.size == 0:
        raise ValueError("residuals must be non-empty")
    loss = np.where(u >= 0, tau * u, (tau - 1.0) * u)
    return float(loss.mean())


def quantile_regression(
    x: Sequence[float],
    y: Sequence[float],
    tau: float = 0.5,
) -> QuantileFit:
    r"""Fit ``y ~ a + b*x`` at quantile ``tau`` by minimising the pinball loss.

    The estimator is the Koenker & Bassett (1978) regression quantile, obtained as the
    exact solution of the equivalent linear program via :func:`scipy.optimize.linprog`
    (HiGHS). At ``tau = 0.5`` this is least-absolute-deviations (median) regression.

    Parameters
    ----------
    x
        Covariate (e.g. causal-chain depth), one value per observation.
    y
        Response (e.g. compliance rate), one value per observation.
    tau
        Quantile level in ``(0, 1)``. Defaults to the median.

    Returns
    -------
    QuantileFit
        Fitted intercept, slope, and the minimised mean pinball loss.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` differ in length, fewer than two observations are given, or
        ``tau`` is not in ``(0, 1)``.
    RuntimeError
        If the underlying linear program fails to converge.
    """
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must be in (0, 1)")
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    if xv.shape != yv.shape:
        raise ValueError("x and y must have the same length")
    n = int(xv.size)
    if n < 2:
        raise ValueError("need at least two observations")

    # Decision vector: [a, b, u+ (n), u- (n)]. Intercept/slope are free; the positive
    # and negative residual parts are non-negative. Objective weights the parts by the
    # tilt so the optimum reproduces the pinball loss.
    cost = np.concatenate(([0.0, 0.0], np.full(n, tau), np.full(n, 1.0 - tau)))
    a_eq = np.zeros((n, 2 + 2 * n), dtype=np.float64)
    a_eq[:, 0] = 1.0
    a_eq[:, 1] = xv
    a_eq[:, 2 : 2 + n] = np.eye(n)
    a_eq[:, 2 + n :] = -np.eye(n)

    bounds: list[tuple[float | None, float | None]] = [(None, None), (None, None)]
    bounds.extend((0.0, None) for _ in range(2 * n))

    result = linprog(cost, A_eq=a_eq, b_eq=yv, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"quantile-regression LP did not converge: {result.message}")

    intercept = float(result.x[0])
    slope = float(result.x[1])
    residuals = yv - (intercept + slope * xv)
    loss = pinball_loss(residuals.tolist(), tau)
    return QuantileFit(
        tau=tau,
        intercept=intercept,
        slope=slope,
        pinball_loss=loss,
        n_obs=n,
    )


def quantile_regression_path(
    x: Sequence[float],
    y: Sequence[float],
    taus: Sequence[float] = (0.1, 0.5, 0.9),
) -> tuple[QuantileFit, ...]:
    """Fit one :class:`QuantileFit` per quantile level in ``taus``.

    Parameters
    ----------
    x
        Covariate, one value per observation.
    y
        Response, one value per observation.
    taus
        Quantile levels, each in ``(0, 1)``. Defaults to the lower decile, median, and
        upper decile.

    Returns
    -------
    tuple of QuantileFit
        One fit per level, in the order given by ``taus``.
    """
    return tuple(quantile_regression(x, y, tau) for tau in taus)
