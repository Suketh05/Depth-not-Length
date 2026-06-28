r"""Studentized (bootstrap-t) and subsampling confidence intervals.

These two methods complement the percentile / BCa / cluster bootstraps in
:mod:`membench.stats.bootstrap` with *pivot-based* intervals, which invert the
sampling distribution of a (rate-normalised) root rather than reading raw
quantiles of the bootstrap statistic.

Studentized bootstrap-t
-----------------------
For a sample :math:`x` with estimate :math:`\hat\theta` and standard-error
estimate :math:`\widehat{SE}`, the bootstrap-t pivots on the studentized root

.. math::

    t^*_b = \frac{\hat\theta^*_b - \hat\theta}{\widehat{SE}^*_b},

computed on each resample :math:`b` (sampling the original data *with*
replacement). With :math:`t^*_{(\alpha)}` the empirical :math:`\alpha`-quantile of
the ``t*`` values, the equi-tailed :math:`100(1-\alpha)\%` interval is

.. math::

    \Big[\; \hat\theta - t^*_{(1-\alpha/2)}\,\widehat{SE},\;\;
            \hat\theta - t^*_{(\alpha/2)}\,\widehat{SE} \;\Big].

Note the inversion: the *upper* quantile of the pivot sets the *lower* endpoint.
The bootstrap-t is second-order accurate for location statistics and is the
recommended interval when a (approximately) pivotal studentized statistic is
available (Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993,
Chapter 12, Algorithm 12.1).

Subsampling
-----------
Subsampling (Politis, Romano & Wolf, *Subsampling*, 1999, Ch. 2) draws
subsamples of size :math:`b < n` *without* replacement and rescales the root by
the convergence rate :math:`\tau` (for the mean, :math:`\tau_k = k^{1/2}`):

.. math::

    L_{n,b}(x) = \frac{1}{N}\sum_i \mathbf 1\!\left\{
        \tau_b\,(\hat\theta_{b,i} - \hat\theta_n) \le x \right\}
    \;\xrightarrow{}\; L(x),

the limiting law of :math:`\tau_n(\hat\theta_n - \theta)`. Inverting the
:math:`\alpha`-quantiles :math:`c_{(\alpha)}` of that root gives

.. math::

    \Big[\; \hat\theta_n - c_{(1-\alpha/2)}/\tau_n,\;\;
            \hat\theta_n - c_{(\alpha/2)}/\tau_n \;\Big].

Subsampling is consistent under far weaker conditions than the bootstrap
(requiring only that a non-degenerate limit law exists), at the price of choosing
the block size :math:`b`.

All intervals are seeded for reproducibility and reuse the
:class:`~membench.stats.bootstrap.ConfidenceInterval` value object.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from membench.stats.bootstrap import ConfidenceInterval

__all__ = [
    "ConfidenceInterval",
    "studentized_ci",
    "studentized_t_statistic",
    "subsampling_ci",
]

Statistic = Callable[[NDArray[np.float64]], float]
StandardError = Callable[[NDArray[np.float64]], float]


def _mean(x: NDArray[np.float64]) -> float:
    """Return the arithmetic mean as a plain ``float``."""
    return float(np.mean(x))


def _mean_se(x: NDArray[np.float64]) -> float:
    r"""Return the standard error of the mean, :math:`s / \sqrt{n}` (``ddof=1``)."""
    return float(np.std(x, ddof=1) / np.sqrt(x.size))


def studentized_t_statistic(
    resample: Sequence[float],
    point_estimate: float,
    statistic: Statistic = _mean,
    *,
    standard_error: StandardError = _mean_se,
) -> float:
    r"""Pivotal studentized statistic for a single resample.

    Computes :math:`t^* = (\hat\theta^* - \hat\theta) / \widehat{SE}^*`, the root
    inverted by :func:`studentized_ci`.

    Parameters
    ----------
    resample
        The resampled observations :math:`x^*` (drawn with replacement).
    point_estimate
        The estimate :math:`\hat\theta` on the *original* sample.
    statistic
        Estimator applied to the resample; defaults to the mean.
    standard_error
        Standard-error estimator of ``statistic`` on the resample; defaults to
        the standard error of the mean.

    Returns
    -------
    float
        The studentized pivot :math:`t^*`.

    Raises
    ------
    ValueError
        If the resample's standard error is not strictly positive (a degenerate,
        constant resample), making the pivot undefined.
    """
    r = np.asarray(resample, dtype=np.float64)
    se = standard_error(r)
    if se <= 0.0:
        raise ValueError("standard error of the resample must be positive")
    return float((statistic(r) - point_estimate) / se)


def studentized_ci(
    values: Sequence[float],
    statistic: Statistic = _mean,
    *,
    standard_error: StandardError = _mean_se,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> ConfidenceInterval:
    r"""Studentized (bootstrap-t) confidence interval.

    Inverts the bootstrap distribution of the studentized root
    :math:`t^* = (\hat\theta^* - \hat\theta)/\widehat{SE}^*` (see module
    docstring). Second-order accurate for location statistics when a good
    standard-error estimate is supplied.

    Parameters
    ----------
    values
        Observed sample (at least two observations).
    statistic
        Estimator :math:`\hat\theta`; defaults to the mean.
    standard_error
        Standard-error estimator of ``statistic``; defaults to the standard
        error of the mean. For statistics without a closed-form SE an inner
        bootstrap SE estimator may be supplied here.
    confidence
        Nominal coverage, e.g. ``0.95``.
    n_resamples
        Number of with-replacement bootstrap resamples.
    seed
        Seed for the random generator (reproducible).

    Returns
    -------
    ConfidenceInterval
        ``point`` is the estimate on the original sample; ``low``/``high`` are
        the bootstrap-t bounds.

    Raises
    ------
    ValueError
        If fewer than two observations are given or the sample's standard error
        is not positive.

    References
    ----------
    Efron, B. & Tibshirani, R. (1993). *An Introduction to the Bootstrap*,
    Chapter 12 (Algorithm 12.1).
    """
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    if n < 2:
        raise ValueError("need at least two observations")
    theta_hat = statistic(x)
    se_hat = standard_error(x)
    if se_hat <= 0.0:
        raise ValueError("standard error of the sample must be positive")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    t_star = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        sample = x[idx[b]]
        se_b = standard_error(sample)
        # A degenerate (constant) resample has SE 0; fall back to the plug-in SE
        # so the pivot stays finite rather than dividing by zero.
        denom = se_b if se_b > 0.0 else se_hat
        t_star[b] = (statistic(sample) - theta_hat) / denom

    alpha = 1.0 - confidence
    t_lo, t_hi = np.quantile(t_star, [alpha / 2.0, 1.0 - alpha / 2.0])
    low = theta_hat - float(t_hi) * se_hat
    high = theta_hat - float(t_lo) * se_hat
    return ConfidenceInterval(theta_hat, low, high, confidence)


def subsampling_ci(
    values: Sequence[float],
    statistic: Statistic = _mean,
    *,
    confidence: float = 0.95,
    subsample_size: int | None = None,
    rate: float = 0.5,
    n_subsamples: int = 2000,
    seed: int = 0,
) -> ConfidenceInterval:
    r"""Subsampling confidence interval (Politis-Romano-Wolf).

    Draws ``n_subsamples`` subsamples of size :math:`b` *without* replacement,
    forms the rate-normalised root :math:`\tau_b(\hat\theta_b - \hat\theta_n)`
    with :math:`\tau_k = k^{\text{rate}}`, and inverts its quantiles (see module
    docstring).

    Parameters
    ----------
    values
        Observed sample (at least two observations).
    statistic
        Estimator :math:`\hat\theta`; defaults to the mean.
    confidence
        Nominal coverage, e.g. ``0.95``.
    subsample_size
        Block size :math:`b` with :math:`1 < b < n`. Must satisfy
        :math:`b\to\infty,\, b/n\to 0`; defaults to :math:`\lfloor\sqrt n\rfloor`.
    rate
        Convergence-rate exponent of :math:`\tau_k = k^{\text{rate}}`. The
        :math:`\sqrt n`-rate of the mean is ``0.5`` (the default).
    n_subsamples
        Number of randomly drawn subsamples used to estimate the root law.
    seed
        Seed for the random generator (reproducible).

    Returns
    -------
    ConfidenceInterval
        ``point`` is the estimate on the full sample; ``low``/``high`` are the
        subsampling bounds.

    Raises
    ------
    ValueError
        If fewer than two observations are given or ``subsample_size`` violates
        :math:`1 < b < n`.

    References
    ----------
    Politis, D. N., Romano, J. P. & Wolf, M. (1999). *Subsampling*, Chapter 2.
    """
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    if n < 2:
        raise ValueError("need at least two observations")
    b = subsample_size if subsample_size is not None else max(2, int(np.floor(np.sqrt(n))))
    if not 1 < b < n:
        raise ValueError("subsample_size must satisfy 1 < b < n")

    theta_hat = statistic(x)
    tau_n = float(n) ** rate
    tau_b = float(b) ** rate

    rng = np.random.default_rng(seed)
    roots = np.empty(n_subsamples, dtype=np.float64)
    for i in range(n_subsamples):
        sub = x[rng.choice(n, size=b, replace=False)]
        roots[i] = tau_b * (statistic(sub) - theta_hat)

    alpha = 1.0 - confidence
    c_lo, c_hi = np.quantile(roots, [alpha / 2.0, 1.0 - alpha / 2.0])
    low = theta_hat - float(c_hi) / tau_n
    high = theta_hat - float(c_lo) / tau_n
    return ConfidenceInterval(theta_hat, low, high, confidence)
