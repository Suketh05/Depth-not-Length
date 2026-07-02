r"""Exact small-sample inference: Clopper--Pearson intervals and exact McNemar.

The paper's boundary cells sit exactly where asymptotic inference breaks: at
synthetic depth 3 the typed store recovers 40/40 chains (:math:`\hat p = 1`),
so bootstrap and Wald intervals degenerate. The paper prescribes exact methods
there -- worked examples (j) (Figure fig:chain) and (n) (Figure
fig:supersession) in app:worked, and the fig:forest caption ("exact/score
interval (Wilson/Clopper--Pearson) or McNemar exact"). This module implements
those exact methods; the Wilson interval in
:mod:`membench.metrics.factorization` remains the default away from the
boundary.

References
----------
Clopper, C. J. & Pearson, E. S. (1934). "The use of confidence or fiducial
limits illustrated in the case of the binomial." *Biometrika*, 26(4), 404--413.
McNemar, Q. (1947). "Note on the sampling error of the difference between
correlated proportions or percentages." *Psychometrika*, 12(2), 153--157.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy import stats

__all__ = [
    "ExactBinomialInterval",
    "McNemarExactResult",
    "clopper_pearson",
]


@dataclass(frozen=True, slots=True)
class ExactBinomialInterval:
    """An exact (Clopper--Pearson) confidence interval for a binomial proportion.

    Attributes
    ----------
    estimate : float
        The point estimate ``successes / n``.
    low : float
        Exact lower confidence limit (0.0 when ``successes == 0``).
    high : float
        Exact upper confidence limit (1.0 when ``successes == n``).
    successes : int
        Number of successes observed.
    n : int
        Number of trials.
    alpha : float
        Total two-sided error rate; the interval has confidence ``1 - alpha``.
    method : str
        Always ``"clopper_pearson"``.
    """

    estimate: float
    low: float
    high: float
    successes: int
    n: int
    alpha: float
    method: str = "clopper_pearson"


@dataclass(frozen=True, slots=True)
class McNemarExactResult:
    """Outcome of the exact (binomial) McNemar test on paired binary outcomes.

    Attributes
    ----------
    statistic : float
        Signed discordance ``b - c`` (tasks only A got right minus tasks only
        B got right); positive means A wins more discordant pairs.
    p_value : float
        Exact two-sided p-value ``min(1, 2 * P(X <= min(b, c)))`` with
        ``X ~ Binomial(b + c, 1/2)`` -- the conservative doubled-tail form.
    mid_p : float
        Mid-p variant: the doubled tail counting only *half* the probability of
        the observed count, ``min(1, 2 * P(X <= k) - P(X = k))`` with
        ``k = min(b, c)``. Always ``<= p_value``; less conservative, better
        calibrated on average (Lancaster 1961).
    b : int
        Discordant pairs where A is correct and B is not.
    c : int
        Discordant pairs where B is correct and A is not.
    method : str
        Always ``"mcnemar_exact"``.
    """

    statistic: float
    p_value: float
    mid_p: float
    b: int
    c: int
    method: str = "mcnemar_exact"


def clopper_pearson(successes: int, n: int, *, alpha: float = 0.05) -> ExactBinomialInterval:
    r"""Compute the exact Clopper--Pearson interval via the Beta-quantile formulation.

    The interval inverts the binomial tail tests directly. Writing ``s`` for the
    number of successes, the equal-tailed exact limits are the Beta quantiles

    .. math::

        L = B^{-1}(\alpha/2;\; s,\; n - s + 1), \qquad
        U = B^{-1}(1 - \alpha/2;\; s + 1,\; n - s),

    with the boundary conventions :math:`L = 0` when ``s == 0`` and
    :math:`U = 1` when ``s == n``. The Beta form is an identity, not an
    approximation: :math:`P(X \ge s \mid p) = I_p(s, n - s + 1)` (the regularized
    incomplete Beta function), so the quantile solves the tail equation exactly.
    At the boundary the interval collapses to a closed form -- for ``s == n``,
    :math:`L = (\alpha/2)^{1/n}`, which is the paper's worked example (j):
    40/40 recovered chains give lower limit :math:`0.025^{1/40} = 0.912`
    (Figure fig:chain).

    Parameters
    ----------
    successes : int
        Number of successes, ``0 <= successes <= n``.
    n : int
        Number of trials, ``n >= 1``.
    alpha : float, optional
        Total two-sided error rate in ``(0, 1)`` (default ``0.05`` for a 95%
        interval).

    Returns
    -------
    ExactBinomialInterval
        The exact equal-tailed interval; coverage is guaranteed to be at least
        ``1 - alpha`` for every ``n`` and true ``p`` (conservative).
    """
    if n < 1:
        raise ValueError("n (number of trials) must be >= 1")
    if not 0 <= successes <= n:
        raise ValueError("require 0 <= successes <= n")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    if successes == 0:
        low = 0.0
    else:
        low = float(stats.beta.ppf(alpha / 2.0, successes, n - successes + 1))
    if successes == n:
        high = 1.0
    else:
        high = float(stats.beta.ppf(1.0 - alpha / 2.0, successes + 1, n - successes))

    return ExactBinomialInterval(
        estimate=successes / n,
        low=low,
        high=high,
        successes=successes,
        n=n,
        alpha=alpha,
    )
