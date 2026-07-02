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

__all__ = [
    "ExactBinomialInterval",
    "McNemarExactResult",
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
