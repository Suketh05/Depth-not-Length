r"""Exact small-sample inference: Clopper--Pearson intervals and exact McNemar.

The benchmark's headline cells are often *boundary* proportions -- a typed store
that recovers a chain at every one of ``n`` paired queries (:math:`\hat p = 1`), or a
supersession contrast where one arm is right and the other wrong on a handful of
discordant tasks. At those cells the large-sample approximations the rest of
``membench.stats`` leans on (Wald, and even the Wilson score interval in
:mod:`membench.metrics.factorization`) misreport coverage: a Wald interval at
:math:`\hat p = 1` collapses to the degenerate point ``[1, 1]``, and a normal-theory
McNemar :math:`\chi^2` is untrustworthy when the discordant count is small. This
module supplies the *exact* counterparts the paper's worked examples and forest plot
prescribe for exactly those cells.

* :func:`clopper_pearson` -- the exact (Clopper--Pearson 1934) binomial confidence
  interval, obtained from the Beta-quantile inversion of the binomial tail. It is
  guaranteed-conservative (coverage :math:`\ge 1 - \alpha` for every true ``p``) and,
  unlike Wald/Wilson, stays honest at the ``0/n`` and ``n/n`` saturation cells, where
  it reduces to the closed forms :math:`[0,\,1-(\alpha/2)^{1/n}]` and
  :math:`[(\alpha/2)^{1/n},\,1]`.
* :func:`mcnemar_exact` -- the exact (binomial) McNemar test for a paired-binary
  contrast, conditioning on the discordant pairs and reading the two-sided p-value off
  the ``Binomial(b + c, 1/2)`` tail, plus the Lancaster mid-p variant that trades a
  little conservatism for calibration.

Paper anchors
-------------
* App. `app:worked` worked example (j) -- chain-recovery ``40/40`` reported with the
  exact interval ``[0.912, 1.000]`` (a Clopper--Pearson boundary cell); worked example
  (n) applies the same exact treatment to the supersession contrast.
* Fig. `fig:forest` (F024) caption -- prescribes exact/score intervals and an exact
  McNemar at boundary cells where :math:`\hat p = 1` (e.g. Brief vs. dense at ``d = 3``,
  ``40/40`` vs. ``27/40``).
* `sec:stats` -- statistical-methodology section listing Wilson intervals and exact
  paired tests as the reporting standard for the bounded, often-binary outcomes here.

See Also
--------
membench.metrics.factorization.wilson_interval : the large-sample score interval this
    module's :func:`clopper_pearson` conservatively contains; prefer Wilson away from
    the boundaries and Clopper--Pearson at the ``0/n`` / ``n/n`` cells.
membench.stats.hypothesis.mcnemar_test : the same exact test driven from paired boolean
    outcome vectors rather than the pre-tabulated discordant counts ``b`` and ``c``.

References
----------
Clopper, C. J. and Pearson, E. S. (1934). "The use of confidence or fiducial limits
illustrated in the case of the binomial." *Biometrika*, 26(4), 404--413.
McNemar, Q. (1947). "Note on the sampling error of the difference between correlated
proportions or percentages." *Psychometrika*, 12(2), 153--157.
Lancaster, H. O. (1961). "Significance tests in discrete distributions." *Journal of
the American Statistical Association*, 56(294), 223--234. (The mid-p correction.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats

__all__ = [
    "ClopperPearsonInterval",
    "McNemarExactResult",
    "clopper_pearson",
    "mcnemar_exact",
]


@dataclass(frozen=True, slots=True)
class ClopperPearsonInterval:
    """An exact Clopper--Pearson binomial confidence interval.

    Attributes
    ----------
    low, high : float
        Lower and upper confidence limits, both in ``[0, 1]`` with ``low <= high``.
    point : float
        The observed proportion ``successes / n`` the interval is built around.
    successes, n : int
        The observed success count and number of trials.
    alpha : float
        The two-sided miscoverage level; the interval has confidence ``1 - alpha`` and
        coverage guaranteed to be at least ``1 - alpha`` for every true proportion.
    """

    low: float
    high: float
    point: float
    successes: int
    n: int
    alpha: float


@dataclass(frozen=True, slots=True)
class McNemarExactResult:
    """Outcome of an exact (binomial) McNemar test on paired-binary discordances.

    Attributes
    ----------
    statistic : float
        The signed discordance ``b - c`` (positive favours the first arm).
    p_value : float
        The exact two-sided p-value: twice the smaller ``Binomial(b + c, 1/2)`` tail,
        clipped to ``1.0``. Equals ``1.0`` when there are no discordant pairs.
    mid_p_value : float
        The Lancaster mid-p two-sided p-value (half the point mass at the observed
        count is excluded from each tail); less conservative than ``p_value``.
    b, c : int
        The two discordant counts (first-arm-only and second-arm-only successes).
    n_discordant : int
        Their sum ``b + c`` -- the conditioning sample size the test is defined on.
    method : str
        Identifier of the test, ``"mcnemar_exact"``.
    """

    statistic: float
    p_value: float
    mid_p_value: float
    b: int
    c: int
    n_discordant: int
    method: str


def clopper_pearson(successes: int, n: int, alpha: float = 0.05) -> ClopperPearsonInterval:
    r"""Return the exact Clopper--Pearson binomial confidence interval.

    The interval inverts the two equal-tailed binomial tail tests. Writing
    :math:`x = \text{successes}`, the lower limit is the ``p`` that puts mass
    ``alpha/2`` in the upper tail :math:`P(X \ge x)` and the upper limit the ``p`` that
    puts ``alpha/2`` in the lower tail :math:`P(X \le x)`. Using the standard
    Beta--binomial tail identity these are exactly Beta quantiles:

    .. math::

        \text{low}  &= B^{-1}(\alpha/2;\; x,\; n - x + 1), \\
        \text{high} &= B^{-1}(1 - \alpha/2;\; x + 1,\; n - x),

    with the degenerate shape parameters handled by the closed-form boundaries
    ``low = 0`` when ``x = 0`` and ``high = 1`` when ``x = n``. There the surviving
    limit reduces to :math:`1 - (\alpha/2)^{1/n}` (for ``0/n``) and
    :math:`(\alpha/2)^{1/n}` (for ``n/n``) -- the forms the paper's ``40/40``
    worked example (App. `app:worked` (j)) and the ``\hat p = 1`` forest cells
    (Fig. `fig:forest`) rely on.

    Parameters
    ----------
    successes : int
        Number of observed successes, ``0 <= successes <= n``.
    n : int
        Number of Bernoulli trials, ``n >= 1``.
    alpha : float, optional
        Two-sided miscoverage level (default ``0.05`` for a 95% interval);
        must lie in ``(0, 1)``.

    Returns
    -------
    ClopperPearsonInterval
        The exact interval and the inputs it was built from.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 0 <= successes <= n:
        raise ValueError("require 0 <= successes <= n")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    half = alpha / 2.0
    # Degenerate Beta shapes (a = 0 or b = 0) are undefined for the ppf, so the
    # saturated boundaries are supplied in closed form; scipy handles the interior.
    low = 0.0 if successes == 0 else float(stats.beta.ppf(half, successes, n - successes + 1))
    high = (
        1.0 if successes == n else float(stats.beta.ppf(1.0 - half, successes + 1, n - successes))
    )
    return ClopperPearsonInterval(
        low=low,
        high=high,
        point=successes / n,
        successes=successes,
        n=n,
        alpha=alpha,
    )


def mcnemar_exact(b: int, c: int) -> McNemarExactResult:
    r"""Return the exact (binomial) McNemar test on the discordant counts ``b`` and ``c``.

    Given a paired-binary 2x2 table, the concordant cells carry no information about a
    difference; McNemar's exact test conditions on the ``n = b + c`` discordant pairs
    and asks whether they split evenly under :math:`H_0: p = 1/2`. The two-sided
    p-value is twice the smaller tail of the ``Binomial(n, 1/2)`` law, clipped to one:

    .. math::

        p = \min\!\Bigl(1,\; 2 \sum_{k=0}^{\min(b,c)} \binom{n}{k} 2^{-n}\Bigr).

    Because the null is symmetric the doubling is exact (both tails are equal), so the
    binomial coefficients are summed with :func:`math.comb` and divided by
    :math:`2^{n}` in exact integer arithmetic before the single float conversion -- the
    value is a dyadic rational and is returned without approximation error. The
    Lancaster mid-p variant excludes half of the point mass at the observed count from
    each tail, :math:`p_\text{mid} = \min(1,\, (2\sum_{k<\min(b,c)}\binom{n}{k} +
    \binom{n}{\min(b,c)})2^{-n})`, trading guaranteed conservatism for mean calibration.

    With no discordant pairs (``b = c = 0``) the test is undefined in the usual sense;
    the convention here returns ``p = 1.0`` (no evidence of a difference).

    Parameters
    ----------
    b : int
        Count of pairs where the first arm succeeds and the second fails (``b >= 0``).
    c : int
        Count of pairs where the second arm succeeds and the first fails (``c >= 0``).

    Returns
    -------
    McNemarExactResult
        The signed discordance, the exact two-sided p-value, and its mid-p counterpart.
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts b and c must be non-negative")

    n = b + c
    statistic = float(b - c)
    if n == 0:
        return McNemarExactResult(
            statistic=statistic,
            p_value=1.0,
            mid_p_value=1.0,
            b=b,
            c=c,
            n_discordant=0,
            method="mcnemar_exact",
        )

    m = min(b, c)
    total = 1 << n  # 2 ** n, exact
    tail = sum(math.comb(n, k) for k in range(m + 1))  # P(X <= m) numerator over 2**n
    point_mass = math.comb(n, m)  # the observed count's own probability numerator

    p_value = min(1.0, (2 * tail) / total)
    mid_p_value = min(1.0, (2 * tail - point_mass) / total)
    return McNemarExactResult(
        statistic=statistic,
        p_value=p_value,
        mid_p_value=mid_p_value,
        b=b,
        c=c,
        n_discordant=n,
        method="mcnemar_exact",
    )
