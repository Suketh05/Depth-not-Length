r"""Equivalence testing via Schuirmann's Two One-Sided Tests (TOST).

A *significant* difference is not the same as a *meaningful* one, and -- just as
important for this benchmark -- a non-significant difference is **not** evidence that
two arms are equivalent (absence of evidence is not evidence of absence). To claim
"arm A and arm B perform equivalently" one must actively test for equivalence inside a
pre-specified margin, not merely fail to reject the null of no difference.

TOST does exactly that. To test whether the true difference ``theta = mu_a - mu_b``
lies inside an equivalence margin ``[-delta, +delta]`` it runs two one-sided tests:

* lower test -- :math:`H_{0,L}: \theta \le -\delta` against :math:`H_{1,L}: \theta > -\delta`,
  with statistic :math:`t_L = (\hat\theta + \delta) / \mathrm{SE}` (upper-tail p-value);
* upper test -- :math:`H_{0,U}: \theta \ge +\delta` against :math:`H_{1,U}: \theta < +\delta`,
  with statistic :math:`t_U = (\hat\theta - \delta) / \mathrm{SE}` (lower-tail p-value).

Equivalence is concluded only if *both* one-sided nulls are rejected, so the overall
TOST p-value is the **maximum** of the two one-sided p-values, and equivalence holds
at level ``alpha`` iff that maximum is below ``alpha``. This is operationally identical
to the well-known interval-inclusion rule: equivalence holds iff the ordinary
``100(1 - 2*alpha)%`` confidence interval for ``theta`` falls entirely within
``[-delta, +delta]`` -- hence that interval is returned alongside the p-values.

References
----------
Schuirmann, D. J. (1987). "A comparison of the two one-sided tests procedure and the
power approach for assessing the equivalence of average bioavailability." *Journal of
Pharmacokinetics and Biopharmaceutics*, 15(6), 657--680.
Lakens, D. (2017). "Equivalence tests: A practical primer for t tests, correlations,
and meta-analyses." *Social Psychological and Personality Science*, 8(4), 355--362.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "TOSTResult",
    "tost_independent_samples",
    "tost_two_means",
    "tost_two_proportions",
]


@dataclass(frozen=True, slots=True)
class TOSTResult:
    """Outcome of a two one-sided tests (TOST) equivalence test.

    Attributes
    ----------
    p_value : float
        Overall TOST p-value -- the maximum of the two one-sided p-values.
        Equivalence holds at level ``alpha`` iff ``p_value < alpha``.
    lower_p : float
        p-value of the lower one-sided test (null: difference ``<= -delta``).
    upper_p : float
        p-value of the upper one-sided test (null: difference ``>= +delta``).
    equivalent : bool
        Whether equivalence within ``[-delta, +delta]`` is concluded at ``alpha``.
    ci : tuple[float, float]
        The ``100(1 - 2*alpha)%`` confidence interval for the difference; equivalence
        holds iff this interval lies entirely inside ``[-delta, +delta]``.
    """

    p_value: float
    lower_p: float
    upper_p: float
    equivalent: bool
    ci: tuple[float, float]


def _validate(delta: float, alpha: float) -> None:
    if delta <= 0:
        raise ValueError("delta (equivalence margin) must be positive")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")


def tost_two_means(
    mean_a: float,
    sd_a: float,
    n_a: int,
    mean_b: float,
    sd_b: float,
    n_b: int,
    *,
    delta: float,
    alpha: float = 0.05,
    pooled: bool = False,
) -> TOSTResult:
    r"""Run TOST for the equivalence of two independent-group means.

    Tests whether ``mean_a - mean_b`` lies within ``[-delta, +delta]`` using two
    one-sided t-tests on the difference of means. By default the Welch (unequal
    variance) standard error and Welch--Satterthwaite degrees of freedom are used;
    set ``pooled=True`` for the equal-variance pooled-SD two-sample t-test.

    Parameters
    ----------
    mean_a, sd_a, n_a : float, float, int
        Sample mean, sample standard deviation (``ddof = 1``) and size of group A.
    mean_b, sd_b, n_b : float, float, int
        The same summary statistics for group B.
    delta : float
        Positive equivalence margin; the symmetric bounds are ``[-delta, +delta]``.
    alpha : float, optional
        One-sided significance level for each test (default ``0.05``); the returned
        interval is the ``100(1 - 2*alpha)%`` confidence interval.
    pooled : bool, optional
        Use the pooled-variance Student t-test instead of Welch's (default ``False``).

    Returns
    -------
    TOSTResult
        The two one-sided p-values, their maximum, the equivalence verdict and the
        confidence interval for the mean difference.
    """
    _validate(delta, alpha)
    if n_a < 2 or n_b < 2:
        raise ValueError("each group needs at least two observations")
    if sd_a < 0 or sd_b < 0:
        raise ValueError("standard deviations must be non-negative")

    diff = mean_a - mean_b
    var_a = sd_a**2
    var_b = sd_b**2
    if pooled:
        dof = float(n_a + n_b - 2)
        pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / dof
        se = math.sqrt(pooled_var * (1.0 / n_a + 1.0 / n_b))
    else:
        term_a = var_a / n_a
        term_b = var_b / n_b
        se = math.sqrt(term_a + term_b)
        if se == 0.0:
            dof = float(n_a + n_b - 2)
        else:
            dof = (term_a + term_b) ** 2 / (term_a**2 / (n_a - 1) + term_b**2 / (n_b - 1))

    return _assemble(diff, se, delta, alpha, dist=stats.t, dof=dof)


def tost_independent_samples(
    a: Sequence[float],
    b: Sequence[float],
    *,
    delta: float,
    alpha: float = 0.05,
    pooled: bool = False,
) -> TOSTResult:
    """Run :func:`tost_two_means` directly on two raw samples.

    Convenience wrapper that computes each group's mean, sample standard deviation
    (``ddof = 1``) and size from the observations themselves.

    Parameters
    ----------
    a, b : Sequence[float]
        The two independent samples being tested for equivalence.
    delta : float
        Positive equivalence margin (see :func:`tost_two_means`).
    alpha : float, optional
        One-sided significance level (default ``0.05``).
    pooled : bool, optional
        Use the pooled-variance t-test instead of Welch's (default ``False``).

    Returns
    -------
    TOSTResult
        The equivalence-test outcome.
    """
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    if av.size < 2 or bv.size < 2:
        raise ValueError("each sample needs at least two observations")
    return tost_two_means(
        float(av.mean()),
        float(av.std(ddof=1)),
        int(av.size),
        float(bv.mean()),
        float(bv.std(ddof=1)),
        int(bv.size),
        delta=delta,
        alpha=alpha,
        pooled=pooled,
    )


def tost_two_proportions(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
    *,
    delta: float,
    alpha: float = 0.05,
) -> TOSTResult:
    r"""Run TOST for the equivalence of two independent proportions (large-sample z).

    Tests whether ``p_a - p_b`` lies within ``[-delta, +delta]`` using two one-sided
    z-tests with the unpooled (Wald) standard error
    :math:`\sqrt{p_a(1-p_a)/n_a + p_b(1-p_b)/n_b}`.

    Parameters
    ----------
    successes_a, n_a : int, int
        Number of successes and trials in group A.
    successes_b, n_b : int, int
        Number of successes and trials in group B.
    delta : float
        Positive equivalence margin on the proportion difference.
    alpha : float, optional
        One-sided significance level (default ``0.05``).

    Returns
    -------
    TOSTResult
        The two one-sided p-values, their maximum, the equivalence verdict and the
        confidence interval for the proportion difference.
    """
    _validate(delta, alpha)
    if n_a < 1 or n_b < 1:
        raise ValueError("each group needs at least one trial")
    if not (0 <= successes_a <= n_a and 0 <= successes_b <= n_b):
        raise ValueError("require 0 <= successes <= trials")

    p_a = successes_a / n_a
    p_b = successes_b / n_b
    diff = p_a - p_b
    se = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    return _assemble(diff, se, delta, alpha, dist=stats.norm, dof=None)


def _assemble(
    diff: float,
    se: float,
    delta: float,
    alpha: float,
    *,
    dist: object,
    dof: float | None,
) -> TOSTResult:
    """Build a :class:`TOSTResult` from a difference, its SE and a reference dist.

    ``dist`` is a SciPy distribution (``stats.t`` or ``stats.norm``); ``dof`` supplies
    the degrees-of-freedom shape parameter for ``stats.t`` and is ``None`` for the
    normal approximation. Handles the degenerate ``se == 0`` case (a difference exactly
    inside or outside the margin) without dividing by zero.
    """
    args: tuple[float, ...] = () if dof is None else (dof,)
    if se == 0.0:
        # Degenerate: the difference is known exactly; equivalence is a hard bound.
        lower_p = 0.0 if diff > -delta else 1.0
        upper_p = 0.0 if diff < delta else 1.0
        half = 0.0
    else:
        t_lower = (diff + delta) / se
        t_upper = (diff - delta) / se
        lower_p = float(dist.sf(t_lower, *args))  # type: ignore[attr-defined]
        upper_p = float(dist.cdf(t_upper, *args))  # type: ignore[attr-defined]
        crit = float(dist.ppf(1 - alpha, *args))  # type: ignore[attr-defined]
        half = crit * se

    p_value = max(lower_p, upper_p)
    ci = (diff - half, diff + half)
    return TOSTResult(
        p_value=p_value,
        lower_p=lower_p,
        upper_p=upper_p,
        equivalent=p_value < alpha,
        ci=ci,
    )
