r"""Statistical power and minimum-detectable-effect for the benchmark's comparisons.

A benchmark should know whether it can actually detect the effects it reports. For
the two-proportion comparisons here (e.g. structured vs. similarity compliance
rate), this module gives the power of a two-sided z-test at a sample size, the per
-group sample size required to reach a target power, and the minimum detectable
effect at a given size -- so the paper can state "with N tasks per arm we have 80%
power to detect a 12-point compliance gap" rather than hand-wave the sample size.
All use the normal approximation with a pooled null variance.
"""

from __future__ import annotations

import math

from scipy import stats

__all__ = [
    "minimum_detectable_effect",
    "required_sample_size",
    "two_proportion_power",
]


def two_proportion_power(p1: float, p2: float, n: int, *, alpha: float = 0.05) -> float:
    """Return the power of a two-sided two-proportion z-test at ``n`` per group."""
    for p in (p1, p2):
        if not 0.0 <= p <= 1.0:
            raise ValueError("proportions must be in [0, 1]")
    if n < 1:
        raise ValueError("n must be >= 1")
    if p1 == p2:
        return alpha
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    pbar = (p1 + p2) / 2
    se0 = math.sqrt(2 * pbar * (1 - pbar) / n)
    se1 = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    if se1 == 0:
        return 1.0
    return float(stats.norm.cdf((abs(p1 - p2) - z_alpha * se0) / se1))


def required_sample_size(p1: float, p2: float, *, power: float = 0.8, alpha: float = 0.05) -> int:
    """Return the per-group sample size to reach ``power`` for detecting p1 vs p2."""
    if p1 == p2:
        raise ValueError("p1 and p2 must differ")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    numerator = z_alpha * math.sqrt(2 * pbar * (1 - pbar)) + z_beta * math.sqrt(
        p1 * (1 - p1) + p2 * (1 - p2)
    )
    value = float((numerator / abs(p1 - p2)) ** 2)
    return math.ceil(value)


def minimum_detectable_effect(
    p_baseline: float, n: int, *, power: float = 0.8, alpha: float = 0.05
) -> float:
    """Return the smallest absolute rate difference detectable at ``n`` per group.

    Uses the baseline rate for the variance (a conservative approximation when the
    alternative rate is unknown).
    """
    if not 0.0 <= p_baseline <= 1.0:
        raise ValueError("p_baseline must be in [0, 1]")
    if n < 1:
        raise ValueError("n must be >= 1")
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return float((z_alpha + z_beta) * math.sqrt(2 * p_baseline * (1 - p_baseline) / n))
