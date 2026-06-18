r"""Paired significance tests for arm-vs-arm comparisons on the same tasks.

Because every arm is run on the *same* tasks, comparisons are paired, and paired
tests are both more powerful and more correct than unpaired ones here:

* :func:`paired_permutation_test` -- the exact sign-flip randomization test on paired
  differences, the standard significance test in information-retrieval evaluation
  (it makes no distributional assumption; under the null the sign of each task's
  difference is exchangeable).
* :func:`wilcoxon_signed_rank` -- the non-parametric paired rank test (via SciPy).
* :func:`mcnemar_test` -- the exact test for paired *binary* outcomes (e.g. each arm
  correct/incorrect per task), based only on the discordant pairs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = ["TestResult", "mcnemar_test", "paired_permutation_test", "wilcoxon_signed_rank"]


@dataclass(frozen=True, slots=True)
class TestResult:
    """A test statistic, its p-value, and the method name."""

    statistic: float
    p_value: float
    method: str


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> TestResult:
    r"""Exact paired sign-flip permutation test (two-sided) on ``a - b``.

    The statistic is the mean paired difference; the null distribution flips the sign
    of each task's difference independently. Returns the two-sided p-value
    :math:`(1 + \#\{|t^*| \ge |t_\text{obs}|\}) / (1 + R)`.
    """
    da = np.asarray(a, dtype=np.float64)
    db = np.asarray(b, dtype=np.float64)
    if da.shape != db.shape:
        raise ValueError("a and b must be paired (equal length)")
    if da.size == 0:
        raise ValueError("need at least one pair")
    diff = da - db
    observed = float(np.mean(diff))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, diff.size))
    perm_means = (signs * np.abs(diff)).mean(axis=1)
    extreme = int(np.sum(np.abs(perm_means) >= abs(observed) - 1e-12))
    p = (1 + extreme) / (1 + n_resamples)
    return TestResult(statistic=observed, p_value=float(p), method="paired_permutation")


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> TestResult:
    """Wilcoxon signed-rank test on paired samples (via SciPy)."""
    da = np.asarray(a, dtype=np.float64)
    db = np.asarray(b, dtype=np.float64)
    if da.shape != db.shape:
        raise ValueError("a and b must be paired (equal length)")
    if np.all(da == db):
        return TestResult(statistic=0.0, p_value=1.0, method="wilcoxon")
    result = stats.wilcoxon(da, db)
    return TestResult(
        statistic=float(result.statistic), p_value=float(result.pvalue), method="wilcoxon"
    )


def mcnemar_test(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> TestResult:
    """Exact McNemar test for paired binary outcomes (per-task correct/incorrect).

    Uses only the discordant pairs (one arm right, the other wrong); the statistic
    reported is the discordant count where ``a`` wins minus where ``b`` wins, and the
    p-value is the exact two-sided binomial test on the discordant pairs.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("correct_a and correct_b must be paired (equal length)")
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    discordant = a_only + b_only
    if discordant == 0:
        return TestResult(statistic=0.0, p_value=1.0, method="mcnemar_exact")
    p = stats.binomtest(min(a_only, b_only), discordant, 0.5).pvalue
    return TestResult(statistic=float(a_only - b_only), p_value=float(p), method="mcnemar_exact")
