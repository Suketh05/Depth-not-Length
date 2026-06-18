r"""Multiple-comparison control and the Demšar protocol for many systems.

Comparing 20+ arms over several datasets means many simultaneous tests, so p-values
must be corrected and ranks compared properly:

* :func:`holm_adjust` -- Holm-Bonferroni step-down (family-wise error control).
* :func:`benjamini_hochberg_adjust` -- Benjamini-Hochberg step-up (false-discovery-rate
  control), the right choice when many comparisons are screened.
* :func:`friedman_test` -- the non-parametric omnibus test for whether the arms differ
  at all across datasets/blocks.
* :func:`average_ranks` + :func:`nemenyi_critical_difference` -- the Demšar (2006)
  post-hoc: average each arm's rank across datasets, and declare two arms different
  when their mean ranks differ by more than the critical difference. This is exactly
  what a critical-difference diagram draws.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

__all__ = [
    "average_ranks",
    "benjamini_hochberg_adjust",
    "friedman_test",
    "holm_adjust",
    "nemenyi_critical_difference",
]

# Nemenyi critical values q_{0.05} (studentized range / sqrt(2)), Demšar 2006 Table 5.
_NEMENYI_Q05: dict[int, float] = {
    2: 1.960,
    3: 2.344,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
    11: 3.219,
    12: 3.268,
    13: 3.313,
    14: 3.354,
    15: 3.391,
    16: 3.426,
    17: 3.458,
    18: 3.489,
    19: 3.517,
    20: 3.544,
}


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Return Holm-Bonferroni adjusted p-values (family-wise error control)."""
    p = np.asarray(pvalues, dtype=np.float64)
    n = p.size
    if n == 0:
        raise ValueError("need at least one p-value")
    order = np.argsort(p)
    adjusted = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return [float(x) for x in adjusted]


def benjamini_hochberg_adjust(pvalues: list[float]) -> list[float]:
    """Return Benjamini-Hochberg adjusted p-values (false-discovery-rate control)."""
    p = np.asarray(pvalues, dtype=np.float64)
    n = p.size
    if n == 0:
        raise ValueError("need at least one p-value")
    order = np.argsort(p)
    adjusted = np.empty(n)
    running = 1.0
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        running = min(running, p[idx] * n / (rank + 1))
        adjusted[idx] = running
    return [float(x) for x in adjusted]


@dataclass(frozen=True, slots=True)
class FriedmanResult:
    """Friedman omnibus statistic and p-value."""

    statistic: float
    p_value: float


def friedman_test(scores: NDArray[np.float64]) -> FriedmanResult:
    """Run the Friedman omnibus test on a (blocks x treatments) score matrix.

    Rows are blocks (datasets/tasks), columns are arms. Tests whether the arms'
    distributions differ across blocks.
    """
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 3:
        raise ValueError("need a 2-D matrix with >= 3 treatments (columns)")
    result = stats.friedmanchisquare(*[matrix[:, j] for j in range(matrix.shape[1])])
    return FriedmanResult(float(result.statistic), float(result.pvalue))


def average_ranks(
    scores: NDArray[np.float64], *, higher_is_better: bool = True
) -> NDArray[np.float64]:
    """Return each arm's mean rank across blocks (rank 1 = best)."""
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("scores must be 2-D (blocks x treatments)")
    signed = -matrix if higher_is_better else matrix
    ranks = np.array([stats.rankdata(row) for row in signed], dtype=np.float64)
    return np.asarray(ranks.mean(axis=0), dtype=np.float64)


def nemenyi_critical_difference(k: int, n_blocks: int, *, alpha: float = 0.05) -> float:
    r"""Return the Nemenyi critical difference for ``k`` arms over ``n_blocks`` datasets.

    :math:`CD = q_\alpha \sqrt{k(k+1)/(6 N)}`. Mean-rank differences larger than CD are
    significant at level ``alpha`` (only ``alpha=0.05`` is tabulated here).
    """
    if alpha != 0.05:
        raise ValueError("only alpha=0.05 is tabulated")
    if k not in _NEMENYI_Q05:
        raise ValueError(f"k={k} outside the tabulated range 2..20")
    if n_blocks < 1:
        raise ValueError("n_blocks must be >= 1")
    q = _NEMENYI_Q05[k]
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n_blocks)))
