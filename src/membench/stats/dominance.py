r"""Pareto frontier and stochastic dominance for the joint accuracy/cost win.

The strongest form of the claim is joint: structured memory is more accurate *and*
cheaper. Two tools test that:

* :func:`pareto_frontier` -- over (accuracy, cost) points, the non-dominated set
  (higher accuracy, lower cost). An arm that sits alone on the frontier's
  high-accuracy/low-cost corner wins outright.
* :func:`first_order_dominates` / :func:`second_order_dominates` -- whether one arm's
  accuracy *distribution* is stochastically larger than another's (a stronger
  statement than comparing means): first-order if its CDF is everywhere below, and
  the weaker second-order (risk-averse) variant on the integrated CDF.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "dominates",
    "first_order_dominates",
    "pareto_frontier",
    "second_order_dominates",
]


def dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Return whether ``a`` dominates ``b`` on (accuracy up, cost down).

    ``a = (accuracy, cost)`` dominates ``b`` if it is no worse on both axes and
    strictly better on at least one.
    """
    acc_a, cost_a = a
    acc_b, cost_b = b
    no_worse = acc_a >= acc_b and cost_a <= cost_b
    strictly_better = acc_a > acc_b or cost_a < cost_b
    return no_worse and strictly_better


def pareto_frontier(points: Sequence[tuple[float, float]]) -> list[int]:
    """Return the indices of the non-dominated (accuracy up, cost down) points."""
    if not points:
        return []
    frontier: list[int] = []
    for i, p in enumerate(points):
        if not any(j != i and dominates(q, p) for j, q in enumerate(points)):
            frontier.append(i)
    return frontier


def _ecdf_on_grid(sample: NDArray[np.float64], grid: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.searchsorted(np.sort(sample), grid, side="right") / sample.size


def first_order_dominates(a: Sequence[float], b: Sequence[float], *, tol: float = 1e-9) -> bool:
    r"""Return whether ``a`` first-order stochastically dominates ``b`` (larger values).

    True when :math:`F_a(x) \le F_b(x)` for all ``x`` (a's mass sits at higher values).
    """
    sa = np.asarray(a, dtype=np.float64)
    sb = np.asarray(b, dtype=np.float64)
    if sa.size == 0 or sb.size == 0:
        raise ValueError("both samples must be non-empty")
    grid = np.unique(np.concatenate([sa, sb]))
    fa = _ecdf_on_grid(sa, grid)
    fb = _ecdf_on_grid(sb, grid)
    return bool(np.all(fa <= fb + tol))


def second_order_dominates(a: Sequence[float], b: Sequence[float], *, tol: float = 1e-9) -> bool:
    r"""Return whether ``a`` second-order stochastically dominates ``b`` (larger values).

    True when the integral of :math:`F_a` is everywhere :math:`\le` that of
    :math:`F_b` -- the weaker, risk-averse criterion implied by (but not implying)
    first-order dominance.
    """
    sa = np.asarray(a, dtype=np.float64)
    sb = np.asarray(b, dtype=np.float64)
    if sa.size == 0 or sb.size == 0:
        raise ValueError("both samples must be non-empty")
    grid = np.unique(np.concatenate([sa, sb]))
    fa = _ecdf_on_grid(sa, grid)
    fb = _ecdf_on_grid(sb, grid)
    # Integral of a right-continuous CDF: over [grid[j], grid[j+1]) the CDF equals
    # F[j], so accumulate F[:-1] * gap. Prepend 0 for the left endpoint.
    gaps = np.diff(grid)
    int_a = np.concatenate([[0.0], np.cumsum(fa[:-1] * gaps)])
    int_b = np.concatenate([[0.0], np.cumsum(fb[:-1] * gaps)])
    return bool(np.all(int_a <= int_b + tol))
