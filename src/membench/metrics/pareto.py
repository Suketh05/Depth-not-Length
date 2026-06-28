r"""Accuracy-cost Pareto frontier and the 2D hypervolume indicator.

The headline trade-off of the benchmark is *accuracy up, dollars/tokens down*: an
arm is only interesting if no other arm beats it on both. This module extracts that
trade-off curve and gives it a single scalar quality score so two whole frontiers
(e.g. graph arms vs. lexical arms) can be compared.

Pareto dominance (maximise accuracy, minimise cost)
---------------------------------------------------
A point :math:`a` *dominates* :math:`b` iff it is no worse on either objective and
strictly better on at least one:

.. math::

    a \succ b \iff
        (a_{acc} \ge b_{acc}) \land (a_{cost} \le b_{cost})
        \land \big[(a_{acc} > b_{acc}) \lor (a_{cost} < b_{cost})\big].

The **Pareto frontier** (a.k.a. the skyline / non-dominated set) is the set of points
dominated by no other point; every remaining point is in the **dominated set**. See
Deb, K. (2001), *Multi-Objective Optimization Using Evolutionary Algorithms*, §2.4.

Hypervolume indicator
---------------------
The hypervolume (S-metric) is the area of the region weakly dominated by the frontier
and bounded by a reference point :math:`r` that is no better than every point
(:math:`r_{acc} \le \min acc`, :math:`r_{cost} \ge \max cost`). In two dimensions, with
the frontier sorted by descending accuracy (so costs descend too,
:math:`c_1 > c_2 > \dots > c_n`), it is the sum of disjoint rectangular slabs:

.. math::

    HV = (a_1 - r_{acc})\,(r_{cost} - c_1)
         + \sum_{i=2}^{n} (a_i - r_{acc})\,(c_{i-1} - c_i).

A larger hypervolume means a better frontier; it is monotone -- adding a
non-dominated point can only increase it (Zitzler & Thiele, 1999, "Multiobjective
evolutionary algorithms: a comparative case study and the strength Pareto approach",
*IEEE Trans. Evolutionary Computation* 3(4): 257-271). It is the standard unary
quality indicator for Pareto sets because it is strictly Pareto-compliant.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from membench.analysis.tables import ScoredRow

__all__ = [
    "ParetoPoint",
    "ReferencePoint",
    "dominated_set",
    "dominates",
    "from_scored_rows",
    "hypervolume",
    "pareto_frontier",
]


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    """One arm's position in accuracy-cost space.

    Attributes
    ----------
    arm : str
        The arm (or arm/model) label the point belongs to.
    accuracy : float
        A maximise-direction score (e.g. compliance rate); higher is better.
    cost : float
        A minimise-direction cost (e.g. dollars or tokens); lower is better.
    """

    arm: str
    accuracy: float
    cost: float


@dataclass(frozen=True, slots=True)
class ReferencePoint:
    """The hypervolume reference: a corner no point should beat.

    Attributes
    ----------
    accuracy : float
        Lower bound on accuracy; must be ``<=`` every frontier point's accuracy.
    cost : float
        Upper bound on cost; must be ``>=`` every frontier point's cost.
    """

    accuracy: float
    cost: float


def dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
    """Return whether ``a`` Pareto-dominates ``b`` (accuracy up, cost down).

    Parameters
    ----------
    a, b : ParetoPoint
        The candidate dominating point and the point being tested.

    Returns
    -------
    bool
        True iff ``a`` is no worse than ``b`` on both objectives and strictly
        better on at least one.
    """
    no_worse = a.accuracy >= b.accuracy and a.cost <= b.cost
    strictly_better = a.accuracy > b.accuracy or a.cost < b.cost
    return no_worse and strictly_better


def pareto_frontier(points: Sequence[ParetoPoint]) -> tuple[ParetoPoint, ...]:
    """Extract the non-dominated set (the accuracy-cost Pareto frontier).

    Parameters
    ----------
    points : Sequence[ParetoPoint]
        The candidate points. Order is preserved for the surviving points.

    Returns
    -------
    tuple[ParetoPoint, ...]
        Every point not dominated by any other point. Duplicate coordinates are
        all retained (no point strictly dominates an identical one).
    """
    return tuple(
        candidate
        for i, candidate in enumerate(points)
        if not any(dominates(other, candidate) for j, other in enumerate(points) if i != j)
    )


def dominated_set(points: Sequence[ParetoPoint]) -> tuple[ParetoPoint, ...]:
    """Return the complement of the frontier: points dominated by some other point.

    Parameters
    ----------
    points : Sequence[ParetoPoint]
        The candidate points.

    Returns
    -------
    tuple[ParetoPoint, ...]
        Every point dominated by at least one other point. Together with
        :func:`pareto_frontier` this exactly partitions ``points``.
    """
    return tuple(
        candidate
        for i, candidate in enumerate(points)
        if any(dominates(other, candidate) for j, other in enumerate(points) if i != j)
    )


def hypervolume(points: Sequence[ParetoPoint], reference: ReferencePoint) -> float:
    """Compute the 2D hypervolume indicator of the points' frontier.

    The frontier is extracted first, so passing the full point set or just its
    frontier yields the same value (dominated points contribute nothing).

    Parameters
    ----------
    points : Sequence[ParetoPoint]
        Points in accuracy-cost space.
    reference : ReferencePoint
        A corner no better than every point: ``reference.accuracy`` must be
        ``<=`` each accuracy and ``reference.cost`` ``>=`` each cost.

    Returns
    -------
    float
        The area weakly dominated by the frontier and bounded by ``reference``.
        ``0.0`` for an empty point set.

    Raises
    ------
    ValueError
        If any frontier point is worse than ``reference`` on either objective
        (which would contribute negative area).
    """
    frontier = pareto_frontier(points)
    if not frontier:
        return 0.0

    for point in frontier:
        if point.accuracy < reference.accuracy or point.cost > reference.cost:
            raise ValueError(
                "every point must weakly dominate the reference point "
                f"(got accuracy={point.accuracy}, cost={point.cost} vs "
                f"reference accuracy={reference.accuracy}, cost={reference.cost})"
            )

    # Unique coordinates, sorted by descending accuracy (hence descending cost on a
    # frontier); each slab spans down to the previous (cheaper-but-less-accurate) cost.
    coords = sorted({(p.accuracy, p.cost) for p in frontier}, key=lambda c: c[0], reverse=True)
    area = 0.0
    upper_cost = reference.cost
    for accuracy, cost in coords:
        area += (accuracy - reference.accuracy) * (upper_cost - cost)
        upper_cost = cost
    return area


def from_scored_rows(
    rows: Sequence[ScoredRow],
    *,
    accuracy: Callable[[ScoredRow], float] = lambda row: row.compliance_rate,
    cost: Callable[[ScoredRow], float] = lambda row: row.dollars,
) -> tuple[ParetoPoint, ...]:
    """Build accuracy-cost points from scored rows (default: compliance vs dollars).

    Parameters
    ----------
    rows : Sequence[ScoredRow]
        Canonical analysis rows (see :class:`membench.analysis.tables.ScoredRow`).
    accuracy : Callable[[ScoredRow], float], optional
        Extracts the maximise-direction score; defaults to ``compliance_rate``.
    cost : Callable[[ScoredRow], float], optional
        Extracts the minimise-direction cost; defaults to ``dollars``.

    Returns
    -------
    tuple[ParetoPoint, ...]
        One :class:`ParetoPoint` per row, labelled by the row's ``arm``.
    """
    return tuple(ParetoPoint(arm=row.arm, accuracy=accuracy(row), cost=cost(row)) for row in rows)
