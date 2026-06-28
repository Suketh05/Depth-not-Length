"""Tests for the accuracy-cost Pareto frontier and hypervolume indicator.

Golden fixture (4 points, accuracy up / cost down):

    A = (accuracy=0.9, cost=10)
    B = (accuracy=0.7, cost=5)
    C = (accuracy=0.6, cost=8)   dominated by B (0.7 >= 0.6 and 5 <= 8)
    D = (accuracy=0.5, cost=12)  dominated by A and by B

The frontier is therefore exactly {A, B} (size 2); C and D are dominated.

Hand-computed hypervolume w.r.t. reference r = (accuracy=0.0, cost=20):
the frontier dominates the union of two rectangles (in accuracy x cost space):
    A: accuracy in [0, 0.9], cost in [10, 20]
    B: accuracy in [0, 0.7], cost in [5, 20]
Slicing by accuracy:
    accuracy in [0.0, 0.7]: both present -> cost union [5, 20], height 15 -> 0.7 * 15 = 10.5
    accuracy in [0.7, 0.9]: only A       -> cost      [10, 20], height 10 -> 0.2 * 10 =  2.0
    HV = 10.5 + 2.0 = 12.5
"""

from __future__ import annotations

import pytest

from membench.analysis.tables import ScoredRow
from membench.metrics.pareto import (
    ParetoPoint,
    ReferencePoint,
    dominated_set,
    dominates,
    from_scored_rows,
    hypervolume,
    pareto_frontier,
)

A = ParetoPoint(arm="a", accuracy=0.9, cost=10.0)
B = ParetoPoint(arm="b", accuracy=0.7, cost=5.0)
C = ParetoPoint(arm="c", accuracy=0.6, cost=8.0)
D = ParetoPoint(arm="d", accuracy=0.5, cost=12.0)
POINTS = [A, B, C, D]
REF = ReferencePoint(accuracy=0.0, cost=20.0)

# Hand-computed golden hypervolume of the {A, B} frontier w.r.t. REF (see module docstring).
GOLDEN_HYPERVOLUME = 12.5


class TestDominates:
    def test_strictly_better_on_both(self) -> None:
        assert dominates(A, D)  # higher accuracy AND lower cost

    def test_better_on_one_equal_on_other(self) -> None:
        # B dominates C: equal-or-better accuracy (0.7 > 0.6) and cheaper (5 < 8).
        assert dominates(B, C)

    def test_identical_points_do_not_dominate(self) -> None:
        twin = ParetoPoint(arm="a2", accuracy=0.9, cost=10.0)
        assert not dominates(A, twin)
        assert not dominates(twin, A)

    def test_trade_off_neither_dominates(self) -> None:
        # A is more accurate but costlier than B -> incomparable.
        assert not dominates(A, B)
        assert not dominates(B, A)


class TestFrontier:
    def test_membership_is_exactly_a_and_b(self) -> None:
        frontier = pareto_frontier(POINTS)
        assert {p.arm for p in frontier} == {"a", "b"}
        assert len(frontier) == 2  # golden frontier size

    def test_dominated_set_is_c_and_d(self) -> None:
        assert {p.arm for p in dominated_set(POINTS)} == {"c", "d"}

    def test_frontier_and_dominated_partition_input(self) -> None:
        frontier = pareto_frontier(POINTS)
        dominated = dominated_set(POINTS)
        assert len(frontier) + len(dominated) == len(POINTS)
        assert set(frontier).isdisjoint(dominated)
        assert set(frontier) | set(dominated) == set(POINTS)

    def test_dominated_points_excluded(self) -> None:
        frontier = pareto_frontier(POINTS)
        assert C not in frontier
        assert D not in frontier

    def test_identical_coordinates_both_retained(self) -> None:
        twin = ParetoPoint(arm="a2", accuracy=0.9, cost=10.0)
        frontier = pareto_frontier([A, twin])
        assert len(frontier) == 2

    def test_empty_input(self) -> None:
        assert pareto_frontier([]) == ()
        assert dominated_set([]) == ()


class TestHypervolume:
    def test_matches_hand_computed_golden(self) -> None:
        assert hypervolume(POINTS, REF) == pytest.approx(GOLDEN_HYPERVOLUME)

    def test_invariant_to_dominated_points(self) -> None:
        # Dropping the dominated points must not change the indicator.
        assert hypervolume([A, B], REF) == pytest.approx(hypervolume(POINTS, REF))

    def test_single_point_is_a_rectangle(self) -> None:
        # One point -> area = (acc - ref_acc) * (ref_cost - cost) = 0.9 * 10 = 9.0.
        assert hypervolume([A], REF) == pytest.approx(9.0)

    def test_empty_set_is_zero(self) -> None:
        assert hypervolume([], REF) == 0.0

    def test_monotone_under_adding_non_dominated_point(self) -> None:
        before = hypervolume([A, B], REF)
        # E is non-dominated by A and B (most accurate, mid cost) -> must raise HV.
        e = ParetoPoint(arm="e", accuracy=0.95, cost=15.0)
        after = hypervolume([A, B, e], REF)
        assert after > before

    def test_adding_dominated_point_leaves_hv_unchanged(self) -> None:
        before = hypervolume([A, B], REF)
        after = hypervolume([A, B, C, D], REF)
        assert after == pytest.approx(before)

    def test_rejects_point_worse_than_reference(self) -> None:
        bad = ParetoPoint(arm="bad", accuracy=0.5, cost=25.0)  # cost exceeds ref cost
        with pytest.raises(ValueError, match="weakly dominate"):
            hypervolume([bad], REF)


class TestFromScoredRows:
    def _row(self, arm: str, compliance: float, dollars: float) -> ScoredRow:
        return ScoredRow(
            dataset="synthetic",
            arm=arm,
            model="claude",
            depth=1,
            spec_variant="stripped",
            compliance_rate=compliance,
            chain_recovered=True,
            recall=1.0,
            precision=1.0,
            correct=True,
            total_tokens=100,
            dollars=dollars,
        )

    def test_default_accessors_use_compliance_and_dollars(self) -> None:
        rows = [self._row("a", 0.9, 10.0), self._row("b", 0.7, 5.0)]
        points = from_scored_rows(rows)
        assert points == (
            ParetoPoint(arm="a", accuracy=0.9, cost=10.0),
            ParetoPoint(arm="b", accuracy=0.7, cost=5.0),
        )

    def test_frontier_over_scored_rows(self) -> None:
        rows = [
            self._row("a", 0.9, 10.0),
            self._row("b", 0.7, 5.0),
            self._row("c", 0.6, 8.0),
        ]
        frontier = pareto_frontier(from_scored_rows(rows))
        assert {p.arm for p in frontier} == {"a", "b"}

    def test_custom_accessors(self) -> None:
        rows = [self._row("a", 0.9, 10.0)]
        points = from_scored_rows(
            rows,
            accuracy=lambda r: r.recall,
            cost=lambda r: float(r.total_tokens),
        )
        assert points[0] == ParetoPoint(arm="a", accuracy=1.0, cost=100.0)
