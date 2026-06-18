"""Tests for Pareto frontier and stochastic dominance."""

from __future__ import annotations

import pytest

from membench.stats.dominance import (
    dominates,
    first_order_dominates,
    pareto_frontier,
    second_order_dominates,
)


class TestPareto:
    def test_dominates(self) -> None:
        assert dominates((0.9, 100.0), (0.8, 150.0))  # higher acc, lower cost
        assert not dominates((0.9, 100.0), (0.9, 100.0))  # equal -> no strict win
        assert not dominates((0.9, 200.0), (0.8, 100.0))  # better acc but worse cost

    def test_frontier(self) -> None:
        # (acc, cost): Brief best corner, plus two trade-off points and one dominated
        points = [(0.95, 90.0), (0.80, 80.0), (0.70, 200.0), (0.90, 95.0)]
        front = set(pareto_frontier(points))
        assert 0 in front and 1 in front  # both non-dominated
        assert 2 not in front  # dominated by everything

    def test_empty(self) -> None:
        assert pareto_frontier([]) == []


class TestStochasticDominance:
    def test_first_order(self) -> None:
        a = [3, 4, 5, 6]
        b = [1, 2, 3, 4]
        assert first_order_dominates(a, b)
        assert not first_order_dominates(b, a)

    def test_first_order_implies_second_order(self) -> None:
        a = [3, 4, 5, 6]
        b = [1, 2, 3, 4]
        assert second_order_dominates(a, b)

    def test_second_order_without_first(self) -> None:
        # a has higher mean but crossing CDFs -> SOSD may hold while FOSD does not
        a = [2, 2, 2, 2]
        b = [0, 0, 4, 4]
        assert second_order_dominates(a, b)
        assert not first_order_dominates(a, b)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            first_order_dominates([], [1.0])
