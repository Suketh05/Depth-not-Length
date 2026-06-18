"""Tests for cost-to-correct and Return-on-Tokens metrics."""

from __future__ import annotations

import math

import pytest

from membench.metrics.cost import (
    AttemptCost,
    cost_per_correct,
    return_on_tokens,
    score_cost_to_correct,
    useful_signal,
)


def _a(correct: bool, inp: int = 100, out: int = 20, dollars: float = 0.01) -> AttemptCost:
    return AttemptCost(input_tokens=inp, output_tokens=out, dollars=dollars, correct=correct)


class TestCostToCorrect:
    def test_stops_at_first_correct(self) -> None:
        c = score_cost_to_correct([_a(False), _a(True), _a(True)])
        assert c.resolved and c.iterations_to_correct == 2
        assert c.total_tokens == 2 * 120  # only the first two attempts counted
        assert c.total_dollars == pytest.approx(0.02)

    def test_never_correct_counts_all(self) -> None:
        c = score_cost_to_correct([_a(False), _a(False)])
        assert not c.resolved and c.iterations_to_correct is None
        assert c.attempts == 2 and c.total_tokens == 2 * 120

    def test_single_correct(self) -> None:
        c = score_cost_to_correct([_a(True)])
        assert c.resolved and c.iterations_to_correct == 1

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            score_cost_to_correct([])


class TestEconomics:
    def test_useful_signal(self) -> None:
        assert useful_signal(100.0, 0.5) == pytest.approx(50.0)

    def test_return_on_tokens(self) -> None:
        # 1 correct output per 2000 tokens -> 0.5 per 1k
        assert return_on_tokens(1.0, 2000) == pytest.approx(0.5)

    def test_return_on_tokens_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            return_on_tokens(1.0, 0)

    def test_cost_per_correct(self) -> None:
        assert cost_per_correct(2.0, 4.0) == pytest.approx(0.5)
        assert math.isinf(cost_per_correct(2.0, 0))
