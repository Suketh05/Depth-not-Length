"""Tests for retrieval-quality metrics."""

from __future__ import annotations

import math

import pytest

from membench.metrics.retrieval import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_retrieval,
)


class TestPrecisionRecall:
    def test_precision_at_k(self) -> None:
        # top-2 of [g, x, g] are [g, x] -> 1/2
        assert precision_at_k(["g1", "x", "g2"], {"g1", "g2"}, k=2) == pytest.approx(0.5)

    def test_recall_at_k(self) -> None:
        assert recall_at_k(["g1", "x", "g2"], {"g1", "g2"}, k=2) == pytest.approx(0.5)

    def test_recall_vacuous_on_no_gold(self) -> None:
        assert recall_at_k(["x"], set()) == 1.0

    def test_precision_empty_retrieval(self) -> None:
        assert precision_at_k([], {"g"}) == 0.0
        assert precision_at_k([], set()) == 1.0


class TestRankMetrics:
    def test_reciprocal_rank(self) -> None:
        assert reciprocal_rank(["x", "g", "y"], {"g"}) == pytest.approx(0.5)
        assert reciprocal_rank(["x", "y"], {"g"}) == 0.0

    def test_average_precision(self) -> None:
        # gold {g1,g2}; retrieved [g1, x, g2]: hits at ranks 1 (p=1) and 3 (p=2/3)
        ap = average_precision(["g1", "x", "g2"], {"g1", "g2"})
        assert ap == pytest.approx((1.0 + 2 / 3) / 2)

    def test_ndcg_perfect_when_gold_first(self) -> None:
        assert ndcg_at_k(["g1", "g2", "x"], {"g1", "g2"}) == pytest.approx(1.0)

    def test_ndcg_discounts_late_hits(self) -> None:
        # one gold at rank 2: dcg = 1/log2(3); idcg (1 gold) = 1/log2(2)=1
        val = ndcg_at_k(["x", "g1"], {"g1"})
        assert val == pytest.approx(1.0 / math.log2(3))


class TestScoreRetrieval:
    def test_chain_recovered_requires_all_gold(self) -> None:
        full = score_retrieval(["g1", "g2", "x"], ["g1", "g2"])
        assert full.chain_recovered and full.recall == 1.0
        partial = score_retrieval(["g1", "x"], ["g1", "g2"])
        assert not partial.chain_recovered and partial.recall == 0.5

    def test_dedups_retrieved(self) -> None:
        s = score_retrieval(["g1", "g1", "x"], ["g1"])
        assert s.retrieved == 2  # de-duplicated
        assert s.true_positives == ("g1",)

    def test_no_gold_is_vacuous(self) -> None:
        s = score_retrieval(["x"], [])
        assert s.recall == 1.0 and not s.chain_recovered  # no chain to recover
