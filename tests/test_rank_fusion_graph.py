"""Tests for the RRF similarity+graph-hop fusion arm.

The first test pins the Reciprocal Rank Fusion math to a hand-computed golden
(derived independently of the implementation). The remaining tests assert the
structural promotion the arm exists to deliver: a deep, link-reachable node that
similarity ranks last is lifted by the graph-hop ranking.
"""

from __future__ import annotations

import pytest

from membench.retrieval.base import build_arm
from membench.retrieval.dense import DenseMemory
from membench.retrieval.rank_fusion_graph import (
    RankFusionGraphMemory,
    reciprocal_rank_fusion,
)
from membench.types import MemoryItem


def test_rrf_golden_scores_and_order() -> None:
    """Hand-computed RRF golden for two rank lists with k=60 (1-based ranks).

    similarity ranking A = [a, b, c]   (ranks 1, 2, 3)
    graph-hop  ranking B = [c, b]      (ranks 1, 2;  'a' is absent from B)

    RRF(d) = sum_r 1 / (k + rank_r(d)), k = 60:
      a = 1/61                  = 0.016393442622950820
      b = 1/62 + 1/62 = 2/62    = 0.032258064516129032   (== 124/3844)
      c = 1/63 + 1/61           = 0.032266458496487550   (== 124/3843)

    Exact comparison: c = 124/3843 > b = 124/3844 (same numerator, smaller
    denominator) > a = 63/3843. So the fused order is [c, b, a]: 'c', last on
    similarity, is promoted to the top because it also tops the graph ranking.
    """
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["c", "b"]], k=60)

    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(2 / 62)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)

    fused_order = sorted(scores, key=lambda d: -scores[d])
    assert fused_order == ["c", "b", "a"]
    # c edges out b by exactly 124/3843 vs 124/3844.
    assert scores["c"] > scores["b"] > scores["a"]


def test_rrf_rewards_agreement_between_rankings() -> None:
    """A doc present in both rankings beats a doc present in only one at rank 1."""
    scores = reciprocal_rank_fusion([["x", "y"], ["x"]], k=60)
    # x: 1/61 + 1/61 ; y: 1/62 (only the first ranking)
    assert scores["x"] == pytest.approx(2 / 61)
    assert scores["y"] == pytest.approx(1 / 62)
    assert scores["x"] > scores["y"]


def test_rrf_rejects_non_positive_k() -> None:
    """k must be positive (matches the arm's own validation)."""
    with pytest.raises(ValueError, match="k must be positive"):
        reciprocal_rank_fusion([["a"]], k=0)


def _depth_chain_corpus() -> list[MemoryItem]:
    """A corpus where the governing node is deep behind a query-similar seed.

    ``entry`` is near-identical to the query (top similarity, the sole seed);
    ``entry -> mid -> deep`` is a typed-edge chain; ``deep`` shares no query
    vocabulary so similarity ranks it last; the distractors share query words and
    so outrank ``deep`` on similarity but are graph-isolated.
    """
    edge = lambda target: {"edges": [(target, "derived_from")]}  # noqa: E731
    return [
        MemoryItem(
            "entry", "deploy the payment service to the production cluster now", edge("mid")
        ),
        MemoryItem("mid", "internal rationale node alpha bridging the records", edge("deep")),
        MemoryItem("deep", "every refund must reference ledger snapshot zeta omega"),
        MemoryItem("dist1", "production cluster autoscaling and load balancing tuning"),
        MemoryItem("dist2", "the payment service exposes a public deploy webhook today"),
        MemoryItem("dist3", "cluster region failover for the production service tier"),
    ]


_QUERY = "deploy the payment service to the production cluster"


def test_graph_promotes_deep_link_reachable_node() -> None:
    """The deep node, last on similarity, is promoted above every distractor."""
    corpus = _depth_chain_corpus()

    # Pure similarity baseline: 'deep' shares no query words -> ranks at the bottom.
    dense = DenseMemory()
    dense.write(corpus)
    dense_order = list(dense.retrieve(_QUERY, 1_000_000_000).ids)
    distractors = {"dist1", "dist2", "dist3"}
    assert dense_order.index("deep") > max(dense_order.index(d) for d in distractors)

    # Fusion arm: seeds=1 so only 'entry' seeds the walk, giving a clean chain
    # graph ranking [entry, mid, deep]; 'deep' inherits a strong graph term.
    arm = RankFusionGraphMemory(seeds=1)
    arm.write(corpus)
    fused_order = list(arm.retrieve(_QUERY, 1_000_000_000).ids)

    # Promotion: deep moves up relative to the similarity-only ranking ...
    assert fused_order.index("deep") < dense_order.index("deep")
    # ... and now outranks every (graph-isolated) distractor.
    assert all(fused_order.index("deep") < fused_order.index(d) for d in distractors)
    # entry tops both signals, so it stays first.
    assert fused_order[0] == "entry"


def test_graph_fusion_retrieves_deep_node_under_tight_budget() -> None:
    """Under a 3-item budget, fusion includes the deep node but similarity does not."""
    corpus = _depth_chain_corpus()
    text_by_id = {item.item_id: item.text for item in corpus}
    chars_per_token = 4
    # Budget derived from the corpus texts (not from any ranking): exactly enough
    # for the three chain nodes entry+mid+deep.
    budget = sum(len(text_by_id[i]) // chars_per_token for i in ("entry", "mid", "deep"))

    dense = DenseMemory()
    dense.write(corpus)
    dense_ids = set(dense.retrieve(_QUERY, budget).ids)
    assert "deep" not in dense_ids  # crowded out by query-similar distractors

    arm = RankFusionGraphMemory(seeds=1)
    arm.write(corpus)
    fused_ids = set(arm.retrieve(_QUERY, budget).ids)
    assert "deep" in fused_ids  # graph term lifts it into the budget


def test_arm_constructs_with_no_required_args() -> None:
    """The harness builds arms via build_arm(name) with no kwargs."""
    arm = build_arm("rank_fusion_graph")
    assert isinstance(arm, RankFusionGraphMemory)
    # No-corpus retrieve is well-defined (empty context).
    assert arm.retrieve("anything", 100).ids == ()
