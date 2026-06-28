"""Tests for the spreading-activation retrieval arm.

The golden values are hand-computed from the Collins & Loftus (1975) / Anderson
(1983) spreading-activation recurrence (see the module docstring), never read back
from the implementation:

    a^(0)_i = s_i                                  (seed activation)
    a^(t)_i = decay * sum_{j: a^(t-1)_j >= theta} a^(t-1)_j * w_{ji}
    A_i     = sum_t a^(t)_i                         (accumulated total)

On a unit-weight chain from a single seed of strength s the node d hops away
accumulates exactly s * decay**d.
"""

from __future__ import annotations

import itertools
import math

import pytest

from membench.retrieval.base import ArmFamily, build_arm, get_spec
from membench.retrieval.graph.store import GraphEdge, GraphNode, RelationshipType, TypedGraph
from membench.retrieval.spreading_activation import (
    SpreadingActivationMemory,
    spread_activation,
)
from membench.types import MemoryItem


def _chain(node_ids: list[str], *, confidence: float = 1.0) -> TypedGraph:
    """Build a directed unit-weight chain ``node_ids[0] -> node_ids[1] -> ...``."""
    graph = TypedGraph()
    for node_id in node_ids:
        graph.add_node(GraphNode(node_id, f"text for {node_id}"))
    for source, target in itertools.pairwise(node_ids):
        graph.add_edge(
            GraphEdge(source, target, RelationshipType.RELATED_TO, confidence=confidence)
        )
    return graph


class TestSpreadActivationGolden:
    def test_chain_geometric_decay(self) -> None:
        # Chain S -> A -> B -> C, all unit-weight; seed S = 1.0, decay = 0.5,
        # threshold 0, 3 hops, spreading forward only ("out").
        #   pulse 0: S = 1.0
        #   pulse 1: A = 0.5 * 1.0 * 1.0          = 0.5
        #   pulse 2: B = 0.5 * 0.5 * 1.0          = 0.25
        #   pulse 3: C = 0.5 * 0.25 * 1.0         = 0.125
        # i.e. node d hops out accumulates 0.5**d.
        graph = _chain(["S", "A", "B", "C"])
        result = spread_activation(graph, {"S": 1.0}, decay=0.5, hops=3, direction="out")
        assert result == {"S": 1.0, "A": 0.5, "B": 0.25, "C": 0.125}
        # explicit geometric golden, independent of the dict above
        for distance, node_id in enumerate(["S", "A", "B", "C"]):
            assert math.isclose(result[node_id], 0.5**distance)

    def test_zero_decay_keeps_activation_on_seeds_only(self) -> None:
        # decay = 0 => no activation leaves the seed; only the seed is activated.
        graph = _chain(["S", "A", "B", "C"])
        result = spread_activation(graph, {"S": 1.0}, decay=0.0, hops=3, direction="out")
        assert result == {"S": 1.0}

    def test_firing_threshold_stops_propagation(self) -> None:
        # Chain S -> A -> B -> C -> D, seed 1.0, decay 0.5, threshold 0.2.
        # Activations carried: S=1.0, A=0.5, B=0.25, C=0.125. C (0.125) is below
        # the 0.2 threshold, so it does NOT fire and D is never reached.
        graph = _chain(["S", "A", "B", "C", "D"])
        result = spread_activation(
            graph, {"S": 1.0}, decay=0.5, hops=4, threshold=0.2, direction="out"
        )
        assert "D" not in result
        assert result == {"S": 1.0, "A": 0.5, "B": 0.25, "C": 0.125}

    def test_edge_confidence_weights_contribution(self) -> None:
        # Star S -> A (conf 1.0), S -> B (conf 0.5); seed 1.0, decay 0.5, 1 hop.
        #   A = 0.5 * 1.0 * 1.0 = 0.5
        #   B = 0.5 * 1.0 * 0.5 = 0.25
        graph = TypedGraph()
        for node_id in ["S", "A", "B"]:
            graph.add_node(GraphNode(node_id, node_id))
        graph.add_edge(GraphEdge("S", "A", RelationshipType.RELATED_TO, confidence=1.0))
        graph.add_edge(GraphEdge("S", "B", RelationshipType.RELATED_TO, confidence=0.5))
        result = spread_activation(graph, {"S": 1.0}, decay=0.5, hops=1, direction="out")
        assert result == {"S": 1.0, "A": 0.5, "B": 0.25}

    def test_decay_one_preserves_magnitude(self) -> None:
        # decay = 1, unit weights: activation passes undiminished one hop at a time.
        graph = _chain(["S", "A", "B"])
        result = spread_activation(graph, {"S": 2.0}, decay=1.0, hops=2, direction="out")
        assert result == {"S": 2.0, "A": 2.0, "B": 2.0}

    def test_seed_not_in_graph_is_ignored(self) -> None:
        graph = _chain(["S", "A"])
        result = spread_activation(graph, {"missing": 1.0}, decay=0.5, hops=2, direction="out")
        assert result == {}


class TestSpreadActivationInvariants:
    def test_hops_zero_returns_seeds(self) -> None:
        graph = _chain(["S", "A", "B"])
        assert spread_activation(graph, {"S": 1.0}, decay=0.5, hops=0) == {"S": 1.0}

    def test_invalid_decay_rejected(self) -> None:
        graph = _chain(["S", "A"])
        with pytest.raises(ValueError, match="decay"):
            spread_activation(graph, {"S": 1.0}, decay=1.5, hops=1)

    def test_invalid_hops_rejected(self) -> None:
        graph = _chain(["S", "A"])
        with pytest.raises(ValueError, match="hops"):
            spread_activation(graph, {"S": 1.0}, decay=0.5, hops=-1)

    def test_decay_monotone_in_distance(self) -> None:
        # Along a unit-weight chain, accumulated activation is non-increasing with
        # distance for any decay in [0, 1].
        graph = _chain(["S", "A", "B", "C"])
        result = spread_activation(graph, {"S": 1.0}, decay=0.7, hops=3, direction="out")
        scores = [result["S"], result["A"], result["B"], result["C"]]
        assert all(earlier >= later for earlier, later in itertools.pairwise(scores))


# The depth corpus from the graph-arm tests: a governing constraint D-1 whose text
# does not resemble the query, reachable only through a CONSTRAINS link from the
# query-similar justification D-1-rationale.
_DEPTH_CORPUS = [
    MemoryItem(
        "D-1",
        "primary buttons must use the shared Button component",
        metadata={"node_type": "constraint"},
    ),
    MemoryItem(
        "D-1-rationale",
        "the export dialog call to action should be obvious and accessible to users",
        metadata={"node_type": "justification", "constrains": "D-1"},
    ),
    MemoryItem("D-2", "tooltips use the Popover primitive"),
    MemoryItem("D-3", "dates use the DateRangePicker widget"),
]
_QUERY = "make the export dialog call to action accessible"


class TestArmRegistration:
    def test_registered_as_link_following_graph_arm(self) -> None:
        spec = get_spec("spreading_activation")
        assert spec.capabilities.family is ArmFamily.GRAPH
        assert spec.capabilities.follows_links is True
        assert spec.capabilities.uses_embeddings is True

    def test_constructible_with_no_args(self) -> None:
        # The harness calls build_arm(name) with no kwargs.
        arm = build_arm("spreading_activation")
        assert isinstance(arm, SpreadingActivationMemory)


class TestArmRetrieval:
    def test_recovers_governing_decision_through_link(self) -> None:
        arm = build_arm("spreading_activation", seeds=1, max_hops=2)
        arm.write(_DEPTH_CORPUS)  # type: ignore[attr-defined]
        context = arm.retrieve(_QUERY, budget_tokens=10_000)  # type: ignore[attr-defined]
        # The query-similar justification is the seed; the governing constraint is
        # reached through the CONSTRAINS hop and therefore activated.
        assert "D-1-rationale" in context.ids
        assert "D-1" in context.ids
        # The seed outranks the node reached one decayed hop away.
        ids = list(context.ids)
        assert ids.index("D-1-rationale") < ids.index("D-1")

    def test_zero_decay_returns_only_the_seed(self) -> None:
        arm = build_arm("spreading_activation", seeds=1, max_hops=3, decay=0.0)
        arm.write(_DEPTH_CORPUS)  # type: ignore[attr-defined]
        ids = arm.retrieve(_QUERY, budget_tokens=10_000).ids  # type: ignore[attr-defined]
        assert ids == ("D-1-rationale",)

    def test_empty_corpus_returns_empty(self) -> None:
        arm = SpreadingActivationMemory()
        assert arm.retrieve(_QUERY, budget_tokens=100).ids == ()

    def test_seeds_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="seeds"):
            SpreadingActivationMemory(seeds=0)
