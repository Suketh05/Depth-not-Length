"""Tests for bounded multi-hop BFS traversal."""

from __future__ import annotations

import pytest

from membench.retrieval.graph.store import GraphEdge, GraphNode, RelationshipType, TypedGraph
from membench.retrieval.graph.traversal import MAX_HOPS_LIMIT, traverse


def _chain(length: int) -> TypedGraph:
    """A directed chain n0 -> n1 -> ... -> n{length-1} via CONSTRAINS edges."""
    g = TypedGraph()
    for i in range(length):
        g.add_node(GraphNode(f"n{i}", f"text {i}"))
    for i in range(length - 1):
        g.add_edge(GraphEdge(f"n{i}", f"n{i + 1}", RelationshipType.CONSTRAINS, confidence=0.9))
    return g


class TestTraverse:
    def test_two_hops_recovers_chain(self) -> None:
        result = traverse(_chain(5), ["n0"], max_hops=2, direction="out")
        assert result.node_ids == ["n0", "n1", "n2"]
        assert result.max_depth_reached == 2

    def test_one_hop_limits_reach(self) -> None:
        result = traverse(_chain(5), ["n0"], max_hops=1, direction="out")
        assert result.node_ids == ["n0", "n1"]

    def test_hops_clamped_to_limit(self) -> None:
        # asking for 10 hops is clamped to MAX_HOPS_LIMIT (3)
        result = traverse(_chain(10), ["n0"], max_hops=10, direction="out")
        assert result.max_depth_reached == MAX_HOPS_LIMIT

    def test_node_budget_truncates(self) -> None:
        result = traverse(_chain(10), ["n0"], max_hops=3, max_nodes=2, direction="out")
        assert len(result.node_ids) == 2
        assert result.truncated

    def test_cycle_is_safe(self) -> None:
        g = _chain(3)
        g.add_edge(GraphEdge("n2", "n0", RelationshipType.RELATED_TO))  # close the loop
        result = traverse(g, ["n0"], max_hops=3, direction="out")
        assert sorted(result.node_ids) == ["n0", "n1", "n2"]  # each visited once

    def test_missing_seed_skipped(self) -> None:
        result = traverse(_chain(3), ["nope", "n0"], direction="out")
        assert result.node_ids[0] == "n0"

    def test_relationship_filter(self) -> None:
        g = _chain(3)
        g.add_node(GraphNode("x", "unrelated"))
        g.add_edge(GraphEdge("n0", "x", RelationshipType.RELATED_TO))
        result = traverse(
            g,
            ["n0"],
            max_hops=1,
            direction="out",
            relationship_types=frozenset({RelationshipType.CONSTRAINS}),
        )
        assert "x" not in result.node_ids and "n1" in result.node_ids

    def test_track_traversal_increments_edges(self) -> None:
        g = _chain(3)
        traverse(g, ["n0"], max_hops=2, direction="out", track_traversal=True, tick=7)
        followed = [e for e in g.edges() if e.traversal_count > 0]
        assert len(followed) == 2
        assert all(e.last_traversed_tick == 7 for e in followed)

    def test_no_tracking_by_default(self) -> None:
        g = _chain(3)
        traverse(g, ["n0"], max_hops=2, direction="out")
        assert all(e.traversal_count == 0 for e in g.edges())

    def test_rejects_bad_max_nodes(self) -> None:
        with pytest.raises(ValueError, match="max_nodes"):
            traverse(_chain(3), ["n0"], max_nodes=0)
