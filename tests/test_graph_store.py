"""Tests for the typed property-graph store."""

from __future__ import annotations

import pytest

from membench.retrieval.graph.store import GraphEdge, GraphNode, RelationshipType, TypedGraph


def _graph() -> TypedGraph:
    g = TypedGraph()
    g.add_node(GraphNode("D-1", "constraint text", node_type="constraint"))
    g.add_node(GraphNode("D-1-rationale", "justification text", node_type="justification"))
    g.add_node(GraphNode("D-2", "another decision"))
    g.add_edge(GraphEdge("D-1-rationale", "D-1", RelationshipType.CONSTRAINS, confidence=0.9))
    g.add_edge(GraphEdge("D-1", "D-2", RelationshipType.RELATED_TO, confidence=0.5))
    return g


class TestGraphNode:
    def test_rejects_empty_id(self) -> None:
        with pytest.raises(ValueError, match="node_id"):
            GraphNode("", "t")

    def test_metadata_readonly(self) -> None:
        node = GraphNode("n", "t", metadata={"k": "v"})
        with pytest.raises(TypeError):
            node.metadata["k"] = "z"  # type: ignore[index]


class TestGraphEdge:
    def test_coerces_string_relationship(self) -> None:
        edge = GraphEdge("a", "b", "constrains")  # type: ignore[arg-type]
        assert edge.relationship_type is RelationshipType.CONSTRAINS

    def test_rejects_bad_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            GraphEdge("a", "b", RelationshipType.RELATED_TO, confidence=1.5)


class TestTypedGraph:
    def test_counts(self) -> None:
        g = _graph()
        assert g.num_nodes == 3 and g.num_edges == 2

    def test_duplicate_node_rejected(self) -> None:
        g = _graph()
        with pytest.raises(ValueError, match="duplicate"):
            g.add_node(GraphNode("D-1", "x"))

    def test_edge_fk_enforced(self) -> None:
        g = TypedGraph()
        g.add_node(GraphNode("a", "t"))
        with pytest.raises(ValueError, match="target"):
            g.add_edge(GraphEdge("a", "missing", RelationshipType.RELATED_TO))
        with pytest.raises(ValueError, match="source"):
            g.add_edge(GraphEdge("missing", "a", RelationshipType.RELATED_TO))

    def test_outgoing_neighbors(self) -> None:
        g = _graph()
        out = g.neighbors("D-1-rationale", direction="out")
        assert len(out) == 1
        edge, node = out[0]
        assert node.node_id == "D-1" and edge.relationship_type is RelationshipType.CONSTRAINS

    def test_incoming_neighbors(self) -> None:
        g = _graph()
        incoming = g.neighbors("D-1", direction="in")
        assert {n.node_id for _, n in incoming} == {"D-1-rationale"}

    def test_both_directions(self) -> None:
        g = _graph()
        both = g.neighbors("D-1", direction="both")
        assert {n.node_id for _, n in both} == {"D-1-rationale", "D-2"}

    def test_relationship_type_filter(self) -> None:
        g = _graph()
        related = g.neighbors(
            "D-1", direction="both", relationship_types=frozenset({RelationshipType.RELATED_TO})
        )
        assert {n.node_id for _, n in related} == {"D-2"}

    def test_min_confidence_filter(self) -> None:
        g = _graph()
        high = g.neighbors("D-1", direction="both", min_confidence=0.8)
        assert {n.node_id for _, n in high} == {"D-1-rationale"}  # 0.9 kept, 0.5 dropped

    def test_unknown_node_and_direction(self) -> None:
        g = _graph()
        with pytest.raises(KeyError):
            g.neighbors("nope")
        with pytest.raises(ValueError, match="direction"):
            g.neighbors("D-1", direction="sideways")
