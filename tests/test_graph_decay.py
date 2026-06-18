"""Tests for usage-aware confidence decay and pruning."""

from __future__ import annotations

from membench.retrieval.graph.decay import DecayPolicy, apply_decay
from membench.retrieval.graph.store import GraphEdge, GraphNode, RelationshipType, TypedGraph


def _two_node_graph(**edge_kw: object) -> tuple[TypedGraph, GraphEdge]:
    g = TypedGraph()
    g.add_node(GraphNode("a", "a"))
    g.add_node(GraphNode("b", "b"))
    edge = GraphEdge("a", "b", RelationshipType.RELATED_TO, **edge_kw)  # type: ignore[arg-type]
    g.add_edge(edge)
    return g, edge


class TestApplyDecay:
    def test_stale_edge_decays(self) -> None:
        g, edge = _two_node_graph(confidence=0.8, last_traversed_tick=None)
        report = apply_decay(g, current_tick=10, policy=DecayPolicy(default_decay_rate=0.5))
        assert edge.confidence == 0.4  # 0.8 * (1 - 0.5)
        assert report.decayed == 1 and report.reinforced == 0

    def test_recently_traversed_edge_not_decayed(self) -> None:
        g, edge = _two_node_graph(confidence=0.8, last_traversed_tick=10)
        apply_decay(
            g, current_tick=10, policy=DecayPolicy(default_decay_rate=0.5, stale_after_ticks=1)
        )
        assert edge.confidence == 0.8  # fresh -> untouched

    def test_frequently_traversed_edge_reinforced(self) -> None:
        g, edge = _two_node_graph(confidence=0.5, traversal_count=15, last_traversed_tick=10)
        report = apply_decay(
            g, current_tick=10, policy=DecayPolicy(default_decay_rate=0.4, restore_factor=0.5)
        )
        assert edge.confidence == 0.7  # 0.5 + 0.5 * 0.4
        assert report.reinforced == 1

    def test_reinforcement_capped_at_one(self) -> None:
        g, edge = _two_node_graph(confidence=0.95, traversal_count=20, last_traversed_tick=5)
        apply_decay(
            g, current_tick=5, policy=DecayPolicy(default_decay_rate=0.5, restore_factor=1.0)
        )
        assert edge.confidence == 1.0

    def test_prunes_stale_low_confidence_never_traversed(self) -> None:
        g, _ = _two_node_graph(confidence=0.15, last_traversed_tick=None)
        report = apply_decay(
            g, current_tick=10, policy=DecayPolicy(default_decay_rate=0.5, prune_below=0.1)
        )
        # 0.15 -> 0.075 < 0.1 and never traversed -> pruned
        assert report.pruned == 1
        assert g.num_edges == 0

    def test_traversed_low_confidence_not_pruned(self) -> None:
        g, _ = _two_node_graph(confidence=0.05, traversal_count=1, last_traversed_tick=10)
        report = apply_decay(g, current_tick=10, policy=DecayPolicy(prune_below=0.1))
        assert report.pruned == 0  # traversed -> survives even at low confidence

    def test_per_type_rate_override(self) -> None:
        g, edge = _two_node_graph(confidence=1.0, last_traversed_tick=None)
        policy = DecayPolicy(default_decay_rate=0.1, decay_rates={RelationshipType.RELATED_TO: 0.9})
        apply_decay(g, current_tick=10, policy=policy)
        assert edge.confidence == 1.0 * (1 - 0.9)
