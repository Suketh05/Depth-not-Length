"""Tests for the structured-graph memory arm (the hero arm).

The load-bearing test demonstrates the thesis in miniature: a governing decision
whose text does not resemble the query is recovered by following an explicit
CONSTRAINS link from a query-similar justification node — a hop the dense arm
cannot make, so it buries the decision.
"""

from __future__ import annotations

import pytest

from membench.retrieval import dense  # noqa: F401
from membench.retrieval.base import ArmFamily, build_arm, get_spec
from membench.retrieval.graph import arm  # noqa: F401  (registration side effect)
from membench.types import MemoryItem

# D-1 is the governing decision; its text shares no words with the query. The
# justification (D-1-rationale) shares the query's vocabulary and CONSTRAINS D-1.
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


class TestRegistration:
    @pytest.mark.parametrize(
        "name", ["brief_graph", "brief_graph_1hop", "brief_graph_3hop", "brief_graph_decay"]
    )
    def test_variants_registered_as_link_following_graph(self, name: str) -> None:
        spec = get_spec(name)
        assert spec.capabilities.family is ArmFamily.GRAPH
        assert spec.capabilities.follows_links is True


class TestLinkFollowingRecovery:
    def test_recovers_governing_decision_through_link(self) -> None:
        graph_arm = build_arm("brief_graph", seeds=1, max_hops=1)
        graph_arm.write(_DEPTH_CORPUS)  # type: ignore[attr-defined]
        ids = graph_arm.retrieve(_QUERY, budget_tokens=10_000).ids  # type: ignore[attr-defined]
        # seed = the query-similar justification; D-1 reached by the CONSTRAINS hop
        assert "D-1-rationale" in ids
        assert "D-1" in ids

    def test_dense_buries_the_decision_the_graph_recovers(self) -> None:
        # Same corpus/query through the dense arm: the decision text is dissimilar,
        # so dense ranks D-1 below the query-similar justification.
        dense_arm = build_arm("dense")
        dense_arm.write(_DEPTH_CORPUS)  # type: ignore[attr-defined]
        dense_ids = list(dense_arm.retrieve(_QUERY, 10_000).ids)  # type: ignore[attr-defined]
        graph_arm = build_arm("brief_graph", seeds=1, max_hops=1)
        graph_arm.write(_DEPTH_CORPUS)  # type: ignore[attr-defined]
        graph_ids = list(graph_arm.retrieve(_QUERY, 10_000).ids)  # type: ignore[attr-defined]
        # dense ranks the justification above the governing decision...
        assert dense_ids.index("D-1-rationale") < dense_ids.index("D-1")
        # ...while the graph places the governing decision adjacent to its seed.
        assert graph_ids.index("D-1") <= 1

    def test_degrades_to_seed_ranking_without_edges(self) -> None:
        # A shallow corpus (no constrains edges) -> the arm just returns its seeds.
        flat = [MemoryItem("A", "audit log on export"), MemoryItem("B", "button colours")]
        a = build_arm("brief_graph", seeds=1, max_hops=2)
        a.write(flat)  # type: ignore[attr-defined]
        ids = a.retrieve("audit log export", 10_000).ids  # type: ignore[attr-defined]
        assert ids[0] == "A"  # most similar seed, no link to follow

    def test_explicit_edges_metadata(self) -> None:
        corpus = [
            MemoryItem("X", "alpha", metadata={"edges": [("Y", "related_to")]}),
            MemoryItem("Y", "beta"),
        ]
        a = build_arm("brief_graph", seeds=1, max_hops=1)
        a.write(corpus)  # type: ignore[attr-defined]
        assert set(a.retrieve("alpha", 10_000).ids) == {"X", "Y"}  # type: ignore[attr-defined]

    def test_decay_variant_runs_and_reinforces(self) -> None:
        a = build_arm("brief_graph_decay", seeds=1, max_hops=1)
        a.write(_DEPTH_CORPUS)  # type: ignore[attr-defined]
        for _ in range(3):
            a.retrieve(_QUERY, 10_000)  # repeated use reinforces the traversed edge
        # no exception, and the governing decision is still recovered
        assert "D-1" in a.retrieve(_QUERY, 10_000).ids  # type: ignore[attr-defined]

    def test_rejects_bad_seeds(self) -> None:
        with pytest.raises(ValueError, match="seeds"):
            build_arm("brief_graph", seeds=0)

    def test_empty_corpus(self) -> None:
        assert build_arm("brief_graph").retrieve("x", 1000).ids == ()
