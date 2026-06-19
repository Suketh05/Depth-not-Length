"""The structured-graph memory arm — Brief's mechanism as a benchmark arm.

This is the positive arm the whole benchmark is built to evaluate. It works in two
stages, mirroring how ``brief_ask`` finds relevant entities and then walks the
graph:

1. **Seed by similarity.** Embed the query and pick the most similar nodes as entry
   points (the same single-hop signal the similarity arms use, and no better).
2. **Follow the links.** From those seeds, run a bounded multi-hop BFS over the
   typed edges, recovering nodes that are *connected* to the seeds even when they
   do not resemble the query — the governing decision sitting behind a
   ``CONSTRAINS`` edge, several hops from anything the query looks like.

The graph is built from the task corpus: each item becomes a node, and any item
carrying a ``constrains`` metadata key (a stripped justification node) becomes a
``CONSTRAINS`` edge to the decision it justifies. Explicit edge lists in metadata
(``edges``) are honoured too. When the corpus has no edges (shallow tasks) the arm
degrades gracefully to the dense seed ranking — which is the point: the structured
advantage appears only with depth.
"""

from __future__ import annotations

from collections.abc import Iterable

from membench.retrieval._native import cosine_topk_reference
from membench.retrieval.base import ArmFamily, MemorySystem, pack_to_budget, register_arm
from membench.retrieval.embeddings import EmbeddingProvider, HashingEmbeddingProvider
from membench.retrieval.graph.decay import DecayPolicy, apply_decay
from membench.retrieval.graph.store import GraphEdge, GraphNode, RelationshipType, TypedGraph
from membench.retrieval.graph.traversal import DEFAULT_MAX_HOPS, traverse
from membench.types import MemoryItem, RetrievedContext

__all__ = [
    "BriefGraphMemory",
    "BriefGraphMemory1Hop",
    "BriefGraphMemory3Hop",
    "BriefGraphMemoryDecay",
]


@register_arm("brief_graph", family=ArmFamily.GRAPH, follows_links=True, uses_embeddings=True)
class BriefGraphMemory(MemorySystem):
    """Seed by similarity, then follow typed links via bounded multi-hop traversal.

    Parameters
    ----------
    max_hops
        Traversal depth (clamped to [1, 3] by the traversal layer).
    seeds
        Number of similarity-selected entry nodes.
    provider
        Embedding backend for seed selection (defaults to hashing).
    use_decay
        Track traversals and run a usage-aware decay pass after each retrieval,
        reinforcing links the agent actually relies on.
    decay_policy
        Policy for the decay pass (defaults to :class:`DecayPolicy`).
    """

    def __init__(
        self,
        max_hops: int = DEFAULT_MAX_HOPS,
        seeds: int = 5,
        provider: EmbeddingProvider | None = None,
        use_decay: bool = False,
        decay_policy: DecayPolicy | None = None,
    ) -> None:
        if seeds < 1:
            raise ValueError(f"seeds must be >= 1, got {seeds}")
        self._max_hops = max_hops
        self._seeds = seeds
        self._provider = provider or HashingEmbeddingProvider()
        self._use_decay = use_decay
        self._decay_policy = decay_policy or DecayPolicy()
        self._graph = TypedGraph()
        self._items: list[MemoryItem] = []
        self._tick = 0

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Build the typed graph from the corpus (nodes + constrains/edges metadata)."""
        self._items.extend(items)
        for item in self._items:
            self._graph.add_node(
                GraphNode(item.item_id, item.text, node_type=item.node_type or "node")
            )
        for item in self._items:
            if item.constrains and self._graph.has_node(item.constrains):
                self._graph.add_edge(
                    GraphEdge(item.item_id, item.constrains, RelationshipType.CONSTRAINS)
                )
            for spec in item.metadata.get("edges", ()):  # explicit edges: (target, rel_type)
                target, rel = spec
                if self._graph.has_node(target):
                    self._graph.add_edge(GraphEdge(item.item_id, target, RelationshipType(rel)))

    def _seed_ids(self, query: str) -> list[str]:
        matrix = self._provider.embed([item.text for item in self._items])
        query_vec = self._provider.embed_one(query)
        idx, _ = cosine_topk_reference(matrix, query_vec, min(self._seeds, len(self._items)))
        return [self._items[i].item_id for i in idx]

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Seed by similarity, follow links most-relevant-thread-first, pack to budget.

        Seeds are expanded in descending similarity order, each via its own bounded
        traversal, and the reached nodes are concatenated (de-duplicated). This is
        the standard relevance-ordered graph expansion: the chain hanging off the
        most query-relevant entity -- which carries the governing decision behind
        its links -- is packed before less-relevant chains, so a deep governing node
        survives a tight budget instead of being crowded out.
        """
        if not self._items:
            return RetrievedContext.empty()
        ordered: list[tuple[str, str]] = []
        seen: set[str] = set()
        for seed_id in self._seed_ids(query):
            result = traverse(
                self._graph,
                [seed_id],
                max_hops=self._max_hops,
                track_traversal=self._use_decay,
                tick=self._tick,
            )
            # The governing decisions are the nodes that code/justifications link TO via
            # typed dependency edges -- the structured signal Brief is built to surface.
            # Spend the budget on the seed plus those linked decisions, not on incidental
            # neighbours reached along the way, so the governing context is not diluted
            # (and more of a multi-decision chain fits under a tight budget). With no
            # dependency edges (shallow tasks) this is a no-op and the seed ranking stands.
            dep_targets = {
                edge.target_id
                for edge in result.edges
                if edge.relationship_type is RelationshipType.CONSTRAINS
            }
            for node, depth in result.nodes:
                if dep_targets and depth != 0 and node.node_id not in dep_targets:
                    continue
                if node.node_id not in seen:
                    seen.add(node.node_id)
                    ordered.append((node.node_id, node.text))
        if self._use_decay:
            self._tick += 1
            apply_decay(self._graph, self._tick, self._decay_policy)
        return pack_to_budget(ordered, budget_tokens)


@register_arm("brief_graph_1hop", family=ArmFamily.GRAPH, follows_links=True, uses_embeddings=True)
class BriefGraphMemory1Hop(BriefGraphMemory):
    """One-hop variant: follows a single typed link from each seed."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(max_hops=1, **kwargs)  # type: ignore[arg-type]


@register_arm("brief_graph_3hop", family=ArmFamily.GRAPH, follows_links=True, uses_embeddings=True)
class BriefGraphMemory3Hop(BriefGraphMemory):
    """Three-hop variant: follows up to three typed links from each seed."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(max_hops=3, **kwargs)  # type: ignore[arg-type]


@register_arm("brief_graph_decay", family=ArmFamily.GRAPH, follows_links=True, uses_embeddings=True)
class BriefGraphMemoryDecay(BriefGraphMemory):
    """Usage-aware variant: reinforces traversed links via the decay pass."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(use_decay=True, **kwargs)  # type: ignore[arg-type]
