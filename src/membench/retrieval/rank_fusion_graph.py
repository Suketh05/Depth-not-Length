r"""Graph-aware retrieval arm: Reciprocal Rank Fusion of similarity and graph-hop ranks.

This arm fuses two *complementary* rankings of the same corpus with Reciprocal
Rank Fusion (RRF):

1. a **similarity ranking** — corpus items ordered by embedding cosine similarity
   to the query (the single-hop signal the dense/lexical arms use); and
2. a **graph-hop ranking** — corpus items ordered by breadth-first *hop distance*
   over the typed graph, starting from the most query-similar nodes (the seeds).
   A node reached in one hop ranks above a node reached in two, and so on.

The two are combined with the standard RRF rule (Cormack, Clarke, and Buettcher,
2009, "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning
Methods", SIGIR '09, eq. 1):

.. math::

    \mathrm{RRF}(d) = \sum_{r \in R} \frac{1}{k + \mathrm{rank}_r(d)}

where :math:`R` is the set of input rankings, :math:`\mathrm{rank}_r(d)` is the
1-based position of document :math:`d` in ranking :math:`r` (rankings that omit
:math:`d` contribute nothing), and :math:`k` is a smoothing constant (60 in the
original paper). RRF needs no score normalisation because it consumes only ranks,
which is exactly why it can fuse an embedding cosine score with an integer hop
distance — two signals on incommensurable scales.

This differs from the existing ``hybrid_rrf`` arm, which fuses two *similarity*
rankings (BM25 + dense) and therefore still cannot follow an explicit link. By
fusing a similarity ranking with a *structural* graph-hop ranking, this arm
promotes a governing node that sits several hops behind a query-relevant seed even
when that node is phrased unlike the query and so ranks low on similarity alone.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from membench.retrieval.base import ArmFamily, MemorySystem, pack_to_budget, register_arm
from membench.retrieval.dense import DenseMemory
from membench.retrieval.embeddings import EmbeddingProvider
from membench.retrieval.graph.store import GraphEdge, GraphNode, RelationshipType, TypedGraph
from membench.retrieval.graph.traversal import MAX_HOPS_LIMIT, traverse
from membench.types import MemoryItem, RetrievedContext

__all__ = ["RankFusionGraphMemory", "reciprocal_rank_fusion"]

# Budget large enough to force a full ranking out of the similarity sub-arm: we
# fuse the complete orderings first, then truncate the fused order to the budget.
_FULL_RANKING_BUDGET = 1_000_000_000

_DEFAULT_RRF_K = 60
_DEFAULT_SEEDS = 5


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = _DEFAULT_RRF_K,
) -> dict[str, float]:
    r"""Fuse several rank lists into per-item Reciprocal Rank Fusion scores.

    Implements the rank-only fusion of Cormack et al. (2009):

    .. math::

        \mathrm{RRF}(d) = \sum_{r \in R} \frac{1}{k + \mathrm{rank}_r(d)}

    Ranks are **1-based** (the top of each list is rank 1, as in the original
    paper). A document contributes a term only for the rankings that contain it,
    so a document present in two rankings is rewarded over one present in a single
    ranking at the same position — the property that lets RRF reward agreement
    between rankers without comparing their raw scores.

    Parameters
    ----------
    rankings
        The input rank lists, each an ordered sequence of item ids (best first).
        Ids may repeat across lists but should be unique within a single list; a
        repeated id within one list is scored at each of its positions.
    k
        Smoothing constant; larger values flatten the contribution of top ranks.
        Must be positive. Defaults to 60 (the original-paper value).

    Returns
    -------
    dict[str, float]
        Mapping from item id to fused score, accumulated across all rankings.
        Keys appear in first-seen order across the input rankings.

    Raises
    ------
    ValueError
        If ``k`` is not positive.

    References
    ----------
    .. [1] G. V. Cormack, C. L. A. Clarke, and S. Buettcher, "Reciprocal Rank
       Fusion outperforms Condorcet and individual Rank Learning Methods",
       SIGIR '09, pp. 758-759, 2009.
    """
    if k <= 0:
        raise ValueError(f"RRF k must be positive, got {k}")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores


@register_arm(
    "rank_fusion_graph",
    family=ArmFamily.HYBRID,
    follows_links=True,
    uses_embeddings=True,
)
class RankFusionGraphMemory(MemorySystem):
    """Fuse an embedding-similarity ranking with a graph-hop ranking via RRF.

    The similarity ranking comes from an embedding cosine ranker over the whole
    corpus; the graph-hop ranking comes from a bounded breadth-first walk of the
    typed graph seeded by the top similarity hits, ordered by hop distance. The two
    rankings are fused with :func:`reciprocal_rank_fusion` and packed to budget.

    Parameters
    ----------
    k
        RRF smoothing constant (see :func:`reciprocal_rank_fusion`). Must be
        positive. Defaults to 60.
    max_hops
        Maximum graph-traversal depth from the seeds (clamped to
        ``[1, MAX_HOPS_LIMIT]`` by the traversal layer). Defaults to the traversal
        limit so deep chains are reachable.
    seeds
        Number of top similarity hits used as traversal entry points. Must be at
        least 1. Defaults to 5.
    provider
        Embedding backend for the similarity ranking and seed selection; defaults
        to the deterministic offline hashing provider.
    """

    def __init__(
        self,
        k: int = _DEFAULT_RRF_K,
        max_hops: int = MAX_HOPS_LIMIT,
        seeds: int = _DEFAULT_SEEDS,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError(f"RRF k must be positive, got {k}")
        if seeds < 1:
            raise ValueError(f"seeds must be >= 1, got {seeds}")
        self._k = k
        self._max_hops = max_hops
        self._seeds = seeds
        self._dense = DenseMemory(provider=provider)
        self._graph = TypedGraph()
        self._items: list[MemoryItem] = []
        self._text_by_id: dict[str, str] = {}
        self._order: list[str] = []

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Ingest the corpus into the similarity ranker and build the typed graph.

        Nodes are added for every item; a ``CONSTRAINS`` edge is added for each item
        carrying a ``constrains`` metadata key, and any explicit ``edges`` metadata
        ``(target, relationship_type)`` pairs are honoured (both mirror the
        structured-graph arm's construction).
        """
        materialised = list(items)
        self._items.extend(materialised)
        self._dense.write(materialised)
        for item in materialised:
            self._text_by_id.setdefault(item.item_id, item.text)
            self._order.append(item.item_id)
            self._graph.add_node(
                GraphNode(item.item_id, item.text, node_type=item.node_type or "node")
            )
        for item in materialised:
            if item.constrains and self._graph.has_node(item.constrains):
                self._graph.add_edge(
                    GraphEdge(item.item_id, item.constrains, RelationshipType.CONSTRAINS)
                )
            for spec in item.metadata.get("edges", ()):  # explicit edges: (target, rel_type)
                target, rel = spec
                if self._graph.has_node(target):
                    self._graph.add_edge(GraphEdge(item.item_id, target, RelationshipType(rel)))

    def _similarity_ranking(self, query: str) -> tuple[str, ...]:
        """Full corpus ordering by embedding cosine similarity to ``query``."""
        return self._dense.retrieve(query, _FULL_RANKING_BUDGET).ids

    def _graph_hop_ranking(self, similarity_ranking: Sequence[str]) -> list[str]:
        """Corpus ordering by BFS hop distance from the top similarity seeds.

        The most query-similar items become traversal seeds (depth 0); the bounded
        breadth-first walk then orders reached nodes by increasing hop distance, so
        a node one link away outranks a node two links away. Items not reachable
        from any seed are absent (they contribute no graph term to the fusion).
        """
        seed_ids = list(similarity_ranking[: self._seeds])
        result = traverse(self._graph, seed_ids, max_hops=self._max_hops)
        return result.node_ids

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Return the RRF-fused similarity+graph-hop ranking packed into the budget."""
        if not self._items:
            return RetrievedContext.empty()
        similarity_ranking = self._similarity_ranking(query)
        graph_ranking = self._graph_hop_ranking(similarity_ranking)
        scores = reciprocal_rank_fusion([similarity_ranking, graph_ranking], k=self._k)
        # self._order is corpus insertion order; a stable sort by -score therefore
        # breaks ties by original corpus order without an O(n^2) index lookup.
        ordered = sorted(self._order, key=lambda i: -scores.get(i, 0.0))
        ranked = [(item_id, self._text_by_id[item_id]) for item_id in ordered]
        return pack_to_budget(ranked, budget_tokens)
