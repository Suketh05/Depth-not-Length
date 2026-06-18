"""Hybrid retrieval arm: Reciprocal Rank Fusion of lexical and dense rankings.

Production RAG stacks rarely use BM25 *or* embeddings alone; they fuse both.
Reciprocal Rank Fusion (Cormack et al., 2009) combines the two rankings without
tuning score scales: an item's fused score is the sum over rankers of
``1 / (k + rank)``. This is the strongest non-graph baseline in the suite — it
inherits both lexical-overlap and semantic recall — which makes it the most
demanding similarity comparison for the structured arm. It still cannot follow an
explicit link, so it remains subject to the depth ceiling.
"""

from __future__ import annotations

from collections.abc import Iterable

from membench.retrieval.base import ArmFamily, MemorySystem, pack_to_budget, register_arm
from membench.retrieval.bm25 import BM25Memory
from membench.retrieval.dense import DenseMemory
from membench.retrieval.embeddings import EmbeddingProvider
from membench.types import MemoryItem, RetrievedContext

__all__ = ["HybridRRFMemory"]

# Budget large enough to force a full ranking out of each sub-arm (we fuse the
# complete orderings, then truncate the fused order to the real budget).
_FULL_RANKING_BUDGET = 1_000_000_000


@register_arm("hybrid_rrf", family=ArmFamily.HYBRID, follows_links=False, uses_embeddings=True)
class HybridRRFMemory(MemorySystem):
    """Fuse BM25 and dense rankings with Reciprocal Rank Fusion, then pack to budget.

    Parameters
    ----------
    k
        RRF constant; larger values flatten the contribution of top ranks. 60 is
        the value from the original paper and the common default.
    provider
        Embedding backend for the dense component (defaults to hashing).
    """

    def __init__(self, k: int = 60, provider: EmbeddingProvider | None = None) -> None:
        if k <= 0:
            raise ValueError(f"RRF k must be positive, got {k}")
        self._k = k
        self._bm25 = BM25Memory()
        self._dense = DenseMemory(provider=provider)
        self._text_by_id: dict[str, str] = {}
        self._order: list[str] = []

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Ingest the corpus into both sub-rankers."""
        materialised = list(items)
        self._bm25.write(materialised)
        self._dense.write(materialised)
        for item in materialised:
            self._text_by_id.setdefault(item.item_id, item.text)
            self._order.append(item.item_id)

    def _rrf_scores(self, query: str) -> dict[str, float]:
        scores: dict[str, float] = dict.fromkeys(self._order, 0.0)
        for arm in (self._bm25, self._dense):
            ranking = arm.retrieve(query, _FULL_RANKING_BUDGET).ids
            for rank, item_id in enumerate(ranking):
                scores[item_id] += 1.0 / (self._k + rank)
        return scores

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Return the RRF-fused ranking packed into the budget."""
        if not self._text_by_id:
            return RetrievedContext.empty()
        scores = self._rrf_scores(query)
        # scores preserves corpus insertion order; a stable sort by -score therefore
        # breaks ties by original corpus order without an O(n^2) index lookup.
        ordered = sorted(scores, key=lambda i: -scores[i])
        ranked = [(item_id, self._text_by_id[item_id]) for item_id in ordered]
        return pack_to_budget(ranked, budget_tokens)
