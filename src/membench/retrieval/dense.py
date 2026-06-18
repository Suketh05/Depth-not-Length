"""Dense embedding retrieval arm.

Embed the corpus and the query into a shared vector space and rank items by cosine
similarity — the modern RAG default and the closest single-hop analogue to Brief's
own embedding fallback. The ranking is computed with the native cosine-top-k kernel
when it is built and the NumPy reference otherwise (the two are parity-validated),
so this arm exercises the native path end to end.

Like the lexical arms, dense retrieval scores by resemblance to the query: it
recovers semantically *similar* nodes well and still misses a governing decision
that sits several hops away and is phrased unlike the task. The embedding backend
is pluggable (deterministic hashing by default; neural backends added separately),
so the arm never depends on a specific provider.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from membench.retrieval._native import (
    cosine_topk_native,
    cosine_topk_reference,
    native_available,
)
from membench.retrieval.base import ArmFamily, MemorySystem, pack_to_budget, register_arm
from membench.retrieval.embeddings import EmbeddingProvider, HashingEmbeddingProvider
from membench.types import MemoryItem, RetrievedContext

__all__ = ["DenseMemory"]


@register_arm("dense", family=ArmFamily.DENSE, follows_links=False, uses_embeddings=True)
class DenseMemory(MemorySystem):
    """Rank corpus items by embedding cosine similarity, then pack to budget.

    Parameters
    ----------
    provider
        Embedding backend; defaults to the deterministic offline hashing provider.
    use_native
        Use the native cosine-top-k kernel when available (falls back to the NumPy
        reference); set ``False`` to force the reference path.
    """

    def __init__(self, provider: EmbeddingProvider | None = None, use_native: bool = True) -> None:
        self._provider = provider or HashingEmbeddingProvider()
        self._use_native = use_native
        self._items: list[MemoryItem] = []
        self._matrix: NDArray[np.float64] | None = None

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Ingest the corpus and embed every item into the vector space."""
        self._items.extend(items)
        if self._items:
            self._matrix = self._provider.embed([item.text for item in self._items])

    def _rank(self, query_vec: NDArray[np.float64]) -> list[int]:
        assert self._matrix is not None
        k = self._matrix.shape[0]
        if self._use_native and native_available():
            idx, _ = cosine_topk_native(self._matrix, query_vec, k)
        else:
            idx, _ = cosine_topk_reference(self._matrix, query_vec, k)
        return [int(i) for i in idx]

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Return the embedding-ranked corpus packed into the budget."""
        if not self._items or self._matrix is None:
            return RetrievedContext.empty()
        query_vec = self._provider.embed_one(query)
        ranked = [(self._items[i].item_id, self._items[i].text) for i in self._rank(query_vec)]
        return pack_to_budget(ranked, budget_tokens)
