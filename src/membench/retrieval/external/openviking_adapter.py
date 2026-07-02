r"""Offline protocol model of OpenViking as evaluated in the paper (tab:stdbench).

OpenViking appears in the paper only as a competitor *product name* in the unified-harness
field (tab:stdbench, tab:tok_agent_economics); the paper states no internal mechanism for it.
Its mid-field, no-standout profile is consistent with a plain managed *vector store*, so we
model it as pure dense k-nearest-neighbour retrieval that returns a fixed top-``k`` page. This
is an offline, deterministic stand-in, **not** a reproduction of the vendor's internals nor of
the paper's measured tab:stdbench numbers (those come from the full public-benchmark harness).

Two behaviours distinguish a bare vector store: it will surface an embedding-similar item even
when it shares no query term (no lexical gate, unlike the ``driver`` pipeline), and it returns
a fixed page of ``top_k`` hits regardless of the remaining budget (a managed-store default).
It is a similarity retriever with no link to follow, so by Theorem thm:sim it remains
depth-decaying below the typed-traversal ``q^d`` guarantee (Theorem thm:struct). The arm is
runnable with no network or API key.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from membench.retrieval.base import ArmFamily, MemorySystem, pack_to_budget, register_arm
from membench.retrieval.embeddings import (
    EmbeddingProvider,
    HashingEmbeddingProvider,
    cosine_similarity_matrix,
)
from membench.types import MemoryItem, RetrievedContext

__all__ = ["OpenVikingMemory"]

_DEFAULT_TOP_K = 10


@register_arm(
    "openviking",
    family=ArmFamily.DENSE,
    follows_links=False,
    uses_embeddings=True,
    requires_network=False,
    deterministic=True,
)
class OpenVikingMemory(MemorySystem):
    """Pure dense kNN vector store returning a fixed page (offline protocol model of OpenViking).

    ``write`` embeds the corpus; ``retrieve`` ranks by cosine similarity to the query and
    returns at most ``top_k`` hits, packed to budget. There is no lexical gate, so a
    semantically-close item surfaces even with zero term overlap, and no more than ``top_k``
    items are ever returned even when the budget could hold more.

    Parameters
    ----------
    top_k
        Fixed page size returned by the store. Must be >= 1.
    provider
        Embedding backend; defaults to the deterministic offline hashing provider.
    """

    def __init__(
        self, top_k: int = _DEFAULT_TOP_K, provider: EmbeddingProvider | None = None
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self._top_k = top_k
        self._provider = provider or HashingEmbeddingProvider()
        self._items: list[MemoryItem] = []
        self._matrix: NDArray[np.float64] | None = None

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Ingest the corpus and embed every item into the vector space."""
        self._items.extend(items)
        if self._items:
            self._matrix = self._provider.embed([item.text for item in self._items])

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Return the top-``k`` cosine-nearest items packed into the budget."""
        if not self._items or self._matrix is None:
            return RetrievedContext.empty()
        query_vec = self._provider.embed_one(query)
        sims = cosine_similarity_matrix(query_vec, self._matrix).ravel()
        order = np.argsort(-sims, kind="stable").tolist()[: self._top_k]
        ranked = [(self._items[i].item_id, self._items[i].text) for i in order]
        return pack_to_budget(ranked, budget_tokens)
