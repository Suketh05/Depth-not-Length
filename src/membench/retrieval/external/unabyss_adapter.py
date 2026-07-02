r"""Offline protocol model of Unabyss as evaluated in the paper (tab:stdbench).

Unabyss appears in the paper only as a competitor *product name* in the unified-harness
field (tab:stdbench, where it is the strongest non-Brief system on SWE-ContextBench
resolution at 37.6%, cited in fig:stdbench). The paper states no internal mechanism for it.
We model it faithfully to its evaluated profile -- a dense retriever that spends its budget
on *coverage* rather than on near-duplicates -- as embedding retrieval followed by Maximal
Marginal Relevance diversification (Carbonell & Goldstein, *The Use of MMR*, SIGIR 1998).
This is an offline, deterministic stand-in for that mechanism, **not** a reproduction of the
vendor's internals nor of the paper's measured tab:stdbench numbers (those come from the
full public-benchmark harness).

MMR only reorders similarity-ranked candidates to reduce redundancy; the arm still ranks by
resemblance and cannot follow a governance edge, so by Theorem thm:sim it remains
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
from membench.retrieval.mmr import DEFAULT_LAMBDA, mmr_order
from membench.types import MemoryItem, RetrievedContext

__all__ = ["UnabyssMemory"]


@register_arm(
    "unabyss",
    family=ArmFamily.DENSE,
    follows_links=False,
    uses_embeddings=True,
    requires_network=False,
    deterministic=True,
)
class UnabyssMemory(MemorySystem):
    """Dense retrieval with MMR diversification (offline protocol model of Unabyss).

    Items are embedded and scored by cosine similarity to the query, then reordered by the
    greedy Maximal Marginal Relevance rule so each successive pick trades query relevance
    against redundancy with the already-selected set. The result covers distinct facets of
    the corpus instead of packing the budget with near-duplicate items.

    Parameters
    ----------
    lambda_
        Relevance/diversity trade-off passed to :func:`membench.retrieval.mmr.mmr_order`
        (``1`` is pure relevance, ``0`` is pure diversity); must be in ``[0, 1]``.
    provider
        Embedding backend; defaults to the deterministic offline hashing provider.
    """

    def __init__(
        self,
        lambda_: float = DEFAULT_LAMBDA,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}")
        self._lambda = lambda_
        self._provider = provider or HashingEmbeddingProvider()
        self._items: list[MemoryItem] = []
        self._matrix: NDArray[np.float64] | None = None

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Ingest the corpus and embed every item into the vector space."""
        self._items.extend(items)
        if self._items:
            self._matrix = self._provider.embed([item.text for item in self._items])

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Return the MMR-diversified similarity ranking packed into the budget."""
        if not self._items or self._matrix is None:
            return RetrievedContext.empty()
        query_vec = self._provider.embed_one(query)
        relevance = cosine_similarity_matrix(query_vec, self._matrix).ravel()
        redundancy = cosine_similarity_matrix(self._matrix, self._matrix)
        order = mmr_order(relevance, redundancy, self._lambda)
        ranked = [(self._items[i].item_id, self._items[i].text) for i in order]
        return pack_to_budget(ranked, budget_tokens)
