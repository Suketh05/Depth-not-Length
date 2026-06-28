r"""Maximal Marginal Relevance (MMR) diversity reranking arm.

MMR (Carbonell & Goldstein, 1998, *The Use of MMR, Diversity-Based Reranking for
Reordering Documents and Producing Summaries*, SIGIR '98) reorders a candidate set
so that each selected item is both relevant to the query *and* novel relative to
what has already been selected. Starting from an empty selection ``S``, it greedily
adds the document ``d`` from the remaining pool ``R`` that maximises

.. math::

    \mathrm{MMR}(d) = \lambda \, \mathrm{Sim}_1(d, q)
                      - (1 - \lambda) \max_{j \in S} \mathrm{Sim}_2(d, j),

where :math:`\mathrm{Sim}_1` scores query relevance, :math:`\mathrm{Sim}_2` scores
document-document redundancy, and :math:`\lambda \in [0, 1]` trades relevance off
against diversity. With an empty selection the redundancy term is taken to be 0, so
the first pick is the most relevant item; ``lambda = 1`` ignores diversity entirely
and reduces to the pure relevance ranking, while ``lambda = 0`` maximises novelty.

Here the first-stage relevance and the pairwise redundancy both come from the same
offline, deterministic hashing-embedding cosine used by the dense arm, so the arm
needs no network or API key. Like every other similarity arm it cannot follow an
explicit link; MMR only changes the *order* of similarity-ranked candidates to
reduce near-duplicates before budget packing.
"""

from __future__ import annotations

import math
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

__all__ = ["MMRMemory", "mmr_order"]

DEFAULT_LAMBDA = 0.5
"""Default relevance/diversity trade-off (equal weight, the common MMR default)."""


def mmr_order(
    relevance: NDArray[np.float64],
    similarity: NDArray[np.float64],
    lambda_: float = DEFAULT_LAMBDA,
    *,
    k: int | None = None,
) -> list[int]:
    r"""Return the Maximal Marginal Relevance selection order of candidate indices.

    Implements the greedy MMR objective of Carbonell & Goldstein (1998): repeatedly
    select the remaining index ``d`` maximising
    ``lambda_ * relevance[d] - (1 - lambda_) * max_{j in S} similarity[d, j]``,
    where ``S`` is the set already selected. The redundancy term is 0 while ``S`` is
    empty, so the first pick is ``argmax(relevance)``. Ties are broken by lowest
    index, making the order fully deterministic.

    Parameters
    ----------
    relevance
        Length-``n`` array of first-stage query relevance scores (e.g. cosine of
        each document to the query). Higher is more relevant.
    similarity
        ``(n, n)`` symmetric matrix of pairwise document redundancy scores
        (``similarity[i, j]`` is the resemblance of documents ``i`` and ``j``).
    lambda_
        Trade-off in ``[0, 1]``: ``1`` is pure relevance, ``0`` is pure diversity.
    k
        Number of items to select; ``None`` (default) selects all ``n``.

    Returns
    -------
    list[int]
        Indices into ``relevance`` in MMR selection order, length ``min(k, n)``.

    Raises
    ------
    ValueError
        If ``lambda_`` is outside ``[0, 1]`` or ``similarity`` is not ``(n, n)``.
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}")
    n = int(relevance.shape[0])
    if similarity.shape != (n, n):
        raise ValueError(f"similarity must be ({n}, {n}), got {similarity.shape}")
    limit = n if k is None else max(0, min(k, n))

    selected: list[int] = []
    remaining = list(range(n))
    while remaining and len(selected) < limit:
        best_index = remaining[0]
        best_score = -math.inf
        for d in remaining:
            redundancy = max((float(similarity[d, j]) for j in selected), default=0.0)
            score = lambda_ * float(relevance[d]) - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_score = score
                best_index = d
        selected.append(best_index)
        remaining.remove(best_index)
    return selected


@register_arm("mmr", family=ArmFamily.DENSE, follows_links=False, uses_embeddings=True)
class MMRMemory(MemorySystem):
    """Rank by embedding cosine, then MMR-reorder for diversity before packing.

    Parameters
    ----------
    lambda_
        Relevance/diversity trade-off passed to :func:`mmr_order` (``1`` is pure
        relevance, ``0`` is pure diversity); defaults to :data:`DEFAULT_LAMBDA`.
    provider
        Embedding backend; defaults to the deterministic offline hashing provider,
        so the arm is constructible with no arguments.
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
        """Return the MMR-reordered corpus packed into the budget."""
        if not self._items or self._matrix is None:
            return RetrievedContext.empty()
        query_vec = self._provider.embed_one(query)
        relevance = cosine_similarity_matrix(query_vec, self._matrix).ravel()
        similarity = cosine_similarity_matrix(self._matrix, self._matrix)
        order = mmr_order(relevance, similarity, self._lambda)
        ranked = [(self._items[i].item_id, self._items[i].text) for i in order]
        return pack_to_budget(ranked, budget_tokens)
