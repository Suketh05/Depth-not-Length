r"""Offline protocol model of Driver as evaluated in the paper (tab:stdbench).

Driver appears in the paper only as a competitor *product name* in the unified-harness field
(tab:stdbench, tab:tok_agent_economics); the paper states no internal mechanism for it. We
model it faithfully as the canonical two-stage retrieve-then-rerank *pipeline* (Nogueira &
Cho, *Passage Re-ranking with BERT*, 2019): a cheap lexical first stage proposes a candidate
set, and a dense re-ranker orders that set. This is an offline, deterministic stand-in for
that mechanism, **not** a reproduction of the vendor's internals nor of the paper's measured
tab:stdbench numbers (those come from the full public-benchmark harness).

The distinctive consequence of a first-stage *gate* is a recall ceiling: an item the lexical
stage never proposes cannot be recovered by the re-ranker no matter how embedding-similar it
is -- so the pipeline stays resemblance-bounded and, by Theorem thm:sim, depth-decaying below
the typed-traversal ``q^d`` guarantee (Theorem thm:struct). The arm is runnable with no
network or API key.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from membench.retrieval._text import tokenize
from membench.retrieval.base import ArmFamily, MemorySystem, pack_to_budget, register_arm
from membench.retrieval.embeddings import (
    EmbeddingProvider,
    HashingEmbeddingProvider,
    cosine_similarity_matrix,
)
from membench.types import MemoryItem, RetrievedContext

__all__ = ["DriverMemory"]


@register_arm(
    "driver",
    family=ArmFamily.HYBRID,
    follows_links=False,
    uses_embeddings=True,
    requires_network=False,
    deterministic=True,
)
class DriverMemory(MemorySystem):
    """Lexical-gated, dense-reranked retrieval pipeline (offline protocol model of Driver).

    ``retrieve`` first selects the candidate set of items that share at least one term with
    the query (the lexical first stage), re-ranks *those* by embedding cosine similarity, and
    only then appends the remaining items (also cosine-ordered) as a graceful tail. A
    candidate therefore always outranks a non-candidate, so an item the first stage does not
    propose cannot be pulled to the top by resemblance alone -- the recall ceiling a
    first-stage gate imposes.

    Parameters
    ----------
    provider
        Embedding backend for the re-ranking stage; defaults to the deterministic offline
        hashing provider.
    """

    def __init__(self, provider: EmbeddingProvider | None = None) -> None:
        self._provider = provider or HashingEmbeddingProvider()
        self._items: list[MemoryItem] = []
        self._tfs: list[Counter[str]] = []
        self._matrix: NDArray[np.float64] | None = None

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Ingest the corpus, caching term frequencies and embedding every item."""
        for item in items:
            self._items.append(item)
            self._tfs.append(Counter(tokenize(item.text)))
        if self._items:
            self._matrix = self._provider.embed([item.text for item in self._items])

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Gate by lexical overlap, re-rank by cosine, and pack to budget."""
        if not self._items or self._matrix is None:
            return RetrievedContext.empty()
        query_terms = set(tokenize(query))
        query_vec = self._provider.embed_one(query)
        sims = cosine_similarity_matrix(query_vec, self._matrix).ravel()
        # First stage: candidate iff it shares a term with the query; everything else is a
        # non-candidate that can only appear behind every candidate.
        candidates = [i for i in range(len(self._items)) if query_terms & set(self._tfs[i])]
        rest = [i for i in range(len(self._items)) if i not in set(candidates)]
        candidates.sort(key=lambda i: -sims[i])
        rest.sort(key=lambda i: -sims[i])
        order = candidates + rest
        ranked = [(self._items[i].item_id, self._items[i].text) for i in order]
        return pack_to_budget(ranked, budget_tokens)
