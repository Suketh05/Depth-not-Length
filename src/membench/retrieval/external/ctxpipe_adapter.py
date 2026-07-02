r"""Offline protocol model of ctx| as evaluated in the paper (tab:stdbench).

``ctx|`` (written with a trailing pipe) appears in the paper only as a competitor *product
name* in the unified-harness field (tab:stdbench, tab:tok_agent_economics); the paper states
no internal mechanism for it. Its pipe-styled name reads as a retrieval *pipeline*, so we
model it faithfully as a two-signal fusion arm: a lexical ranking and a dense ranking of the
same corpus, combined with Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).
This is an offline, deterministic stand-in for that mechanism, **not** a reproduction of the
vendor's internals nor of the paper's measured tab:stdbench numbers (those come from the full
public-benchmark harness).

Both fused rankings are *similarity* rankings, so -- exactly like the built-in ``hybrid_rrf``
arm -- the pipeline cannot follow a governance edge and by Theorem thm:sim remains
depth-decaying below the typed-traversal ``q^d`` guarantee (Theorem thm:struct). It reuses the
shared, tested :func:`membench.retrieval.rank_fusion_graph.reciprocal_rank_fusion` so its RRF
identity matches every other fusion arm exactly. The arm is runnable with no network or API
key.
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
from membench.retrieval.rank_fusion_graph import reciprocal_rank_fusion
from membench.types import MemoryItem, RetrievedContext

__all__ = ["CtxPipeMemory"]

_DEFAULT_RRF_K = 60


@register_arm(
    "ctxpipe",
    family=ArmFamily.HYBRID,
    follows_links=False,
    uses_embeddings=True,
    requires_network=False,
    deterministic=True,
)
class CtxPipeMemory(MemorySystem):
    """Lexical + dense pipeline fused by Reciprocal Rank Fusion (offline model of ctx|).

    ``write`` caches term-frequency counts (lexical signal) and embeds the corpus (dense
    signal). ``retrieve`` produces a lexical ranking and a dense ranking, fuses them with
    :func:`reciprocal_rank_fusion`, and packs the fused order to budget. RRF consumes only
    ranks, which is what lets a term-overlap score and an embedding cosine -- two
    incommensurable scales -- be combined without normalisation.

    Parameters
    ----------
    k
        RRF smoothing constant (60 in the original paper). Must be positive.
    provider
        Embedding backend for the dense ranking; defaults to the deterministic offline
        hashing provider.
    """

    def __init__(self, k: int = _DEFAULT_RRF_K, provider: EmbeddingProvider | None = None) -> None:
        if k <= 0:
            raise ValueError(f"RRF k must be positive, got {k}")
        self._k = k
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

    def _lexical_ranking(self, query: str) -> list[str]:
        query_terms = tokenize(query)
        scores = [sum(tf.get(term, 0) for term in query_terms) for tf in self._tfs]
        order = sorted(range(len(self._items)), key=lambda i: -scores[i])
        return [self._items[i].item_id for i in order]

    def _dense_ranking(self, query: str) -> list[str]:
        assert self._matrix is not None
        query_vec = self._provider.embed_one(query)
        sims = cosine_similarity_matrix(query_vec, self._matrix).ravel()
        order = np.argsort(-sims, kind="stable").tolist()
        return [self._items[i].item_id for i in order]

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Fuse the lexical and dense rankings via RRF and pack the result to budget."""
        if not self._items or self._matrix is None:
            return RetrievedContext.empty()
        lexical = self._lexical_ranking(query)
        dense = self._dense_ranking(query)
        scores = reciprocal_rank_fusion([lexical, dense], k=self._k)
        text_by_id = {item.item_id: item.text for item in self._items}
        insertion = [item.item_id for item in self._items]
        ordered = sorted(insertion, key=lambda i: -scores.get(i, 0.0))
        ranked = [(item_id, text_by_id[item_id]) for item_id in ordered]
        return pack_to_budget(ranked, budget_tokens)
