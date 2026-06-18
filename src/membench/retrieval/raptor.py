"""RAPTOR-style hierarchical retrieval arm.

RAPTOR (Sarthi et al., 2024) clusters the corpus, builds higher-level *summary*
nodes over each cluster, and retrieves from a collapsed tree of leaves and summary
nodes at once, so a query can match either a specific item or a coarse theme. It is
the strongest *hierarchical* similarity baseline.

This implementation clusters the deterministic embeddings (agglomerative, cosine
average-linkage), represents each cluster by its centroid (an extractive,
model-free summary so the arm stays offline; an LLM summariser is the only part a
faithful RAPTOR would add), ranks the collapsed pool of leaf + centroid nodes
against the query, and expands matched summary nodes to their member leaves. It
still ranks by resemblance, so — like every similarity arm — it cannot follow an
explicit link and remains subject to the depth ceiling.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering

from membench.retrieval.base import ArmFamily, MemorySystem, pack_to_budget, register_arm
from membench.retrieval.embeddings import (
    EmbeddingProvider,
    HashingEmbeddingProvider,
    cosine_similarity_matrix,
)
from membench.types import MemoryItem, RetrievedContext

__all__ = ["RaptorMemory"]


@register_arm("raptor", family=ArmFamily.DENSE, follows_links=False, uses_embeddings=True)
class RaptorMemory(MemorySystem):
    """Cluster the corpus, build centroid summaries, retrieve over the collapsed tree.

    Parameters
    ----------
    provider
        Embedding backend (defaults to the deterministic hashing provider).
    n_clusters
        Number of cluster/summary nodes; defaults to ``round(sqrt(n))``.
    """

    def __init__(
        self, provider: EmbeddingProvider | None = None, n_clusters: int | None = None
    ) -> None:
        self._provider = provider or HashingEmbeddingProvider()
        self._n_clusters = n_clusters
        self._items: list[MemoryItem] = []
        self._leaf_emb: NDArray[np.float64] | None = None
        self._summary_emb: NDArray[np.float64] | None = None
        self._members: list[list[int]] = []  # member leaf indices per summary node

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Embed the corpus, cluster it, and build centroid summary nodes."""
        self._items.extend(items)
        n = len(self._items)
        if n == 0:
            return
        self._leaf_emb = self._provider.embed([item.text for item in self._items])
        k = self._n_clusters or max(1, round(math.sqrt(n)))
        k = min(k, n)
        if n < 2 or k <= 1:
            labels = np.zeros(n, dtype=int)
            k = 1
        else:
            labels = AgglomerativeClustering(
                n_clusters=k, metric="cosine", linkage="average"
            ).fit_predict(self._leaf_emb)
        self._members = [list(np.nonzero(labels == c)[0]) for c in range(k)]
        self._summary_emb = np.array(
            [self._leaf_emb[idx].mean(axis=0) for idx in self._members], dtype=np.float64
        )

    def _ordered_item_indices(self, query_vec: NDArray[np.float64]) -> list[int]:
        assert self._leaf_emb is not None and self._summary_emb is not None
        # Rank the collapsed pool: leaves first (so a direct leaf hit outranks the
        # summary that contains it on ties), then summary nodes.
        pool = np.vstack([self._leaf_emb, self._summary_emb])
        sims = cosine_similarity_matrix(query_vec[None, :], pool)[0]
        n_leaves = self._leaf_emb.shape[0]
        order = np.argsort(-sims, kind="stable").tolist()

        leaf_sims = sims[:n_leaves]
        emitted: list[int] = []
        seen: set[int] = set()
        for node in order:
            if node < n_leaves:
                members = [node]
            else:
                # expand a summary node to its member leaves, best-first
                members = sorted(self._members[node - n_leaves], key=lambda i: -leaf_sims[i])
            for leaf in members:
                if leaf not in seen:
                    seen.add(leaf)
                    emitted.append(leaf)
        return emitted

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Return the collapsed-tree ranking (summaries expanded) packed to budget."""
        if not self._items or self._leaf_emb is None:
            return RetrievedContext.empty()
        query_vec = self._provider.embed_one(query)
        ranked = [
            (self._items[i].item_id, self._items[i].text)
            for i in self._ordered_item_indices(query_vec)
        ]
        return pack_to_budget(ranked, budget_tokens)
