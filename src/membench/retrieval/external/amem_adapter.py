r"""Offline protocol model of A-Mem as evaluated in the paper (tab:stdbench).

A-Mem (Xu et al., *A-Mem: Agentic Memory for LLM Agents*, NeurIPS 2025;
``\citep{xu2025amem}``) stores an *evolving note* per memory and links notes
*associatively*, in the Zettelkasten tradition: when a new note is written it is
connected to the existing note it most resembles, so recall can spread from a
query-relevant note to an associatively-linked one. The paper's landscape row
(tab:litcompare: "A-Mem -- evolving note -- associative (Zettelkasten) -- long-horizon
QA -- depth-aware? no") records exactly this mechanism, and marks it *not* depth-aware
because an associative link is descriptive resemblance, not a prescriptive governance
edge (Sections sec:litreview, sec:struct): it can hop to a *similar* note but does not
guarantee reaching the *governing* decision at depth d, so by Theorem thm:sim it stays
below the typed-traversal ``q^d`` guarantee (Theorem thm:struct).

This is an offline, deterministic stand-in for that mechanism -- not a re-implementation
of the A-Mem package and not a reproduction of the paper's measured tab:stdbench numbers
(those come from the full public-benchmark harness). Associative links are formed at
write time from the deterministic hashing-embedding cosine (each note links to its
single most similar earlier note), and retrieval seeds by that same cosine and then
follows one associative hop, so the arm is runnable with no network or API key.
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

__all__ = ["AMemMemory"]


@register_arm(
    "amem",
    family=ArmFamily.EXTERNAL,
    follows_links=True,
    uses_embeddings=True,
    requires_network=False,
    deterministic=True,
)
class AMemMemory(MemorySystem):
    """Zettelkasten-style associative note store (offline protocol model of A-Mem).

    Each note is linked, at write time, to the single earlier note it most resembles
    (an undirected associative edge). At retrieval the corpus is ranked by cosine
    similarity to the query, and each similarity seed is immediately followed by its
    associatively-linked neighbours before the next seed -- so a note connected to a
    query-relevant note surfaces even when it does not resemble the query itself.

    Parameters
    ----------
    provider
        Embedding backend for association-building and seeding; defaults to the
        deterministic offline hashing provider, so the arm needs no arguments.
    """

    def __init__(self, provider: EmbeddingProvider | None = None) -> None:
        self._provider = provider or HashingEmbeddingProvider()
        self._items: list[MemoryItem] = []
        self._matrix: NDArray[np.float64] | None = None
        self._links: dict[int, list[int]] = {}

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Embed the corpus and form each note's associative link to its nearest earlier note."""
        self._items.extend(items)
        n = len(self._items)
        if n == 0:
            return
        self._matrix = self._provider.embed([item.text for item in self._items])
        self._links = {i: [] for i in range(n)}
        if n < 2:
            return
        # Zettelkasten link formation: note i connects to the earlier note it most
        # resembles. argmax over the strictly-earlier prefix keeps the link acyclic in
        # time and fully determined by the (deterministic) embedding cosine.
        gram = cosine_similarity_matrix(self._matrix, self._matrix)
        for i in range(1, n):
            j = int(np.argmax(gram[i, :i]))
            if gram[i, j] > 0.0:
                self._links[i].append(j)
                self._links[j].append(i)

    def _ordered_indices(self, query_vec: NDArray[np.float64]) -> list[int]:
        assert self._matrix is not None
        sims = cosine_similarity_matrix(query_vec, self._matrix).ravel()
        seed_order = np.argsort(-sims, kind="stable").tolist()
        emitted: list[int] = []
        seen: set[int] = set()
        for seed in seed_order:
            if seed not in seen:
                seen.add(seed)
                emitted.append(seed)
            # Spread one associative hop from the seed, best-linked-first, so a note
            # reached only through the association ranks right behind its anchor.
            for neighbor in sorted(self._links.get(seed, ()), key=lambda k: -sims[k]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    emitted.append(neighbor)
        return emitted

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Seed by cosine similarity, follow one associative hop, then pack to budget."""
        if not self._items or self._matrix is None:
            return RetrievedContext.empty()
        query_vec = self._provider.embed_one(query)
        ranked = [
            (self._items[i].item_id, self._items[i].text) for i in self._ordered_indices(query_vec)
        ]
        return pack_to_budget(ranked, budget_tokens)
