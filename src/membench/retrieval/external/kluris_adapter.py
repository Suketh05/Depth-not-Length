r"""Offline protocol model of Kluris as evaluated in the paper (tab:stdbench).

Kluris appears in the paper only as a competitor *product name* in the unified-harness field,
where it is the strongest non-Brief system on DMR fact F1 (tab:stdbench, 87.0, cited in
fig:stdbench as Brief 94.2 vs Kluris 87.0). The paper states no internal mechanism for it.
Its strength on *fact retrieval* is consistent with an entity/keyword store, so we model it as
a set-overlap (Jaccard) retriever over the extracted token sets of the query and each item.
This is an offline, deterministic stand-in, **not** a reproduction of the vendor's internals
nor of the paper's measured tab:stdbench numbers (those come from the full public-benchmark
harness).

Jaccard rewards containing the query's distinct terms while penalising unrelated bulk (the
union in the denominator), which is why a compact fact record outranks a long passage that
merely happens to include the same terms. It is still a *resemblance* score with no link to
follow, so by Theorem thm:sim it remains depth-decaying below the typed-traversal ``q^d``
guarantee (Theorem thm:struct). The arm is runnable with no network or API key.
"""

from __future__ import annotations

from collections.abc import Iterable

from membench.retrieval._text import tokenize
from membench.retrieval.base import ArmFamily, MemorySystem, pack_to_budget, register_arm
from membench.types import MemoryItem, RetrievedContext

__all__ = ["KlurisMemory", "jaccard_overlap"]


def jaccard_overlap(query_terms: set[str], doc_terms: set[str]) -> float:
    r"""Jaccard set overlap ``|intersection| / |union|`` of two token sets (0.0 if both empty)."""
    if not query_terms and not doc_terms:
        return 0.0
    union = query_terms | doc_terms
    return len(query_terms & doc_terms) / len(union)


@register_arm(
    "kluris",
    family=ArmFamily.LEXICAL,
    follows_links=False,
    uses_embeddings=False,
    requires_network=False,
    deterministic=True,
)
class KlurisMemory(MemorySystem):
    """Entity/keyword set-overlap (Jaccard) store (offline protocol model of Kluris).

    ``write`` records each item's distinct token set; ``retrieve`` ranks items by the Jaccard
    overlap of that set with the query's token set, then packs to budget. Because the union
    sits in the denominator, an item that covers the query's terms without unrelated bulk
    scores higher than a longer item covering the same terms -- the behaviour of a compact
    fact/entity index.
    """

    def __init__(self) -> None:
        self._items: list[MemoryItem] = []
        self._term_sets: list[set[str]] = []

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Ingest the corpus and record each item's distinct token set."""
        for item in items:
            self._items.append(item)
            self._term_sets.append(set(tokenize(item.text)))

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Rank corpus items by Jaccard overlap with the query and pack to budget."""
        if not self._items:
            return RetrievedContext.empty()
        query_terms = set(tokenize(query))
        scores = [jaccard_overlap(query_terms, terms) for terms in self._term_sets]
        order = sorted(range(len(self._items)), key=lambda i: -scores[i])
        ranked = [(self._items[i].item_id, self._items[i].text) for i in order]
        return pack_to_budget(ranked, budget_tokens)
