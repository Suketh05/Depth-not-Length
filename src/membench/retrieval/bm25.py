"""BM25 lexical retrieval arm.

BM25 (Okapi) is the strongest classical lexical ranker and a standard, hard
baseline in information retrieval: a Brief win over BM25 cannot be dismissed as
beating a strawman keyword matcher. It ranks corpus items by term-frequency
saturation and inverse document frequency against the query, which makes it good
at recovering items that *share vocabulary* with the task — and, by the same
token, blind to a governing decision phrased in different words several hops away.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from rank_bm25 import BM25Okapi

from membench.retrieval._text import tokenize
from membench.retrieval.base import ArmFamily, MemorySystem, pack_to_budget, register_arm
from membench.types import MemoryItem, RetrievedContext

__all__ = ["BM25Memory"]


@register_arm("bm25", family=ArmFamily.LEXICAL, follows_links=False, uses_embeddings=False)
class BM25Memory(MemorySystem):
    """Rank corpus items by Okapi BM25 similarity to the query, then pack to budget."""

    def __init__(self) -> None:
        self._items: list[MemoryItem] = []
        self._tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Ingest the corpus and build the BM25 index over its tokenised texts."""
        self._items.extend(items)
        self._tokens = [tokenize(item.text) for item in self._items]
        # BM25Okapi needs a non-degenerate corpus (average doc length > 0).
        self._bm25 = BM25Okapi(self._tokens) if any(self._tokens) else None

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Return the BM25-ranked corpus packed into the budget."""
        if not self._items:
            return RetrievedContext.empty()
        query_tokens = tokenize(query)
        if self._bm25 is None or not query_tokens:
            # Degenerate corpus or query: fall back to corpus order rather than
            # crashing, so the arm always returns a well-defined ranking.
            order = range(len(self._items))
        else:
            scores = np.asarray(self._bm25.get_scores(query_tokens), dtype=np.float64)
            order = np.argsort(-scores, kind="stable").tolist()
        ranked = [(self._items[i].item_id, self._items[i].text) for i in order]
        return pack_to_budget(ranked, budget_tokens)
