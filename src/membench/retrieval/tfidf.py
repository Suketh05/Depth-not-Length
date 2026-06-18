"""TF-IDF cosine retrieval arm.

The other canonical lexical baseline alongside BM25: rank corpus items by the
cosine similarity of their TF-IDF vectors to the query's. Like BM25 it captures
surface vocabulary overlap, but with the classic linear TF-IDF weighting rather
than BM25's saturating term frequency, so reporting both shows the depth result is
not an artifact of one particular lexical scorer.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from membench.retrieval.base import ArmFamily, MemorySystem, pack_to_budget, register_arm
from membench.types import MemoryItem, RetrievedContext

__all__ = ["TfidfMemory"]


@register_arm("tfidf", family=ArmFamily.LEXICAL, follows_links=False, uses_embeddings=False)
class TfidfMemory(MemorySystem):
    """Rank corpus items by TF-IDF cosine similarity to the query, then pack to budget."""

    def __init__(self) -> None:
        self._items: list[MemoryItem] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix: object | None = None  # scipy sparse matrix (l2-normalised rows)

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Ingest the corpus and fit the TF-IDF vector space over its texts."""
        self._items.extend(items)
        if not self._items:
            return
        vectorizer = TfidfVectorizer()
        try:
            self._matrix = vectorizer.fit_transform(item.text for item in self._items)
            self._vectorizer = vectorizer
        except ValueError:
            # Empty vocabulary (all texts stop-words/blank): no usable space.
            self._vectorizer = None
            self._matrix = None

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Return the TF-IDF-ranked corpus packed into the budget."""
        if not self._items:
            return RetrievedContext.empty()
        if self._vectorizer is None or self._matrix is None:
            order: list[int] = list(range(len(self._items)))
        else:
            query_vec = self._vectorizer.transform([query])
            # Rows are l2-normalised by TfidfVectorizer, so dot product == cosine.
            scores = np.asarray((self._matrix @ query_vec.T).todense(), dtype=np.float64).ravel()
            order = np.argsort(-scores, kind="stable").tolist()
        ranked = [(self._items[i].item_id, self._items[i].text) for i in order]
        return pack_to_budget(ranked, budget_tokens)
