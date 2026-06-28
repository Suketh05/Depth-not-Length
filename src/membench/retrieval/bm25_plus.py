r"""BM25+ lexical retrieval arm (lower-bounded term-frequency normalization).

BM25+ augments Okapi BM25 with a constant lower bound ``delta`` on the normalized
term-frequency component, fixing the well-documented flaw that BM25's length
normalization *over-penalizes long documents*: as a document grows, the saturated
TF term shrinks toward zero, so a long document that genuinely contains a query
term can be driven to score arbitrarily close to a document that does not contain
it at all. Adding ``delta`` for every query term that occurs in the document
guarantees a non-vanishing gap between "contains the term" and "does not", which
is exactly the robustness property Lv & Zhai prove.

For a query ``Q`` and document ``D`` the score is

.. math::

    S(Q, D) = \sum_{t \in Q \cap D} \mathrm{idf}(t)
        \left( \delta +
        \frac{(k_1 + 1)\, \mathrm{tf}_{t,D}}
             {k_1 \big(1 - b + b\, |D| / \mathrm{avgdl}\big) + \mathrm{tf}_{t,D}}
        \right),

with :math:`\mathrm{idf}(t) = \ln\!\big((N + 1) / \mathrm{df}_t\big)` (the
non-negative IDF form used in the BM25+ paper, where :math:`N` is the corpus size
and :math:`\mathrm{df}_t` the document frequency of ``t``). Setting
:math:`\delta = 0` recovers standard BM25 exactly; the paper recommends
:math:`\delta = 1`.

Reference
---------
Yuanhua Lv and ChengXiang Zhai. "Lower-Bounding Term Frequency Normalization for
Robust BM25 Retrieval." In *Proceedings of the 20th ACM International Conference
on Information and Knowledge Management (CIKM '11)*, 2011, pp. 7-16.
https://doi.org/10.1145/2063576.2063584
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from membench.retrieval._text import tokenize
from membench.retrieval.base import ArmFamily, MemorySystem, pack_to_budget, register_arm
from membench.types import MemoryItem, RetrievedContext

__all__ = ["BM25PlusMemory", "bm25_plus_scores"]

K1_DEFAULT = 1.5
"""Default term-frequency saturation parameter (standard BM25 value)."""

B_DEFAULT = 0.75
"""Default length-normalization strength (standard BM25 value)."""

DELTA_DEFAULT = 1.0
"""Default TF lower bound; ``delta = 1`` is the value recommended by Lv & Zhai."""


def bm25_plus_scores(
    corpus_tokens: Sequence[Sequence[str]],
    query_tokens: Sequence[str],
    *,
    k1: float = K1_DEFAULT,
    b: float = B_DEFAULT,
    delta: float = DELTA_DEFAULT,
) -> NDArray[np.float64]:
    """Compute the BM25+ score of every document for a tokenized query.

    The lower bound ``delta`` is added only for query terms that actually occur in
    a document (the ``t in Q intersect D`` sum of Lv & Zhai 2011), which is what
    creates the guaranteed score gap between documents that contain a term and
    those that do not. With ``delta = 0`` this reduces exactly to Okapi BM25 using
    the non-negative ``log((N + 1) / df)`` IDF.

    Parameters
    ----------
    corpus_tokens
        One token list per document, in corpus order.
    query_tokens
        The tokenized query. Repeated query terms contribute multiple times
        (query term frequency), matching the BM25 summation over query terms.
    k1
        Term-frequency saturation parameter (``>= 0``).
    b
        Length-normalization strength in ``[0, 1]``.
    delta
        Lower bound added to the normalized TF component of each matching term.

    Returns
    -------
    numpy.ndarray
        A float64 array of length ``len(corpus_tokens)`` of BM25+ scores, aligned
        with the corpus order. An empty (or all-empty) corpus yields zeros.
    """
    n_docs = len(corpus_tokens)
    scores = np.zeros(n_docs, dtype=np.float64)
    if n_docs == 0:
        return scores

    doc_len = np.asarray([len(doc) for doc in corpus_tokens], dtype=np.float64)
    avgdl = float(doc_len.mean())
    if avgdl == 0.0:
        # Degenerate corpus: every document is empty, so nothing can match.
        return scores

    counters = [Counter(doc) for doc in corpus_tokens]
    # Length-normalized denominator term k1 * (1 - b + b * |D| / avgdl), per doc.
    norm = k1 * (1.0 - b + b * doc_len / avgdl)

    for term in query_tokens:
        df = sum(1 for counter in counters if term in counter)
        if df == 0:
            continue
        idf = math.log((n_docs + 1) / df)
        tf = np.asarray([counter.get(term, 0) for counter in counters], dtype=np.float64)
        present = tf > 0.0
        ratio = (k1 + 1.0) * tf / (norm + tf)
        scores += idf * (np.where(present, delta, 0.0) + ratio)
    return scores


@register_arm("bm25_plus", family=ArmFamily.LEXICAL, follows_links=False, uses_embeddings=False)
class BM25PlusMemory(MemorySystem):
    """Rank corpus items by BM25+ (lower-bounded TF) similarity, then pack to budget."""

    def __init__(
        self,
        k1: float = K1_DEFAULT,
        b: float = B_DEFAULT,
        delta: float = DELTA_DEFAULT,
    ) -> None:
        self._k1 = k1
        self._b = b
        self._delta = delta
        self._items: list[MemoryItem] = []
        self._tokens: list[list[str]] = []

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Ingest the corpus and tokenize its texts for BM25+ scoring."""
        self._items.extend(items)
        self._tokens = [tokenize(item.text) for item in self._items]

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Return the BM25+-ranked corpus packed into the budget."""
        if not self._items:
            return RetrievedContext.empty()
        query_tokens = tokenize(query)
        if not query_tokens or not any(self._tokens):
            # Degenerate query or corpus: fall back to corpus order rather than
            # crashing, so the arm always returns a well-defined ranking.
            order: list[int] = list(range(len(self._items)))
        else:
            scores = bm25_plus_scores(
                self._tokens, query_tokens, k1=self._k1, b=self._b, delta=self._delta
            )
            order = np.argsort(-scores, kind="stable").tolist()
        ranked = [(self._items[i].item_id, self._items[i].text) for i in order]
        return pack_to_budget(ranked, budget_tokens)
