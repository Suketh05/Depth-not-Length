r"""Deterministic, offline SPLADE-style learned-sparse retrieval arm.

SPLADE (SParse Lexical AnD Expansion model) represents a query or document as a
sparse weight vector over the vocabulary in which (a) the importance of each term
is *saturated* through a log transform, and (b) a *term-expansion* step assigns
non-zero weight to terms that never appear literally in the text but are predicted
to be relevant, so that vocabulary-mismatch between query and document can be
bridged. The relevance score is the dot product of the two sparse vectors. See

    T. Formal, B. Piwowarski and S. Clinchant, "SPLADE: Sparse Lexical and
    Expansion Model for First Stage Ranking", *Proc. 44th Int. ACM SIGIR Conf.*
    (SIGIR '21), pp. 2288-2292, 2021. doi:10.1145/3404835.3463098.

The published model derives expansion from a masked-language-model (BERT MLM)
head: ``w_{j} = \max_{i \in t} \log\bigl(1 + \mathrm{ReLU}(s_{ij})\bigr)`` where
``s_{ij}`` is the MLM logit of vocabulary term ``j`` at input position ``i``.

Honest approximation
--------------------
This arm is **deterministic and offline**: it uses *no neural model*. It keeps the
two structural ideas of SPLADE while replacing the learned MLM expansion with a
transparent, corpus-derived co-occurrence statistic:

* **Saturated term importance** (faithful to SPLADE): a literally present term
  ``t`` with raw frequency ``tf`` receives weight ``log(1 + tf)``.
* **Co-occurrence expansion** (the approximation): an absent term ``e`` receives
  weight ``beta * sum_{t in text} w_t * P(e | t)`` where ``w_t = log(1 + tf_t)``
  and ``P(e | t) = cooc(t, e) / df(t)`` is the empirical document-level
  probability of seeing ``e`` given ``t`` (``cooc`` = number of corpus documents
  containing both terms, ``df`` = document frequency). ``beta`` discounts expanded
  terms relative to literal matches. This is a first-order, linear stand-in for
  the MLM's contextual term prediction: it is interpretable and reproducible but
  cannot capture sub-word semantics the way a trained MLM head does.

Both queries and documents are mapped through the *same* transform, fitted on the
written corpus, and ranked by sparse dot product -- exactly SPLADE's first-stage
scoring rule.

Formula
-------
For a piece of text with term frequencies ``tf_t``::

    base_t      = log(1 + tf_t)                              for t present
    expand_e    = beta * sum_{t present, df(t) > 0}
                       base_t * cooc(t, e) / df(t)           for e absent
    vector[k]   = base_k                  if k is present
                = beta * expand_k         if k is absent and expand_k > 0

    score(q, d) = sum_k q_vector[k] * d_vector[k]            (sparse dot product)
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

from membench.retrieval._text import tokenize
from membench.retrieval.base import ArmFamily, MemorySystem, pack_to_budget, register_arm
from membench.types import MemoryItem, RetrievedContext

__all__ = ["SpladeLiteMemory"]


def _sparse_dot(left: dict[str, float], right: dict[str, float]) -> float:
    """Return the dot product of two sparse term-weight vectors.

    Parameters
    ----------
    left, right
        Sparse vectors mapping a term to its weight; missing keys are zero.

    Returns
    -------
    float
        ``sum_k left[k] * right[k]`` over the shared support (iterate the smaller
        vector so the cost is proportional to the smaller support).
    """
    if len(right) < len(left):
        left, right = right, left
    return sum(weight * right[term] for term, weight in left.items() if term in right)


@register_arm(
    "splade_lite",
    family=ArmFamily.LEXICAL,
    follows_links=False,
    uses_embeddings=False,
)
class SpladeLiteMemory(MemorySystem):
    """Rank items by a deterministic SPLADE-style sparse expansion, then pack to budget."""

    #: Discount applied to expanded (non-literal) terms relative to literal matches.
    EXPANSION_FACTOR: float = 0.5

    def __init__(self) -> None:
        """Create an empty arm; corpus statistics are fitted in :meth:`write`."""
        self._items: list[MemoryItem] = []
        self._doc_freq: Counter[str] = Counter()
        self._cooc: dict[str, Counter[str]] = {}
        self._doc_vectors: list[dict[str, float]] = []

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Ingest the corpus, fit document-frequency and co-occurrence, expand each item.

        Parameters
        ----------
        items
            The task's memory corpus. Co-occurrence is counted at document level
            (a pair contributes once per document that contains both terms).
        """
        self._items.extend(items)
        if not self._items:
            return
        self._doc_freq = Counter()
        self._cooc = {}
        for item in self._items:
            terms = sorted(set(tokenize(item.text)))
            for term in terms:
                self._doc_freq[term] += 1
                self._cooc.setdefault(term, Counter())
            for i, term_a in enumerate(terms):
                for term_b in terms[i + 1 :]:
                    self._cooc[term_a][term_b] += 1
                    self._cooc[term_b][term_a] += 1
        self._doc_vectors = [self.sparse_vector(item.text) for item in self._items]

    def sparse_vector(self, text: str) -> dict[str, float]:
        """Map text to its SPLADE-style sparse term-weight vector (saturation + expansion).

        Parameters
        ----------
        text
            The query or document text to encode. Encoding uses the corpus
            statistics fitted in :meth:`write`; before any corpus is written the
            expansion term is empty and only the saturated literal weights remain.

        Returns
        -------
        dict of str to float
            Sparse vector mapping each (literal or expanded) term to its weight.
        """
        term_freq = Counter(tokenize(text))
        if not term_freq:
            return {}
        base = {term: math.log1p(count) for term, count in term_freq.items()}
        present = set(base)

        expansion: dict[str, float] = {}
        for term, weight in base.items():
            df = self._doc_freq.get(term, 0)
            if df == 0:
                continue
            for other, cooc in self._cooc.get(term, {}).items():
                if other in present:
                    continue
                expansion[other] = expansion.get(other, 0.0) + weight * cooc / df

        vector = dict(base)
        for term, raw in expansion.items():
            vector[term] = self.EXPANSION_FACTOR * raw
        return vector

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Return the sparse-dot-product-ranked corpus packed into the budget.

        Parameters
        ----------
        query
            The natural-language query, encoded with the same transform as the
            documents.
        budget_tokens
            Shared token budget passed to :func:`pack_to_budget`.

        Returns
        -------
        RetrievedContext
            Items ranked by descending sparse dot product (ties keep corpus order)
            and packed greedily into ``budget_tokens``.
        """
        if not self._items:
            return RetrievedContext.empty()
        query_vec = self.sparse_vector(query)
        scores = [_sparse_dot(query_vec, doc_vec) for doc_vec in self._doc_vectors]
        # Round the score in the sort key so mathematically-tied documents whose
        # dot products differ only by floating-point round-off (BLAS/NumPy summation
        # order varies across platforms/Python versions) collapse to an exact tie and
        # are broken deterministically by corpus index ``i``.
        order = sorted(range(len(self._items)), key=lambda i: (-round(scores[i], 12), i))
        ranked = [(self._items[i].item_id, self._items[i].text) for i in order]
        return pack_to_budget(ranked, budget_tokens)
