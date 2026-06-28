r"""Query-likelihood language-model retrieval arms (Dirichlet & Jelinek-Mercer).

The query-likelihood model ranks each document ``d`` by the probability that its
estimated language model would generate the query ``q``. Scoring uses the log
query-likelihood, summed over query term occurrences::

    score(q, d) = sum_{w in q} tf(w, q) * log p(w | d)

where ``p(w | d)`` is a *smoothed* unigram estimate. Smoothing is essential: an
unsmoothed maximum-likelihood estimate assigns zero probability to any query term
absent from ``d``, collapsing the whole product to zero. Two standard smoothing
schemes are implemented, both blending the document estimate with the collection
model ``p(w | C) = cf(w) / |C|`` (``cf`` = collection frequency, ``|C|`` = total
tokens in the collection):

* **Dirichlet** (a Bayesian prior with concentration ``mu``)::

      p(w | d) = (tf(w, d) + mu * p(w | C)) / (|d| + mu)

* **Jelinek-Mercer** (a fixed linear interpolation controlled by ``lambda``)::

      p(w | d) = (1 - lambda) * tf(w, d) / |d| + lambda * p(w | C)

Both schemes, the log query-likelihood scoring rule, and the recommended
parameter ranges (Dirichlet ``mu`` ~ 2000; Jelinek-Mercer ``lambda`` ~ 0.1 for
short/title queries) are from:

    Chengxiang Zhai and John Lafferty. "A Study of Smoothing Methods for Language
    Models Applied to Ad Hoc Information Retrieval." SIGIR 2001, pp. 334-342.
    (Also TOIS 22(2):179-214, 2004.)

Out-of-vocabulary query terms — those with ``cf(w) = 0`` — carry no collection
signal and would force ``log 0 = -inf`` under both schemes, so they are skipped;
this is the standard treatment of unseen query terms in the query-likelihood
model. Tokenisation and collection statistics reuse :func:`membench.retrieval._text.tokenize`,
identical to the BM25 and TF-IDF lexical arms.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from typing import Literal

from membench.retrieval._text import tokenize
from membench.retrieval.base import ArmFamily, MemorySystem, pack_to_budget, register_arm
from membench.types import MemoryItem, RetrievedContext

__all__ = [
    "DEFAULT_JM_LAMBDA",
    "DEFAULT_MU",
    "DirichletLMMemory",
    "JelinekMercerLMMemory",
    "QueryLikelihoodMemory",
    "Smoothing",
]

Smoothing = Literal["dirichlet", "jelinek_mercer"]

DEFAULT_MU = 2000.0
"""Dirichlet concentration recommended by Zhai & Lafferty (2001) for ad hoc IR."""

DEFAULT_JM_LAMBDA = 0.1
"""Jelinek-Mercer interpolation weight recommended for short/title queries."""


class QueryLikelihoodMemory(MemorySystem):
    """Rank corpus items by smoothed log query-likelihood, then pack to budget.

    Parameters
    ----------
    smoothing
        ``"dirichlet"`` for Bayesian Dirichlet-prior smoothing or
        ``"jelinek_mercer"`` for fixed linear interpolation with the collection
        model.
    mu
        Dirichlet concentration ``mu >= 0`` (used only when
        ``smoothing == "dirichlet"``). Larger values pull document estimates
        further toward the collection model.
    jm_lambda
        Jelinek-Mercer weight ``0 <= lambda <= 1`` on the collection model (used
        only when ``smoothing == "jelinek_mercer"``).

    Notes
    -----
    See the module docstring for the formulae and the Zhai & Lafferty (2001)
    reference. The arm is purely lexical: it follows no links and uses no
    embeddings.
    """

    def __init__(
        self,
        smoothing: Smoothing = "dirichlet",
        mu: float = DEFAULT_MU,
        jm_lambda: float = DEFAULT_JM_LAMBDA,
    ) -> None:
        if smoothing not in ("dirichlet", "jelinek_mercer"):
            raise ValueError(f"unknown smoothing {smoothing!r}")
        if mu < 0:
            raise ValueError(f"mu must be non-negative, got {mu}")
        if not 0.0 <= jm_lambda <= 1.0:
            raise ValueError(f"jm_lambda must be in [0, 1], got {jm_lambda}")
        self._smoothing: Smoothing = smoothing
        self._mu = float(mu)
        self._jm_lambda = float(jm_lambda)
        self._items: list[MemoryItem] = []
        self._doc_counts: list[Counter[str]] = []
        self._doc_lengths: list[int] = []
        self._collection_counts: Counter[str] = Counter()
        self._collection_length = 0

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Ingest the corpus and accumulate per-document and collection statistics."""
        for item in items:
            tokens = tokenize(item.text)
            counts = Counter(tokens)
            self._items.append(item)
            self._doc_counts.append(counts)
            self._doc_lengths.append(len(tokens))
            self._collection_counts.update(counts)
            self._collection_length += len(tokens)

    def _term_prob(self, word: str, doc_index: int) -> float:
        """Return the smoothed estimate ``p(word | document)`` for one document.

        Parameters
        ----------
        word
            A query term known to occur in the collection (``cf(word) > 0``).
        doc_index
            Index of the document whose language model is evaluated.

        Returns
        -------
        float
            The smoothed probability ``p(word | d)`` under the configured scheme.
        """
        tf = self._doc_counts[doc_index].get(word, 0)
        doc_len = self._doc_lengths[doc_index]
        p_collection = self._collection_counts[word] / self._collection_length
        if self._smoothing == "dirichlet":
            return (tf + self._mu * p_collection) / (doc_len + self._mu)
        ml = tf / doc_len if doc_len > 0 else 0.0
        return (1.0 - self._jm_lambda) * ml + self._jm_lambda * p_collection

    def score_documents(self, query: str) -> dict[str, float]:
        """Return the log query-likelihood of ``query`` under each document model.

        Parameters
        ----------
        query
            The free-text query to score against every ingested document.

        Returns
        -------
        dict of str to float
            Maps each ``item_id`` to ``sum_w tf(w, q) * log p(w | d)``, summed over
            in-vocabulary query terms (out-of-vocabulary terms are skipped). An
            empty mapping is returned for an empty corpus.
        """
        query_counts = Counter(tokenize(query))
        scores: dict[str, float] = {}
        for doc_index, item in enumerate(self._items):
            total = 0.0
            for word, q_tf in query_counts.items():
                if self._collection_counts.get(word, 0) == 0:
                    continue  # out-of-vocabulary: no collection signal, skip.
                total += q_tf * math.log(self._term_prob(word, doc_index))
            scores[item.item_id] = total
        return scores

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Return the query-likelihood-ranked corpus packed into the budget."""
        if not self._items:
            return RetrievedContext.empty()
        scores = self.score_documents(query)
        query_tokens = tokenize(query)
        if not query_tokens or all(
            self._collection_counts.get(word, 0) == 0 for word in query_tokens
        ):
            # Empty or fully out-of-vocabulary query: fall back to corpus order
            # rather than returning an arbitrary tie-break, mirroring the BM25 arm.
            order: list[int] = list(range(len(self._items)))
        else:
            # Higher log-likelihood ranks first; Python's stable sort keeps corpus
            # order among ties.
            order = sorted(
                range(len(self._items)),
                key=lambda i: scores[self._items[i].item_id],
                reverse=True,
            )
        ranked = [(self._items[i].item_id, self._items[i].text) for i in order]
        return pack_to_budget(ranked, budget_tokens)


@register_arm("qlm_dirichlet", family=ArmFamily.LEXICAL, follows_links=False, uses_embeddings=False)
class DirichletLMMemory(QueryLikelihoodMemory):
    """Query-likelihood ranking with Dirichlet-prior smoothing (Zhai & Lafferty 2001)."""

    def __init__(self, mu: float = DEFAULT_MU) -> None:
        super().__init__(smoothing="dirichlet", mu=mu)


@register_arm("qlm_jm", family=ArmFamily.LEXICAL, follows_links=False, uses_embeddings=False)
class JelinekMercerLMMemory(QueryLikelihoodMemory):
    """Query-likelihood ranking with Jelinek-Mercer smoothing (Zhai & Lafferty 2001)."""

    def __init__(self, jm_lambda: float = DEFAULT_JM_LAMBDA) -> None:
        super().__init__(smoothing="jelinek_mercer", jm_lambda=jm_lambda)
