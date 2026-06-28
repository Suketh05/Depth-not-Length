"""RM3 pseudo-relevance-feedback query expansion arm.

RM3 is the standard *relevance-model* query-expansion method. It treats the top
``feedback_docs`` returned by a first-pass retriever as pseudo-relevant, estimates
a relevance language model from them, and interpolates that model with the
original query before re-ranking the corpus. It is a faithful lexical baseline:
expansion can only pull in terms that *co-occur* with query vocabulary in the
feedback set, so — like BM25 and TF-IDF — it cannot recover a governing decision
phrased in disjoint words several hops away.

Algorithm (Lavrenko & Croft, *Relevance-Based Language Models*, SIGIR 2001;
RM3 interpolation as in Abdul-Jaleel et al., *UMass at TREC 2004: Novelty and
HARD*, and Lavrenko, *A Generative Theory of Relevance*, 2009)
------------------------------------------------------------------------------
1. Run an initial retrieval and keep the top ``F`` feedback documents.
2. Weight each feedback document by its query posterior, assuming a uniform
   document prior so that ``p(d | q)`` is proportional to ``p(q | d)`` (the query
   likelihood),

       p(q | d) = prod_{w in q} p_lambda(w | d) ^ c(w, q),

   then normalise ``p(d | q)`` over the feedback set ``F``.
3. Estimate the relevance model (RM1) by mixing the documents' maximum-likelihood
   term distributions, weighted by the posterior:

       p(w | R) = sum_{d in F} p(d | q) * p_mle(w | d),
       p_mle(w | d) = c(w, d) / |d|.

   Keep the ``expansion_terms`` highest-probability terms and renormalise.
4. Interpolate with the maximum-likelihood query model (this is the RM3 step):

       p'(w) = alpha * p_mle(w | q) + (1 - alpha) * p(w | R).

   ``alpha = 1`` recovers the original query exactly (no expansion);
   ``alpha = 0`` is the pure relevance model (RM1).
5. Re-rank the corpus by the negative KL divergence of the expanded query model
   from each document's Jelinek-Mercer-smoothed model (Zhai & Lafferty, 2001),
   which is rank-equivalent to the cross-entropy

       score(d) = sum_w p'(w) * log p_lambda(w | d),
       p_lambda(w | d) = (1 - lambda) * c(w, d) / |d| + lambda * p(w | C),

   and pack the ranking into the shared token budget.

Smoothing (``p_lambda``) is used only for the query-likelihood weighting (step 2)
and
the final re-ranking (step 5); the relevance model itself (step 3) mixes the
unsmoothed maximum-likelihood document models, as in the original derivation.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

from membench.retrieval._text import tokenize
from membench.retrieval.base import ArmFamily, MemorySystem, pack_to_budget, register_arm
from membench.retrieval.bm25 import BM25Memory
from membench.types import MemoryItem, RetrievedContext

__all__ = ["RM3Memory"]

# Budget large enough to force the first-stage retriever to emit its full ranking,
# from which the top ``feedback_docs`` are taken as pseudo-relevant.
_FULL_RANKING_BUDGET = 1_000_000_000


@register_arm("rm3", family=ArmFamily.LEXICAL, follows_links=False, uses_embeddings=False)
class RM3Memory(MemorySystem):
    """Expand the query with an RM3 relevance model, then re-rank and pack to budget.

    Parameters
    ----------
    initial
        First-pass retriever whose top results seed the relevance model. Defaults
        to :class:`~membench.retrieval.bm25.BM25Memory`, a standard lexical arm.
    feedback_docs
        Number of top-ranked documents treated as pseudo-relevant (the ``fbDocs``
        parameter). Must be positive.
    expansion_terms
        Number of highest-probability relevance-model terms retained before
        interpolation (the ``fbTerms`` parameter). Must be positive.
    alpha
        Interpolation weight on the original query model in ``[0, 1]``. ``1.0``
        leaves the query unchanged; lower values trust the relevance model more.
    jm_lambda
        Jelinek-Mercer smoothing weight on the collection model in ``[0, 1]``,
        used for the query-likelihood document weighting and the final re-ranking.

    Notes
    -----
    The arm is constructible with no arguments so the harness can build it via
    ``build_arm("rm3")``. All state is rebuilt in :meth:`write`, so a fresh
    instance is created per task and never leaks between tasks.
    """

    def __init__(
        self,
        initial: MemorySystem | None = None,
        feedback_docs: int = 10,
        expansion_terms: int = 10,
        alpha: float = 0.5,
        jm_lambda: float = 0.5,
    ) -> None:
        if feedback_docs <= 0:
            raise ValueError(f"feedback_docs must be positive, got {feedback_docs}")
        if expansion_terms <= 0:
            raise ValueError(f"expansion_terms must be positive, got {expansion_terms}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not 0.0 <= jm_lambda <= 1.0:
            raise ValueError(f"jm_lambda must be in [0, 1], got {jm_lambda}")
        self._initial: MemorySystem = initial if initial is not None else BM25Memory()
        self._feedback_docs = feedback_docs
        self._expansion_terms = expansion_terms
        self._alpha = alpha
        self._jm_lambda = jm_lambda
        self._items: list[MemoryItem] = []
        self._text_by_id: dict[str, str] = {}
        self._doc_counts: dict[str, Counter[str]] = {}
        self._doc_len: dict[str, int] = {}
        self._collection_counts: Counter[str] = Counter()
        self._collection_len = 0

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Ingest the corpus into the first-pass retriever and the term statistics."""
        materialised = list(items)
        self._initial.write(materialised)
        for item in materialised:
            if item.item_id in self._text_by_id:
                continue
            self._items.append(item)
            self._text_by_id[item.item_id] = item.text
            counts = Counter(tokenize(item.text))
            self._doc_counts[item.item_id] = counts
            self._doc_len[item.item_id] = sum(counts.values())
            self._collection_counts.update(counts)
        self._collection_len = sum(self._collection_counts.values())

    def _smoothed(self, term: str, doc_id: str) -> float:
        """Return the Jelinek-Mercer-smoothed probability ``p_lambda(term | doc_id)``."""
        doc_len = self._doc_len[doc_id]
        doc_p = self._doc_counts[doc_id][term] / doc_len if doc_len else 0.0
        coll_p = (
            self._collection_counts[term] / self._collection_len if self._collection_len else 0.0
        )
        return (1.0 - self._jm_lambda) * doc_p + self._jm_lambda * coll_p

    def _feedback_posteriors(
        self, query_counts: Counter[str], doc_ids: list[str]
    ) -> dict[str, float]:
        """Return ``p(d | q)`` over the feedback set via softmax of the query log-likelihood."""
        log_likelihoods: dict[str, float] = {}
        for doc_id in doc_ids:
            total = 0.0
            for term, count in query_counts.items():
                prob = self._smoothed(term, doc_id)
                # A query term absent from both the document and the collection
                # carries no discriminative signal; skip it rather than emit -inf.
                if prob > 0.0:
                    total += count * math.log(prob)
            log_likelihoods[doc_id] = total
        top = max(log_likelihoods.values())
        weights = {doc_id: math.exp(ll - top) for doc_id, ll in log_likelihoods.items()}
        normaliser = sum(weights.values())
        return {doc_id: weight / normaliser for doc_id, weight in weights.items()}

    def relevance_model(self, query: str) -> dict[str, float]:
        """Estimate the truncated, renormalised RM1 relevance model ``p(w | R)``.

        Parameters
        ----------
        query
            The raw query string.

        Returns
        -------
        dict of str to float
            The top ``expansion_terms`` relevance-model terms and their
            probabilities, renormalised to sum to one. Empty when there are no
            feedback documents or no query terms.
        """
        query_counts = Counter(tokenize(query))
        if not query_counts or not self._items:
            return {}
        feedback_ids = list(self._initial.retrieve(query, _FULL_RANKING_BUDGET).ids)[
            : self._feedback_docs
        ]
        if not feedback_ids:
            return {}
        posteriors = self._feedback_posteriors(query_counts, feedback_ids)
        model: dict[str, float] = {}
        for doc_id, posterior in posteriors.items():
            doc_len = self._doc_len[doc_id]
            if not doc_len:
                continue
            for term, count in self._doc_counts[doc_id].items():
                model[term] = model.get(term, 0.0) + posterior * (count / doc_len)
        # Keep the highest-probability terms (ties broken by term for determinism).
        ranked = sorted(model.items(), key=lambda pair: (-pair[1], pair[0]))
        kept = dict(ranked[: self._expansion_terms])
        total = sum(kept.values())
        if total <= 0.0:
            return {}
        return {term: weight / total for term, weight in kept.items()}

    def expanded_query_model(self, query: str) -> dict[str, float]:
        """Return the RM3 expanded query model ``p'(w)`` after interpolation.

        Parameters
        ----------
        query
            The raw query string.

        Returns
        -------
        dict of str to float
            Terms mapped to their interpolated probabilities
            ``alpha * p_mle(w | q) + (1 - alpha) * p(w | R)``, restricted to
            strictly positive weights. With ``alpha == 1.0`` this is exactly the
            maximum-likelihood query model.
        """
        query_counts = Counter(tokenize(query))
        query_len = sum(query_counts.values())
        query_model = (
            {term: count / query_len for term, count in query_counts.items()} if query_len else {}
        )
        relevance = self.relevance_model(query) if self._alpha < 1.0 else {}
        expanded: dict[str, float] = {}
        for term in set(query_model) | set(relevance):
            weight = self._alpha * query_model.get(term, 0.0) + (1.0 - self._alpha) * relevance.get(
                term, 0.0
            )
            if weight > 0.0:
                expanded[term] = weight
        return expanded

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Re-rank the corpus by the expanded query model and pack into the budget."""
        if not self._items:
            return RetrievedContext.empty()
        expanded = self.expanded_query_model(query)
        if not expanded:
            # No usable query/expansion model: defer to the first-pass ranking.
            return self._initial.retrieve(query, budget_tokens)
        scores: list[tuple[float, int, str]] = []
        for index, item in enumerate(self._items):
            score = 0.0
            for term, weight in expanded.items():
                prob = self._smoothed(term, item.item_id)
                if prob > 0.0:
                    score += weight * math.log(prob)
            scores.append((score, index, item.item_id))
        # Sort by descending score; original index breaks ties for a stable order.
        scores.sort(key=lambda triple: (-triple[0], triple[1]))
        ranked = [(item_id, self._text_by_id[item_id]) for _, _, item_id in scores]
        return pack_to_budget(ranked, budget_tokens)
