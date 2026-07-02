r"""Offline protocol model of ContextQ as evaluated in the paper (tab:stdbench).

ContextQ appears in the paper only as a competitor *product name* in the unified-harness
field (tab:stdbench, tab:tok_agent_economics); the paper states no internal mechanism for
it. We model it faithfully to what its name and evaluated profile imply -- a
*query-focused* context retriever that reformulates the query from its own first-pass
results, i.e. classical pseudo-relevance feedback (Lavrenko & Croft, *Relevance-Based
Language Models*, SIGIR 2001; the RM3 query-expansion family). This is an offline,
deterministic stand-in for that mechanism, **not** a reproduction of the vendor's internals
nor of the paper's measured tab:stdbench numbers (those come from the full public-benchmark
harness).

Mechanism-wise it is a *similarity* retriever with an expansion step, so by Theorem thm:sim
it remains resemblance-bounded and depth-decaying: expansion can bridge a vocabulary gap
within a hop but cannot follow a governance edge to the depth-d decision the way typed
traversal does (Theorem thm:struct). The arm is runnable with no network or API key.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from membench.retrieval._text import tokenize
from membench.retrieval.base import ArmFamily, MemorySystem, pack_to_budget, register_arm
from membench.types import MemoryItem, RetrievedContext

__all__ = ["ContextQMemory"]

_DEFAULT_FEEDBACK_DOCS = 3
_DEFAULT_EXPANSION_TERMS = 5
_DEFAULT_EXPANSION_WEIGHT = 0.5


@register_arm(
    "contextq",
    family=ArmFamily.LEXICAL,
    follows_links=False,
    uses_embeddings=False,
    requires_network=False,
    deterministic=True,
)
class ContextQMemory(MemorySystem):
    """Pseudo-relevance-feedback lexical retriever (offline protocol model of ContextQ).

    A first lexical pass scores each item by query-term frequency; the top feedback
    documents then contribute their most frequent terms as query expansions (down-weighted
    by ``expansion_weight``), and a second pass re-scores every item against the expanded
    query. Expansion lets an item that matches the *result-set* vocabulary but not the raw
    query surface -- the distinctive behaviour of relevance feedback.

    Parameters
    ----------
    feedback_docs
        Number of top first-pass items whose terms seed the expansion. Must be >= 1.
    expansion_terms
        Maximum number of expansion terms added to the query. Must be >= 0.
    expansion_weight
        Weight applied to each expansion term relative to an original query term
        (original terms carry weight 1.0). Must be in ``[0, 1]``.
    """

    def __init__(
        self,
        feedback_docs: int = _DEFAULT_FEEDBACK_DOCS,
        expansion_terms: int = _DEFAULT_EXPANSION_TERMS,
        expansion_weight: float = _DEFAULT_EXPANSION_WEIGHT,
    ) -> None:
        if feedback_docs < 1:
            raise ValueError(f"feedback_docs must be >= 1, got {feedback_docs}")
        if expansion_terms < 0:
            raise ValueError(f"expansion_terms must be >= 0, got {expansion_terms}")
        if not 0.0 <= expansion_weight <= 1.0:
            raise ValueError(f"expansion_weight must be in [0, 1], got {expansion_weight}")
        self._feedback_docs = feedback_docs
        self._expansion_terms = expansion_terms
        self._expansion_weight = expansion_weight
        self._items: list[MemoryItem] = []
        self._tfs: list[Counter[str]] = []

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Ingest the corpus and cache each item's term-frequency counts."""
        for item in items:
            self._items.append(item)
            self._tfs.append(Counter(tokenize(item.text)))

    def _score(self, weights: dict[str, float]) -> list[float]:
        """Weighted term-frequency score of every item against a (possibly expanded) query."""
        return [
            sum(weight * tf.get(term, 0) for term, weight in weights.items()) for tf in self._tfs
        ]

    def _expanded_weights(
        self, query_terms: list[str], first_pass: list[float]
    ) -> dict[str, float]:
        """Blend original query terms (weight 1) with feedback-derived expansion terms."""
        weights: dict[str, float] = dict.fromkeys(query_terms, 1.0)
        if self._expansion_terms == 0:
            return weights
        # Top feedback documents by first-pass score (stable by corpus order on ties).
        order = sorted(range(len(self._items)), key=lambda i: -first_pass[i])
        feedback = [i for i in order[: self._feedback_docs] if first_pass[i] > 0.0]
        pool: Counter[str] = Counter()
        for i in feedback:
            pool.update(self._tfs[i])
        for term in query_terms:  # never let an original term dilute its own weight
            pool.pop(term, None)
        for term, _ in pool.most_common(self._expansion_terms):
            weights[term] = self._expansion_weight
        return weights

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Run PRF (first pass, expand, re-score) and pack the re-ranked corpus to budget."""
        if not self._items:
            return RetrievedContext.empty()
        query_terms = tokenize(query)
        first_pass = self._score(dict.fromkeys(query_terms, 1.0))
        weights = self._expanded_weights(query_terms, first_pass)
        scores = self._score(weights)
        order = sorted(range(len(self._items)), key=lambda i: -scores[i])
        ranked = [(self._items[i].item_id, self._items[i].text) for i in order]
        return pack_to_budget(ranked, budget_tokens)
