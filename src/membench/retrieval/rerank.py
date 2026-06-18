"""Two-stage retrieve-then-rerank arm.

The standard high-precision RAG pattern: a cheap first-stage retriever proposes
candidates, then a more expensive *cross-encoder* that reads the query and a
candidate together re-scores them. Cross-encoders model query-document interaction
directly and are the strongest single-stage relevance signal short of a graph
walk.

To stay reproducible offline, the default reranker is a deterministic lexical
*interaction* feature (token Jaccard between query and candidate) — a faithful
stand-in for "score the pair jointly". A real neural cross-encoder
(sentence-transformers ``CrossEncoder``) plugs into the same interface behind the
optional ``neural`` extra. Either way the arm still cannot follow an explicit link,
so it remains a similarity method subject to the depth ceiling.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from membench.retrieval._text import tokenize
from membench.retrieval.base import ArmFamily, MemorySystem, pack_to_budget, register_arm
from membench.retrieval.dense import DenseMemory
from membench.types import MemoryItem, RetrievedContext

__all__ = ["CrossEncoderReranker", "LexicalOverlapReranker", "RerankMemory", "Reranker"]

_FULL_RANKING_BUDGET = 1_000_000_000


@runtime_checkable
class Reranker(Protocol):
    """Scores a (query, document) pair jointly; higher is more relevant."""

    def score(self, query: str, document: str) -> float:
        """Return a relevance score for the pair."""
        ...


class LexicalOverlapReranker:
    """Deterministic interaction feature: token Jaccard of query and document."""

    def score(self, query: str, document: str) -> float:
        """Return the Jaccard overlap of the query and document token sets."""
        q = set(tokenize(query))
        d = set(tokenize(document))
        if not q or not d:
            return 0.0
        return len(q & d) / len(q | d)


class CrossEncoderReranker:
    """Neural cross-encoder reranker (requires the optional ``neural`` extra)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "CrossEncoderReranker needs the 'neural' extra: uv sync --extra neural"
            ) from exc
        self._model = CrossEncoder(model_name)

    def score(self, query: str, document: str) -> float:
        """Return the cross-encoder relevance score for the pair."""
        return float(self._model.predict([(query, document)])[0])


@register_arm("rerank_ce", family=ArmFamily.HYBRID, follows_links=False, uses_embeddings=True)
class RerankMemory(MemorySystem):
    """Retrieve first-stage candidates, rerank them jointly, then pack to budget.

    Parameters
    ----------
    first_stage
        Candidate retriever (defaults to the dense arm).
    reranker
        Pairwise scorer (defaults to the deterministic lexical-overlap reranker).
    candidates
        Number of first-stage candidates to rerank.
    """

    def __init__(
        self,
        first_stage: MemorySystem | None = None,
        reranker: Reranker | None = None,
        candidates: int = 20,
    ) -> None:
        if candidates <= 0:
            raise ValueError(f"candidates must be positive, got {candidates}")
        self._first_stage = first_stage or DenseMemory()
        self._reranker: Reranker = reranker or LexicalOverlapReranker()
        self._candidates = candidates
        self._text_by_id: dict[str, str] = {}

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Ingest the corpus into the first-stage retriever and the id->text index."""
        materialised = list(items)
        self._first_stage.write(materialised)
        for item in materialised:
            self._text_by_id.setdefault(item.item_id, item.text)

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Return the reranked first-stage candidates packed into the budget."""
        if not self._text_by_id:
            return RetrievedContext.empty()
        candidate_ids = self._first_stage.retrieve(query, _FULL_RANKING_BUDGET).ids[
            : self._candidates
        ]
        scored = [
            (cid, self._reranker.score(query, self._text_by_id[cid])) for cid in candidate_ids
        ]
        # Stable sort by descending score keeps first-stage order for ties.
        scored.sort(key=lambda pair: -pair[1])
        ranked = [(cid, self._text_by_id[cid]) for cid, _ in scored]
        return pack_to_budget(ranked, budget_tokens)
