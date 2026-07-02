r"""Offline protocol model of Oracle Summary as evaluated in the paper (tab:stdbench).

Oracle Summary appears in the paper only as a competitor *product name* in the unified-harness
field; the paper is explicit that "Oracle Summary is a competitor product name, not an
oracle/upper-bound" (caption of tab:stdbench and tab:t101pub). It is the lowest-token-cost
competitor in the economics table (tab:tok_agent_economics: 44767 session tokens, the field
minimum) and strong on ranking (tab:stdbench nDCG@10 0.928). Both are consistent with a
*summary* store, so we model it as an extractive-summary retriever: each memory is compressed
to its leading extractive summary at write time, and retrieval ranks and returns those
summaries. This is an offline, deterministic stand-in, **not** a reproduction of the vendor's
internals nor of the paper's measured tab:stdbench numbers (those come from the full
public-benchmark harness).

Compression is the distinctive behaviour: the assembled context is strictly shorter than the
concatenated raw memories, which is why more items fit a fixed budget (the low token cost) at
the risk of dropping a detail the summary elided. It ranks by resemblance with no link to
follow, so by Theorem thm:sim it remains depth-decaying below the typed-traversal ``q^d``
guarantee (Theorem thm:struct). The arm is runnable with no network or API key.
"""

from __future__ import annotations

import re
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

__all__ = ["OracleSummaryMemory", "extractive_summary"]

_DEFAULT_MAX_SUMMARY_CHARS = 160
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def extractive_summary(text: str, max_chars: int = _DEFAULT_MAX_SUMMARY_CHARS) -> str:
    """Return the leading extractive summary of ``text`` (first sentence, capped to ``max_chars``).

    The first sentence is taken as the summary (a standard lead-based extractive baseline);
    if it still exceeds ``max_chars`` it is hard-truncated. The result is never longer than the
    input, so assembling summaries can only shrink the context.
    """
    if max_chars < 0:
        raise ValueError(f"max_chars must be >= 0, got {max_chars}")
    lead = _SENTENCE_SPLIT.split(text.strip(), maxsplit=1)[0] if text.strip() else ""
    return lead[:max_chars]


@register_arm(
    "oracle_summary",
    family=ArmFamily.EXTERNAL,
    follows_links=False,
    uses_embeddings=True,
    requires_network=False,
    deterministic=True,
)
class OracleSummaryMemory(MemorySystem):
    """Extractive-summary store (offline protocol model of Oracle Summary).

    ``write`` compresses each memory to its leading extractive summary and embeds the *full*
    text for ranking; ``retrieve`` orders items by cosine similarity to the query and returns
    the compressed summaries, packed to budget. Because each returned text is a summary, the
    assembled context is strictly shorter than the raw memories would be.

    Parameters
    ----------
    max_summary_chars
        Hard cap on each item's summary length. Must be >= 0.
    provider
        Embedding backend for ranking; defaults to the deterministic offline hashing provider.
    """

    def __init__(
        self,
        max_summary_chars: int = _DEFAULT_MAX_SUMMARY_CHARS,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        if max_summary_chars < 0:
            raise ValueError(f"max_summary_chars must be >= 0, got {max_summary_chars}")
        self._max_summary_chars = max_summary_chars
        self._provider = provider or HashingEmbeddingProvider()
        self._items: list[MemoryItem] = []
        self._summaries: list[str] = []
        self._matrix: NDArray[np.float64] | None = None

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Ingest the corpus, compressing each item to its summary and embedding the full text."""
        for item in items:
            self._items.append(item)
            self._summaries.append(extractive_summary(item.text, self._max_summary_chars))
        if self._items:
            self._matrix = self._provider.embed([item.text for item in self._items])

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Rank by cosine similarity and pack the compressed summaries to budget."""
        if not self._items or self._matrix is None:
            return RetrievedContext.empty()
        query_vec = self._provider.embed_one(query)
        sims = cosine_similarity_matrix(query_vec, self._matrix).ravel()
        order = np.argsort(-sims, kind="stable").tolist()
        ranked = [(self._items[i].item_id, self._summaries[i]) for i in order]
        return pack_to_budget(ranked, budget_tokens)
