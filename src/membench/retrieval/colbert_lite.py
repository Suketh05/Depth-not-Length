r"""ColBERT-style late-interaction (MaxSim) retrieval arm.

Late interaction represents a query and a document not as single pooled vectors
but as *bags of per-token embeddings*, and scores them with the MaxSim operator:
each query token is softly matched against its single best-matching document
token, and those per-token maxima are summed. This preserves fine-grained term
matching that a single pooled vector washes out, while staying a pure
embedding-similarity method (no link following, no rerank model).

For a query with token embeddings :math:`\{E_{q_1}, \dots, E_{q_n}\}` and a
document with token embeddings :math:`\{E_{d_1}, \dots, E_{d_m}\}`, the score is

.. math::

    S(q, d) = \sum_{i=1}^{n} \max_{j=1}^{m} \; E_{q_i} \cdot E_{d_j},

where the token embeddings are L2-normalised so that the inner product equals
cosine similarity (the MaxSim formulation of Eq. 1 in the reference). The arm
ranks documents by :math:`S(q, d)` and hands the ranking to the shared
budget-packer, exactly like every other arm.

Because each query token contributes the cosine of its *best* document token, a
document that literally contains a query token scores ``1.0`` for that token (the
token's embedding matches itself), so a document containing every query token
scores ``n`` — the late-interaction analogue of exact coverage.

This is an *offline, deterministic* reduction of the architecture: token
embeddings come from the repository's deterministic feature-hashing provider
(:class:`~membench.retrieval.embeddings.HashingEmbeddingProvider`) rather than a
contextual BERT encoder, so there is no network, no model download, and identical
scores across processes. The late-interaction *scoring* is the faithful part; the
token encoder is the offline stand-in.

References
----------
O. Khattab and M. Zaharia. "ColBERT: Efficient and Effective Passage Search via
Contextualized Late Interaction over BERT." In *Proceedings of the 43rd
International ACM SIGIR Conference on Research and Development in Information
Retrieval (SIGIR '20)*, 2020, pp. 39-48. doi:10.1145/3397271.3401075.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from membench.retrieval._text import tokenize
from membench.retrieval.base import (
    ArmFamily,
    MemorySystem,
    pack_to_budget,
    register_arm,
)
from membench.retrieval.embeddings import (
    EmbeddingProvider,
    HashingEmbeddingProvider,
    cosine_similarity_matrix,
)
from membench.types import MemoryItem, RetrievedContext

__all__ = ["ColbertLiteMemory", "maxsim_score"]


def maxsim_score(
    query_token_vecs: NDArray[np.float64],
    doc_token_vecs: NDArray[np.float64],
) -> float:
    r"""Compute the ColBERT MaxSim score between two bags of token embeddings.

    Implements :math:`S = \sum_i \max_j E_{q_i} \cdot E_{d_j}` using cosine
    similarity between every query/document token pair. An empty query or document
    contributes nothing and yields ``0.0``.

    Parameters
    ----------
    query_token_vecs
        ``(n_query_tokens, dim)`` array of query token embeddings.
    doc_token_vecs
        ``(n_doc_tokens, dim)`` array of document token embeddings.

    Returns
    -------
    float
        The MaxSim late-interaction score (sum of per-query-token maxima).
    """
    if query_token_vecs.shape[0] == 0 or doc_token_vecs.shape[0] == 0:
        return 0.0
    sims = cosine_similarity_matrix(query_token_vecs, doc_token_vecs)
    return float(sims.max(axis=1).sum())


@register_arm(
    "colbert_lite",
    family=ArmFamily.DENSE,
    follows_links=False,
    uses_embeddings=True,
)
class ColbertLiteMemory(MemorySystem):
    """Rank corpus items by ColBERT MaxSim late interaction, then pack to budget.

    Parameters
    ----------
    provider
        Token embedding backend; defaults to the deterministic offline hashing
        provider so the arm runs with no network and identical scores across
        processes.
    """

    def __init__(self, provider: EmbeddingProvider | None = None) -> None:
        self._provider: EmbeddingProvider = provider or HashingEmbeddingProvider()
        self._items: list[MemoryItem] = []
        self._token_vecs: list[NDArray[np.float64]] = []

    def _embed_tokens(self, text: str) -> NDArray[np.float64]:
        """Embed each word token of ``text`` into an ``(n_tokens, dim)`` array."""
        tokens = tokenize(text)
        if not tokens:
            return np.empty((0, self._provider.dim), dtype=np.float64)
        return self._provider.embed(tokens)

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Ingest the corpus and embed every item's tokens for late interaction."""
        for item in items:
            self._items.append(item)
            self._token_vecs.append(self._embed_tokens(item.text))

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Return the MaxSim-ranked corpus packed into the budget."""
        if not self._items:
            return RetrievedContext.empty()
        query_vecs = self._embed_tokens(query)
        scores = np.array(
            [maxsim_score(query_vecs, doc_vecs) for doc_vecs in self._token_vecs],
            dtype=np.float64,
        )
        order = np.argsort(-scores, kind="stable").tolist()
        ranked = [(self._items[i].item_id, self._items[i].text) for i in order]
        return pack_to_budget(ranked, budget_tokens)
