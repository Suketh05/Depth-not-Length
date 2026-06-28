"""Tests for the ColBERT-style late-interaction (MaxSim) retrieval arm.

The golden value is derived by hand from the MaxSim definition (Khattab & Zaharia,
SIGIR 2020), not by running the implementation:

    S(q, d) = sum_i  max_j  cos(E_q_i, E_d_j)

Token embeddings are L2-normalised unit vectors, so a token matched against an
identical token scores cos(v, v) = 1.0. For the toy query "alpha beta" and the
document "alpha beta gamma", every query token has an exact match in the document:

    query token "alpha": max cosine = cos("alpha", "alpha") = 1.0
    query token "beta" : max cosine = cos("beta",  "beta")  = 1.0
    S = 1.0 + 1.0 = 2.0

So the hand-computed golden is exactly 2.0 (the number of query tokens, because the
document covers all of them). A document missing those tokens cannot reach 1.0 on
any query token and therefore scores strictly below 2.0 and ranks lower.
"""

from __future__ import annotations

import numpy as np
import pytest

from membench.retrieval.base import build_arm
from membench.retrieval.colbert_lite import ColbertLiteMemory, maxsim_score
from membench.retrieval.embeddings import HashingEmbeddingProvider
from membench.types import MemoryItem

# Hand-derived golden (see module docstring): query "alpha beta" fully covered.
GOLDEN_MAXSIM_FULL_COVERAGE = 2.0

QUERY = "alpha beta"
DOC_COVERING = MemoryItem(item_id="cover", text="alpha beta gamma")
DOC_DISJOINT = MemoryItem(item_id="other", text="delta epsilon zeta")


def _embed_tokens(provider: HashingEmbeddingProvider, text: str) -> np.ndarray:
    return provider.embed(text.split())


def test_maxsim_equals_hand_computed_full_coverage_golden() -> None:
    """MaxSim of a fully-covering doc equals the hand-computed sum of per-token maxima."""
    provider = HashingEmbeddingProvider()
    query_vecs = _embed_tokens(provider, QUERY)
    doc_vecs = _embed_tokens(provider, DOC_COVERING.text)

    score = maxsim_score(query_vecs, doc_vecs)

    assert score == pytest.approx(GOLDEN_MAXSIM_FULL_COVERAGE)


def test_per_query_token_maximum_is_one_on_exact_match() -> None:
    """Each query token's best document cosine is exactly 1.0 (it matches itself)."""
    provider = HashingEmbeddingProvider()
    query_vecs = _embed_tokens(provider, QUERY)
    doc_vecs = _embed_tokens(provider, DOC_COVERING.text)

    # (n_query_tokens, n_doc_tokens) cosine matrix; max over doc tokens per query token.
    sims = (query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)) @ (
        doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    ).T
    per_token_max = sims.max(axis=1)

    assert per_token_max == pytest.approx(np.array([1.0, 1.0]))


def test_covering_doc_outscores_and_outranks_disjoint_doc() -> None:
    """A doc covering the query scores 2.0 and ranks above a disjoint doc (< 2.0)."""
    provider = HashingEmbeddingProvider()
    query_vecs = _embed_tokens(provider, QUERY)

    covering = maxsim_score(query_vecs, _embed_tokens(provider, DOC_COVERING.text))
    disjoint = maxsim_score(query_vecs, _embed_tokens(provider, DOC_DISJOINT.text))

    assert covering == pytest.approx(GOLDEN_MAXSIM_FULL_COVERAGE)
    assert disjoint < covering

    arm = ColbertLiteMemory()
    arm.write([DOC_DISJOINT, DOC_COVERING])
    result = arm.retrieve(QUERY, budget_tokens=100)

    assert result.ids[0] == DOC_COVERING.item_id
    assert set(result.ids) == {DOC_COVERING.item_id, DOC_DISJOINT.item_id}


def test_empty_bag_scores_zero() -> None:
    """An empty query or document bag contributes nothing (score 0.0)."""
    provider = HashingEmbeddingProvider()
    query_vecs = _embed_tokens(provider, QUERY)
    empty = np.empty((0, provider.dim), dtype=np.float64)

    assert maxsim_score(empty, query_vecs) == 0.0
    assert maxsim_score(query_vecs, empty) == 0.0


def test_arm_constructible_with_no_args_via_registry() -> None:
    """The harness builds the arm by name with no kwargs (build_arm contract)."""
    arm = build_arm("colbert_lite")

    assert isinstance(arm, ColbertLiteMemory)


def test_empty_corpus_returns_empty_context() -> None:
    """Retrieval before any write yields an empty context."""
    arm = ColbertLiteMemory()

    result = arm.retrieve(QUERY, budget_tokens=100)

    assert result.ids == ()
    assert result.text == ""


def test_retrieval_respects_token_budget() -> None:
    """Packing honours the shared budget: a zero budget admits no items."""
    arm = ColbertLiteMemory()
    arm.write([DOC_COVERING, DOC_DISJOINT])

    result = arm.retrieve(QUERY, budget_tokens=0)

    assert result.ids == ()
