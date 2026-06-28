"""Tests for the BM25+ lexical arm (Lv & Zhai 2011, lower-bounded TF).

Golden values are derived independently from the closed-form scoring function
(arithmetic shown inline), never by reading back this implementation's output.
"""

from __future__ import annotations

import math

import pytest

from membench.retrieval import bm25_plus  # noqa: F401  (registration side effect)
from membench.retrieval.base import ArmFamily, build_arm, get_spec
from membench.retrieval.bm25_plus import bm25_plus_scores
from membench.types import MemoryItem

_CORPUS = [
    MemoryItem("D-1", "all data exports must call the audit log for compliance"),
    MemoryItem("D-2", "use the date range picker for selecting dates in the interface"),
    MemoryItem("D-3", "every export endpoint requires an audit log entry"),
    MemoryItem("D-4", "primary actions use the button component"),
]


def _arm() -> object:
    arm = build_arm("bm25_plus")
    arm.write(_CORPUS)  # type: ignore[attr-defined]
    return arm


class TestStructure:
    def test_registered_as_lexical(self) -> None:
        spec = get_spec("bm25_plus")
        assert spec.capabilities.family is ArmFamily.LEXICAL
        assert not spec.capabilities.uses_embeddings
        assert not spec.capabilities.follows_links

    def test_constructible_with_no_args(self) -> None:
        # The harness calls build_arm(name) with no kwargs.
        assert build_arm("bm25_plus") is not None

    def test_ranks_lexically_relevant_first(self) -> None:
        ctx = _arm().retrieve("audit log on data export", 10_000)  # type: ignore[attr-defined]
        assert set(ctx.ids[:2]) == {"D-1", "D-3"}

    def test_deterministic(self) -> None:
        a = _arm().retrieve("audit export", 10_000).ids  # type: ignore[attr-defined]
        b = _arm().retrieve("audit export", 10_000).ids  # type: ignore[attr-defined]
        assert a == b

    def test_empty_corpus(self) -> None:
        assert build_arm("bm25_plus").retrieve("anything", 1000).ids == ()

    def test_empty_query_falls_back_to_corpus_order(self) -> None:
        ctx = _arm().retrieve("", 10_000)  # type: ignore[attr-defined]
        assert ctx.ids[0] == "D-1"  # corpus order, no crash

    def test_respects_budget(self) -> None:
        assert _arm().retrieve("audit", 0).ids == ()  # type: ignore[attr-defined]


# --- Golden value (one-document corpus, so |D| == avgdl) ---
# Corpus = a single doc with tokens ["alpha", "beta", "beta"]; query = ["beta"].
# Parameters k1 = 1.5, b = 0.75, delta = 1.
#   N = 1, df(beta) = 1  ->  idf = ln((N + 1) / df) = ln(2)
#   |D| = 3, avgdl = 3   ->  |D| / avgdl = 1
#   norm = k1 * (1 - b + b * 1) = 1.5 * 1 = 1.5
#   tf(beta) = 2  ->  ratio = (k1 + 1) * tf / (norm + tf) = 2.5 * 2 / 3.5 = 10/7
#   beta occurs in D, so the delta lower bound applies:
#   BM25+  = idf * (delta + ratio) = ln(2) * (1 + 10/7) = ln(2) * 17/7
#          = 0.6931471805599453 * 17 / 7 = 1.6833574385027244
#   BM25   = idf * ratio          = ln(2) * 10/7         = 0.9902102579427790
#   difference = BM25+ - BM25 = idf * delta = ln(2)       = 0.6931471805599453
_GOLDEN_DOC = [["alpha", "beta", "beta"]]
_GOLDEN_QUERY = ["beta"]

# --- Long-document lower bound ---
# D0 (long, relevant): "audit" once + 40 filler "x" tokens  ->  |D0| = 41.
# D1 (short, irrelevant): "button component"                ->  |D1| = 2.
# Query = ["audit"];  N = 2, df(audit) = 1  ->  idf = ln((2 + 1) / 1) = ln(3).
_LONG_CORPUS = [["audit", *(["x"] * 40)], ["button", "component"]]
_LONG_QUERY = ["audit"]


class TestGoldenValue:
    """Hand-computed golden for a one-document corpus (so |D| == avgdl)."""

    def test_bm25_plus_matches_hand_computed_value(self) -> None:
        score = bm25_plus_scores(_GOLDEN_DOC, _GOLDEN_QUERY, k1=1.5, b=0.75, delta=1.0)[0]
        assert score == pytest.approx(math.log(2.0) * 17.0 / 7.0)
        assert score == pytest.approx(1.6833574385027244)

    def test_delta_zero_reduces_to_bm25(self) -> None:
        bm25 = bm25_plus_scores(_GOLDEN_DOC, _GOLDEN_QUERY, k1=1.5, b=0.75, delta=0.0)[0]
        # Pure BM25 TF component: ln(2) * 10/7, no lower bound added.
        assert bm25 == pytest.approx(math.log(2.0) * 10.0 / 7.0)
        assert bm25 == pytest.approx(0.9902102579427790)

    def test_delta_adds_exactly_idf_times_delta(self) -> None:
        bm25 = bm25_plus_scores(_GOLDEN_DOC, _GOLDEN_QUERY, k1=1.5, b=0.75, delta=0.0)[0]
        plus = bm25_plus_scores(_GOLDEN_DOC, _GOLDEN_QUERY, k1=1.5, b=0.75, delta=1.0)[0]
        # The lower bound contributes exactly idf * delta = ln(2) * 1.
        assert plus - bm25 == pytest.approx(math.log(2.0))


class TestLongDocumentLowerBound:
    """The intended property: BM25+ rescues an over-penalized long relevant doc."""

    def test_long_relevant_doc_scores_higher_under_bm25_plus(self) -> None:
        bm25 = bm25_plus_scores(_LONG_CORPUS, _LONG_QUERY, delta=0.0)
        plus = bm25_plus_scores(_LONG_CORPUS, _LONG_QUERY, delta=1.0)
        # Same long relevant document, strictly higher under BM25+ than BM25.
        assert plus[0] > bm25[0]
        # The increment is exactly the lower bound idf * delta = ln(3) * 1.
        assert plus[0] - bm25[0] == pytest.approx(math.log(3.0))

    def test_lower_bound_guarantees_gap_over_non_containing_doc(self) -> None:
        plus = bm25_plus_scores(_LONG_CORPUS, _LONG_QUERY, delta=1.0)
        # The non-containing short doc gets nothing for "audit" (gated on presence).
        assert plus[1] == pytest.approx(0.0)
        # ...so the long relevant doc keeps a guaranteed margin of at least idf*delta,
        # which BM25's length normalization would otherwise drive toward zero.
        assert plus[0] - plus[1] >= math.log(3.0)
