"""Tests for the fielded BM25F lexical arm.

The golden values are hand-derived from the BM25F formula (Robertson, Zaragoza &
Taylor, 2004) using exact fractions, never by running the implementation under
test, so the assertions are an independent check of the algorithm.
"""

from __future__ import annotations

import math

import pytest

from membench.retrieval import bm25f  # noqa: F401  (registration side effect)
from membench.retrieval.base import ArmFamily, build_arm, get_spec
from membench.retrieval.bm25f import BM25FMemory, bm25_idf, bm25f_document_score
from membench.types import MemoryItem


class TestBM25FGolden:
    def test_two_doc_one_term_hand_computed(self) -> None:
        # --- Toy corpus: 2 documents, single query term "alpha", two fields ----
        # Field weights w_title = 2.0, w_body = 1.0 ; k1 = 1.2, b = 0.75.
        # Doc 1: title tf=1 len=2 ; body tf=2 len=10
        # Doc 2: title tf=0 len=3 ; body tf=1 len= 5
        # avg_title = (2 + 3) / 2 = 2.5 ; avg_body = (10 + 5) / 2 = 7.5
        # Both docs contain "alpha" -> n_t = 2, N = 2.
        # idf = ln(1 + (N - n_t + 0.5)/(n_t + 0.5)) = ln(1 + 0.5/2.5) = ln(1.2).
        field_weights = {"title": 2.0, "body": 1.0}
        avg = {"title": 2.5, "body": 7.5}
        doc_freqs = {"alpha": 2}
        k1, b, n_docs = 1.2, 0.75, 2
        idf = math.log(1.2)

        # Doc 1 by hand:
        #   B_title = 1 - 0.75 + 0.75*(2/2.5)  = 0.25 + 0.60 = 0.85
        #   B_body  = 1 - 0.75 + 0.75*(10/7.5) = 0.25 + 1.00 = 1.25
        #   x1 = 2.0*1/0.85 + 1.0*2/1.25 = 40/17 + 8/5 = 336/85
        #   k1 + x1 = 1.2 + 336/85 = 438/85
        #   x1/(k1+x1) = 336/438 = 56/73
        #   score1 = ln(1.2) * 56/73  ~= 0.139863
        expected1 = idf * (56 / 73)
        got1 = bm25f_document_score(
            ["alpha"],
            {"title": {"alpha": 1}, "body": {"alpha": 2}},
            {"title": 2, "body": 10},
            avg_field_lengths=avg,
            field_weights=field_weights,
            doc_freqs=doc_freqs,
            n_docs=n_docs,
            k1=k1,
            b=b,
        )
        assert got1 == pytest.approx(expected1)
        assert got1 == pytest.approx(0.139863, abs=1e-5)

        # Doc 2 by hand (term absent from title -> only the body contributes):
        #   B_body = 1 - 0.75 + 0.75*(5/7.5) = 0.25 + 0.50 = 0.75
        #   x2 = 1.0*1/0.75 = 4/3
        #   k1 + x2 = 1.2 + 4/3 = 7.6/3
        #   x2/(k1+x2) = (4/3)/(7.6/3) = 4/7.6 = 10/19
        #   score2 = ln(1.2) * 10/19 ~= 0.095959
        expected2 = idf * (10 / 19)
        got2 = bm25f_document_score(
            ["alpha"],
            {"title": {"alpha": 0}, "body": {"alpha": 1}},
            {"title": 3, "body": 5},
            avg_field_lengths=avg,
            field_weights=field_weights,
            doc_freqs=doc_freqs,
            n_docs=n_docs,
            k1=k1,
            b=b,
        )
        assert got2 == pytest.approx(expected2)
        assert got2 == pytest.approx(0.095959, abs=1e-5)

        # The doc whose title carries the term wins under the higher title weight.
        assert got1 > got2

    def test_single_field_collapses_to_standard_bm25(self) -> None:
        # A single field of weight 1 must reproduce standard BM25:
        #   BM25F single field: (tf/B) / (k1 + tf/B) = tf / (k1*B + tf)
        #   which equals textbook BM25 up to the rank-preserving (k1+1) numerator
        #   constant that BM25F drops. B = 1 - b + b*(dl/avgdl).
        k1, b = 1.5, 0.75
        tf, length, doc_freq, n_docs, avgdl = 3, 9, 4, 10, 6.0

        got = bm25f_document_score(
            ["alpha"],
            {"body": {"alpha": tf}},
            {"body": length},
            avg_field_lengths={"body": avgdl},
            field_weights={"body": 1.0},
            doc_freqs={"alpha": doc_freq},
            n_docs=n_docs,
            k1=k1,
            b=b,
        )
        idf = math.log(1 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        bigb = 1 - b + b * (length / avgdl)
        expected_bm25 = idf * tf / (k1 * bigb + tf)
        assert got == pytest.approx(expected_bm25)

    def test_idf_is_non_negative_for_common_term(self) -> None:
        # Term in every document -> classic RSJ idf would be negative; the +1 form
        # floors at ln(1 + 0.5/(N+0.5)) > 0.
        assert bm25_idf(10, 10) > 0.0
        # Rare term -> larger idf than a common one.
        assert bm25_idf(100, 1) > bm25_idf(100, 50)


_CORPUS = [
    MemoryItem("D-1", "audit logging policy\nall teams should follow standard procedures here"),
    MemoryItem("D-2", "general onboarding guidelines\nthe audit logging policy applies to teams"),
    MemoryItem("D-3", "date range picker\nuse the picker component for selecting calendar dates"),
]


def _arm() -> BM25FMemory:
    arm = build_arm("bm25f")
    assert isinstance(arm, BM25FMemory)
    arm.write(_CORPUS)
    return arm


class TestBM25FArm:
    def test_registered_as_lexical_no_embeddings(self) -> None:
        spec = get_spec("bm25f")
        assert spec.capabilities.family is ArmFamily.LEXICAL
        assert not spec.capabilities.uses_embeddings
        assert not spec.capabilities.follows_links

    def test_constructible_with_no_args(self) -> None:
        # The harness calls build_arm(name) with no kwargs.
        assert isinstance(build_arm("bm25f"), BM25FMemory)

    def test_title_match_outranks_body_match(self) -> None:
        # "audit logging policy" is the title of D-1 but only body text of D-2;
        # the higher title weight makes D-1 rank first.
        ctx = _arm().retrieve("audit logging policy", budget_tokens=10_000)
        assert ctx.ids[0] == "D-1"
        assert set(ctx.ids[:2]) == {"D-1", "D-2"}

    def test_deterministic(self) -> None:
        a = _arm().retrieve("audit logging policy", 10_000)
        b = _arm().retrieve("audit logging policy", 10_000)
        assert a.ids == b.ids

    def test_empty_corpus(self) -> None:
        assert build_arm("bm25f").retrieve("anything", 1000).ids == ()

    def test_empty_query_falls_back_to_corpus_order(self) -> None:
        ctx = _arm().retrieve("", budget_tokens=10_000)
        assert ctx.ids[0] == "D-1"  # corpus order, no crash

    def test_out_of_vocabulary_query_is_safe(self) -> None:
        ctx = _arm().retrieve("zzz nonexistent vocabulary", budget_tokens=10_000)
        # No term matches -> all-zero scores -> corpus order preserved.
        assert ctx.ids[0] == "D-1"

    def test_respects_budget(self) -> None:
        assert _arm().retrieve("audit", budget_tokens=0).ids == ()

    def test_flat_item_without_newline_is_all_body(self) -> None:
        # A single-line item has an empty body; it must still be scored and ranked.
        arm = build_arm("bm25f")
        assert isinstance(arm, BM25FMemory)
        arm.write([MemoryItem("F-1", "audit logging only one line")])
        ctx = arm.retrieve("audit logging", budget_tokens=10_000)
        assert ctx.ids == ("F-1",)
