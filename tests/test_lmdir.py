"""Tests for the query-likelihood language-model arms (Dirichlet & Jelinek-Mercer).

The golden log query-likelihood values below are hand-derived from the formulae in
Zhai & Lafferty (2001), not produced by running the arm, so the assertions are an
independent check of the implementation.

Tiny collection (tokenisation is lowercase-alphanumeric, identical to BM25/TF-IDF):

    D-1 = "alpha alpha beta"   ->  tf: alpha=2, beta=1   |d1| = 3
    D-2 = "alpha beta beta"    ->  tf: alpha=1, beta=2   |d2| = 3

Collection: |C| = 6, cf(alpha) = cf(beta) = 3, so p(alpha|C) = p(beta|C) = 0.5.

Query  q = "alpha alpha beta"  ->  q-tf: alpha=2, beta=1.

Dirichlet smoothing, mu = 2,  p(w|d) = (tf + mu*p(w|C)) / (|d| + mu):
    p(alpha|d1) = (2 + 2*0.5)/(3+2) = 3/5 = 0.6
    p(beta |d1) = (1 + 2*0.5)/(3+2) = 2/5 = 0.4
    p(alpha|d2) = (1 + 2*0.5)/(3+2) = 2/5 = 0.4
    p(beta |d2) = (2 + 2*0.5)/(3+2) = 3/5 = 0.6
    score(q,d1) = 2*ln(0.6) + 1*ln(0.4) = -1.9379419794061366
    score(q,d2) = 2*ln(0.4) + 1*ln(0.6) = -2.3434070875143007
    => D-1 outranks D-2.

Jelinek-Mercer smoothing, lambda = 0.5,  p(w|d) = (1-l)*tf/|d| + l*p(w|C):
    p(alpha|d1) = 0.5*(2/3) + 0.5*0.5 = 7/12;  p(beta|d1) = 0.5*(1/3) + 0.5*0.5 = 5/12
    p(alpha|d2) = 0.5*(1/3) + 0.5*0.5 = 5/12;  p(beta|d2) = 0.5*(2/3) + 0.5*0.5 = 7/12
    score(q,d1) = 2*ln(7/12) + 1*ln(5/12) = -1.9534617388192737
    score(q,d2) = 2*ln(5/12) + 1*ln(7/12) = -2.2899339754404866
    => D-1 outranks D-2.
"""

from __future__ import annotations

import math

import pytest

from membench.retrieval import lmdir  # noqa: F401  (registration side effect)
from membench.retrieval.base import ArmFamily, build_arm, get_spec
from membench.retrieval.lmdir import (
    DirichletLMMemory,
    JelinekMercerLMMemory,
    QueryLikelihoodMemory,
)
from membench.types import MemoryItem

_CORPUS = [
    MemoryItem("D-1", "alpha alpha beta"),
    MemoryItem("D-2", "alpha beta beta"),
]
_QUERY = "alpha alpha beta"


class TestRegistration:
    def test_dirichlet_registered_as_lexical(self) -> None:
        spec = get_spec("qlm_dirichlet")
        assert spec.capabilities.family is ArmFamily.LEXICAL
        assert not spec.capabilities.follows_links
        assert not spec.capabilities.uses_embeddings

    def test_jm_registered_as_lexical(self) -> None:
        spec = get_spec("qlm_jm")
        assert spec.capabilities.family is ArmFamily.LEXICAL
        assert not spec.capabilities.uses_embeddings

    def test_built_with_no_kwargs(self) -> None:
        # The harness calls build_arm(name) with no kwargs; must not raise.
        for name in ("qlm_dirichlet", "qlm_jm"):
            arm = build_arm(name)
            arm.write(_CORPUS)  # type: ignore[attr-defined]
            ctx = arm.retrieve(_QUERY, budget_tokens=10_000)  # type: ignore[attr-defined]
            assert set(ctx.ids) == {"D-1", "D-2"}


class TestDirichletGolden:
    def test_scores_match_hand_computed_values(self) -> None:
        arm = QueryLikelihoodMemory(smoothing="dirichlet", mu=2.0)
        arm.write(_CORPUS)
        scores = arm.score_documents(_QUERY)
        # Golden: probabilities derived by hand above; math.log is the reference.
        expected_d1 = 2 * math.log(0.6) + math.log(0.4)  # -1.9379419794061366
        expected_d2 = 2 * math.log(0.4) + math.log(0.6)  # -2.3434070875143007
        assert scores["D-1"] == pytest.approx(expected_d1)
        assert scores["D-2"] == pytest.approx(expected_d2)
        assert scores["D-1"] == pytest.approx(-1.9379419794061366)
        assert scores["D-2"] == pytest.approx(-2.3434070875143007)

    def test_ranks_higher_likelihood_first(self) -> None:
        arm = DirichletLMMemory(mu=2.0)
        arm.write(_CORPUS)
        ctx = arm.retrieve(_QUERY, budget_tokens=10_000)
        assert ctx.ids == ("D-1", "D-2")


class TestJelinekMercerGolden:
    def test_scores_match_hand_computed_values(self) -> None:
        arm = QueryLikelihoodMemory(smoothing="jelinek_mercer", jm_lambda=0.5)
        arm.write(_CORPUS)
        scores = arm.score_documents(_QUERY)
        expected_d1 = 2 * math.log(7 / 12) + math.log(5 / 12)  # -1.9534617388192737
        expected_d2 = 2 * math.log(5 / 12) + math.log(7 / 12)  # -2.2899339754404866
        assert scores["D-1"] == pytest.approx(expected_d1)
        assert scores["D-2"] == pytest.approx(expected_d2)
        assert scores["D-1"] == pytest.approx(-1.9534617388192737)
        assert scores["D-2"] == pytest.approx(-2.2899339754404866)

    def test_ranks_higher_likelihood_first(self) -> None:
        arm = JelinekMercerLMMemory(jm_lambda=0.5)
        arm.write(_CORPUS)
        ctx = arm.retrieve(_QUERY, budget_tokens=10_000)
        assert ctx.ids == ("D-1", "D-2")


class TestStructuralInvariants:
    def test_empty_corpus_returns_empty(self) -> None:
        arm = build_arm("qlm_dirichlet")
        assert arm.retrieve("anything", 1000).ids == ()

    def test_out_of_vocabulary_terms_are_skipped(self) -> None:
        # "gamma" is absent from the collection; only "alpha" (q-tf 1) contributes.
        arm = QueryLikelihoodMemory(smoothing="dirichlet", mu=2.0)
        arm.write(_CORPUS)
        scores = arm.score_documents("alpha gamma")
        assert scores["D-1"] == pytest.approx(math.log(0.6))
        assert scores["D-2"] == pytest.approx(math.log(0.4))

    def test_fully_oov_query_falls_back_to_corpus_order(self) -> None:
        arm = DirichletLMMemory(mu=2.0)
        arm.write(_CORPUS)
        ctx = arm.retrieve("gamma delta", budget_tokens=10_000)
        assert ctx.ids == ("D-1", "D-2")  # corpus order, no crash, no -inf

    def test_empty_query_falls_back_to_corpus_order(self) -> None:
        arm = DirichletLMMemory(mu=2.0)
        arm.write(_CORPUS)
        ctx = arm.retrieve("", budget_tokens=10_000)
        assert ctx.ids == ("D-1", "D-2")

    def test_respects_budget(self) -> None:
        arm = DirichletLMMemory(mu=2.0)
        arm.write(_CORPUS)
        assert arm.retrieve(_QUERY, budget_tokens=0).ids == ()

    def test_deterministic(self) -> None:
        a = DirichletLMMemory(mu=2.0)
        a.write(_CORPUS)
        b = DirichletLMMemory(mu=2.0)
        b.write(_CORPUS)
        assert a.retrieve(_QUERY, 10_000).ids == b.retrieve(_QUERY, 10_000).ids

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError, match="mu"):
            QueryLikelihoodMemory(smoothing="dirichlet", mu=-1.0)
        with pytest.raises(ValueError, match="jm_lambda"):
            QueryLikelihoodMemory(smoothing="jelinek_mercer", jm_lambda=1.5)
        with pytest.raises(ValueError, match="smoothing"):
            QueryLikelihoodMemory(smoothing="bogus")  # type: ignore[arg-type]
