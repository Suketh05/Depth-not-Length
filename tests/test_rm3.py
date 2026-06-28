"""Tests for the RM3 pseudo-relevance-feedback expansion arm.

The golden values are hand-derived (not produced by running the code under test),
so the assertions pin the textbook RM3 arithmetic rather than the implementation's
own output.

Setup
-----
Query ``"alpha beta"`` over a four-document corpus in which only ``D-1`` shares any
vocabulary with the query. With ``feedback_docs=1`` the single pseudo-relevant
document is ``D-1`` and its query posterior is ``p(D-1 | q) = 1`` (softmax over one
element), so the relevance model is exactly ``D-1``'s maximum-likelihood term
distribution and no smoothing constants enter the hand computation.

``D-1`` = ``"alpha gamma gamma delta"`` -> tokens ``[alpha, gamma, gamma, delta]``,
length 4, so

    p_mle(alpha | D-1) = 1/4 = 0.25
    p_mle(gamma | D-1) = 2/4 = 0.50   <- non-query term, strongly co-occurring
    p_mle(delta | D-1) = 1/4 = 0.25

RM1 keeps all three terms (expansion_terms >> 3) and they already sum to 1, so the
relevance model is unchanged by truncation/renormalisation.

Maximum-likelihood query model: ``p_mle(alpha | q) = p_mle(beta | q) = 0.5``.

RM3 interpolation p'(w) = alpha * p_mle(w | q) + (1 - alpha) * p(w | R) with
alpha = 0.5:

    alpha: 0.5 * 0.5 + 0.5 * 0.25 = 0.375
    beta : 0.5 * 0.5 + 0.5 * 0.00 = 0.250
    gamma: 0.5 * 0.0 + 0.5 * 0.50 = 0.250   <- GOLDEN: expansion term weight
    delta: 0.5 * 0.0 + 0.5 * 0.25 = 0.125
    (sum = 1.0, a proper distribution)

With alpha = 1.0 the expanded model collapses to the query model {alpha, beta}.
"""

from __future__ import annotations

import math

from membench.retrieval import rm3  # noqa: F401  (registration side effect)
from membench.retrieval.base import ArmFamily, build_arm, get_spec
from membench.retrieval.rm3 import RM3Memory
from membench.types import MemoryItem

_CORPUS = [
    MemoryItem("D-1", "alpha gamma gamma delta"),
    MemoryItem("D-2", "epsilon zeta eta"),
    MemoryItem("D-3", "theta iota kappa"),
    MemoryItem("D-4", "lambda mu nu"),
]


def _arm(**kwargs: object) -> RM3Memory:
    arm = RM3Memory(feedback_docs=1, **kwargs)  # type: ignore[arg-type]
    arm.write(_CORPUS)
    return arm


class TestRM3Registration:
    def test_registered_as_lexical(self) -> None:
        spec = get_spec("rm3")
        assert spec.capabilities.family is ArmFamily.LEXICAL
        assert not spec.capabilities.uses_embeddings
        assert not spec.capabilities.follows_links

    def test_buildable_with_no_kwargs(self) -> None:
        # The harness calls build_arm(name) with no kwargs; defaults must suffice.
        arm = build_arm("rm3")
        arm.write(_CORPUS)  # type: ignore[attr-defined]
        ctx = arm.retrieve("alpha beta", budget_tokens=10_000)  # type: ignore[attr-defined]
        assert ctx.ids[0] == "D-1"


class TestRM3RelevanceModel:
    def test_relevance_model_matches_doc_mle(self) -> None:
        rm = _arm().relevance_model("alpha beta")
        # p(D-1 | q) = 1 -> relevance model == D-1's MLE term distribution.
        assert math.isclose(rm["alpha"], 0.25, abs_tol=1e-9)
        assert math.isclose(rm["gamma"], 0.50, abs_tol=1e-9)
        assert math.isclose(rm["delta"], 0.25, abs_tol=1e-9)
        assert "beta" not in rm  # not present in the feedback document
        assert math.isclose(sum(rm.values()), 1.0, abs_tol=1e-9)


class TestRM3Expansion:
    def test_non_query_term_enters_with_golden_weight(self) -> None:
        expanded = _arm(alpha=0.5).expanded_query_model("alpha beta")
        # GOLDEN (hand-computed in the module docstring):
        assert math.isclose(expanded["gamma"], 0.25, abs_tol=1e-9)
        assert math.isclose(expanded["alpha"], 0.375, abs_tol=1e-9)
        assert math.isclose(expanded["beta"], 0.25, abs_tol=1e-9)
        assert math.isclose(expanded["delta"], 0.125, abs_tol=1e-9)
        # The interpolated model is a proper probability distribution.
        assert math.isclose(sum(expanded.values()), 1.0, abs_tol=1e-9)

    def test_alpha_one_leaves_query_unchanged(self) -> None:
        expanded = _arm(alpha=1.0).expanded_query_model("alpha beta")
        # alpha == 1 -> pure maximum-likelihood query model, no expansion terms.
        assert set(expanded) == {"alpha", "beta"}
        assert math.isclose(expanded["alpha"], 0.5, abs_tol=1e-9)
        assert math.isclose(expanded["beta"], 0.5, abs_tol=1e-9)

    def test_expansion_widens_term_set(self) -> None:
        query_only = _arm(alpha=1.0).expanded_query_model("alpha beta")
        expanded = _arm(alpha=0.5).expanded_query_model("alpha beta")
        # Expansion strictly adds the co-occurring feedback terms.
        assert set(query_only) < set(expanded)
        assert {"gamma", "delta"} <= set(expanded)


class TestRM3Retrieve:
    def test_reranks_feedback_doc_first(self) -> None:
        arm = _arm(alpha=0.5)
        ctx = arm.retrieve("alpha beta", budget_tokens=10_000)
        assert ctx.ids[0] == "D-1"
        assert set(ctx.ids) == {"D-1", "D-2", "D-3", "D-4"}

    def test_empty_corpus(self) -> None:
        arm = RM3Memory(feedback_docs=1)
        assert arm.retrieve("alpha beta", 1000).ids == ()

    def test_respects_budget(self) -> None:
        arm = _arm()
        assert arm.retrieve("alpha beta", budget_tokens=0).ids == ()

    def test_deterministic(self) -> None:
        a = _arm(alpha=0.5).retrieve("alpha beta", 10_000)
        b = _arm(alpha=0.5).retrieve("alpha beta", 10_000)
        assert a.ids == b.ids


class TestRM3Validation:
    def test_rejects_out_of_range_alpha(self) -> None:
        for bad in (-0.1, 1.1):
            try:
                RM3Memory(alpha=bad)
            except ValueError:
                continue
            raise AssertionError(f"alpha={bad} should have raised ValueError")

    def test_rejects_non_positive_feedback_docs(self) -> None:
        try:
            RM3Memory(feedback_docs=0)
        except ValueError:
            return
        raise AssertionError("feedback_docs=0 should have raised ValueError")
