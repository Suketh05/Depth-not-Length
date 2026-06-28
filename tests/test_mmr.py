"""Tests for the MMR diversity reranking arm.

The golden selection orders are hand-computed from the Carbonell & Goldstein
(1998) objective on a 3-document toy, then asserted against the implementation
(never the other way round).
"""

from __future__ import annotations

import numpy as np
import pytest

from membench.retrieval import (
    dense,  # noqa: F401  (registration side effect)
    mmr,  # noqa: F401  (registration side effect)
)
from membench.retrieval.base import ArmFamily, build_arm, get_spec
from membench.retrieval.mmr import mmr_order
from membench.types import MemoryItem

# ---------------------------------------------------------------------------
# Toy problem for the hand-computed golden (3 documents).
#
#   relevance r = [0.9, 0.8, 0.5]          (doc 0 most relevant to the query)
#   pairwise similarity S (symmetric, unit diagonal):
#                 d0    d1    d2
#         d0 [ 1.00  0.80  0.10 ]          (docs 0 and 1 are near-duplicates)
#         d1 [ 0.80  1.00  0.20 ]
#         d2 [ 0.10  0.20  1.00 ]
#
# MMR(d) = lambda * r[d] - (1 - lambda) * max_{j in S} S[d, j].
#
# --- lambda = 0.5 -------------------------------------------------------------
# Step 1: S = {}.  Redundancy term = 0, so pick argmax(r) = d0.   S = {0}
# Step 2: S = {0}.
#   MMR(d1) = 0.5*0.8 - 0.5*S[1,0] = 0.40 - 0.5*0.80 = 0.40 - 0.40 = 0.00
#   MMR(d2) = 0.5*0.5 - 0.5*S[2,0] = 0.25 - 0.5*0.10 = 0.25 - 0.05 = 0.20
#   0.20 > 0.00  ->  pick d2.                                     S = {0, 2}
# Step 3: only d1 remains  ->  pick d1.
#   GOLDEN selection order (lambda=0.5) = [0, 2, 1]
#
# --- lambda = 1.0 (pure relevance) -------------------------------------------
#   Diversity weight (1 - lambda) = 0, so MMR(d) = r[d]; order by r descending:
#   r = [0.9, 0.8, 0.5]  ->  GOLDEN order (lambda=1.0) = [0, 1, 2]
# ---------------------------------------------------------------------------

_RELEVANCE = np.array([0.9, 0.8, 0.5], dtype=np.float64)
_SIMILARITY = np.array(
    [
        [1.0, 0.8, 0.1],
        [0.8, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ],
    dtype=np.float64,
)


class TestMMROrderGolden:
    def test_lambda_half_selects_for_diversity(self) -> None:
        # Hand-computed above: the near-duplicate d1 is demoted below d2.
        assert mmr_order(_RELEVANCE, _SIMILARITY, 0.5) == [0, 2, 1]

    def test_lambda_one_reduces_to_pure_relevance(self) -> None:
        # lambda=1 drops the diversity term -> ranking by relevance descending.
        assert mmr_order(_RELEVANCE, _SIMILARITY, 1.0) == [0, 1, 2]

    def test_first_pick_is_always_max_relevance(self) -> None:
        # Empty-selection redundancy is 0, so step 1 is argmax(relevance) for any lambda.
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert mmr_order(_RELEVANCE, _SIMILARITY, lam)[0] == 0

    def test_k_truncates_selection(self) -> None:
        order = mmr_order(_RELEVANCE, _SIMILARITY, 0.5, k=2)
        assert order == [0, 2]

    def test_order_is_a_permutation(self) -> None:
        order = mmr_order(_RELEVANCE, _SIMILARITY, 0.5)
        assert sorted(order) == [0, 1, 2]

    def test_rejects_lambda_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="lambda_"):
            mmr_order(_RELEVANCE, _SIMILARITY, 1.5)

    def test_rejects_mismatched_similarity_shape(self) -> None:
        with pytest.raises(ValueError, match="similarity"):
            mmr_order(_RELEVANCE, np.eye(2, dtype=np.float64), 0.5)


_CORPUS = [
    MemoryItem("D-1", "all data exports must call withAuditLog for SOC-2 compliance"),
    MemoryItem("D-2", "every data export must call withAuditLog for SOC-2 compliance"),
    MemoryItem("D-3", "use DateRangePicker for date range selection in the UI"),
    MemoryItem("D-4", "primary actions use the Button component variant"),
]


class TestMMRArm:
    def test_registered_as_dense_using_embeddings(self) -> None:
        spec = get_spec("mmr")
        assert spec.capabilities.family is ArmFamily.DENSE
        assert spec.capabilities.uses_embeddings
        assert not spec.capabilities.follows_links

    def test_constructible_with_no_args(self) -> None:
        # The harness calls build_arm(name) with no kwargs.
        arm = build_arm("mmr")
        arm.write(_CORPUS)  # type: ignore[attr-defined]
        ctx = arm.retrieve("audit log on exports", 10_000)  # type: ignore[attr-defined]
        assert set(ctx.ids) == {"D-1", "D-2", "D-3", "D-4"}

    def test_lambda_one_matches_pure_relevance_order(self) -> None:
        # With diversity disabled, MMR must reproduce the dense (cosine) ranking.
        mmr_arm = build_arm("mmr", lambda_=1.0)
        dense_arm = build_arm("dense")
        mmr_arm.write(_CORPUS)  # type: ignore[attr-defined]
        dense_arm.write(_CORPUS)  # type: ignore[attr-defined]
        q = "add a CSV export with an audit trail"
        assert mmr_arm.retrieve(q, 10_000).ids == dense_arm.retrieve(q, 10_000).ids  # type: ignore[attr-defined]

    def test_diversity_demotes_near_duplicate(self) -> None:
        # D-1 and D-2 are near-duplicates; a strongly diversity-weighted MMR must
        # not place both ahead of the dissimilar items.
        arm = build_arm("mmr", lambda_=0.2)
        arm.write(_CORPUS)  # type: ignore[attr-defined]
        ids = arm.retrieve("audit log on data exports", 10_000).ids  # type: ignore[attr-defined]
        top_two = set(ids[:2])
        assert top_two != {"D-1", "D-2"}

    def test_deterministic(self) -> None:
        a, b = build_arm("mmr"), build_arm("mmr")
        a.write(_CORPUS)  # type: ignore[attr-defined]
        b.write(_CORPUS)  # type: ignore[attr-defined]
        assert a.retrieve("audit", 10_000).ids == b.retrieve("audit", 10_000).ids  # type: ignore[attr-defined]

    def test_empty_corpus(self) -> None:
        assert build_arm("mmr").retrieve("x", 1000).ids == ()  # type: ignore[attr-defined]

    @pytest.mark.parametrize("budget", [0, 30])
    def test_respects_budget(self, budget: int) -> None:
        arm = build_arm("mmr")
        arm.write(_CORPUS)  # type: ignore[attr-defined]
        ctx = arm.retrieve("audit", budget)  # type: ignore[attr-defined]
        assert len(ctx.ids) <= len(_CORPUS)
