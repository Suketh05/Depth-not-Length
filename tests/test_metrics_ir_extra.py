"""Tests for the extra rank-quality IR metrics (RBP / ERR / AP-MAP / R-precision).

Golden values are hand-computed from the published formulae (Moffat & Zobel 2008;
Chapelle et al. 2009; Manning/Raghavan/Schutze 2008), NOT read back from the
implementation, so the assertions are an independent check.
"""

from __future__ import annotations

import pytest

from membench.metrics.ir_extra import (
    average_precision,
    expected_reciprocal_rank,
    mean_average_precision,
    r_precision,
    rank_biased_precision,
)

# Fixed relevance vector used for the hand-computed goldens (binary, best-first):
#   REL = [1, 0, 1, 0]  -> 2 relevant items, at ranks 1 and 3.
REL = [1.0, 0.0, 1.0, 0.0]

# RBP, p = 0.8:  (1-p) * sum_i rel_i p^(i-1)
#   = 0.2 * (1*0.8^0 + 0*0.8^1 + 1*0.8^2 + 0*0.8^3)
#   = 0.2 * (1 + 0.64) = 0.2 * 1.64 = 0.328
GOLDEN_RBP = 0.328

# AP (binary): precision at each relevant rank / num_relevant
#   rank 1 -> 1/1 = 1 ;  rank 3 -> 2/3 ;  (1 + 2/3) / 2 = 5/6
GOLDEN_AP = 5.0 / 6.0

# R-precision: R = 2 relevant; top-2 = [1, 0] -> 1 relevant / 2 = 0.5
GOLDEN_RPREC = 0.5

# ERR with max_grade = 1 -> stop prob R_g = (2^g - 1)/2^1: g=1 -> 0.5, g=0 -> 0.
#   r=1: (1/1)*0.5            = 0.5
#   r=2: (1/2)*0              = 0
#   r=3: (1/3)*0.5*(1-0.5)*(1-0) = (1/3)*0.25 = 1/12
#   r=4: 0
#   ERR = 0.5 + 1/12 = 7/12
GOLDEN_ERR = 7.0 / 12.0


class TestHandComputedGoldens:
    def test_rbp_matches_hand_value(self) -> None:
        assert rank_biased_precision(REL, p=0.8) == pytest.approx(GOLDEN_RBP)

    def test_average_precision_matches_hand_value(self) -> None:
        assert average_precision(REL) == pytest.approx(GOLDEN_AP)

    def test_r_precision_matches_hand_value(self) -> None:
        assert r_precision(REL) == pytest.approx(GOLDEN_RPREC)

    def test_err_matches_hand_value(self) -> None:
        assert expected_reciprocal_rank(REL, max_grade=1.0) == pytest.approx(GOLDEN_ERR)


# Same relevant multiset as REL but ranked perfectly: [1, 1, 0, 0].
PERFECT = [1.0, 1.0, 0.0, 0.0]


class TestPerfectRankingMaximises:
    def test_perfect_rbp_is_higher(self) -> None:
        # 0.2 * (1 + 0.8) = 0.36 > 0.328
        assert rank_biased_precision(PERFECT, p=0.8) == pytest.approx(0.36)
        assert rank_biased_precision(PERFECT, p=0.8) > rank_biased_precision(REL, p=0.8)

    def test_perfect_ap_is_one(self) -> None:
        assert average_precision(PERFECT) == pytest.approx(1.0)
        assert average_precision(PERFECT) > average_precision(REL)

    def test_perfect_r_precision_is_one(self) -> None:
        assert r_precision(PERFECT) == pytest.approx(1.0)
        assert r_precision(PERFECT) > r_precision(REL)

    def test_perfect_err_is_higher(self) -> None:
        # r=1: 0.5 ; r=2: 0.5*0.5*0.5 = 0.125 ; ERR = 0.625 > 7/12
        assert expected_reciprocal_rank(PERFECT, max_grade=1.0) == pytest.approx(0.625)
        assert expected_reciprocal_rank(PERFECT, max_grade=1.0) > expected_reciprocal_rank(
            REL, max_grade=1.0
        )


INVARIANT_VECTORS = [
    [1.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0],
]


class TestInvariants:
    @pytest.mark.parametrize("vec", INVARIANT_VECTORS)
    def test_all_metrics_in_unit_interval(self, vec: list[float]) -> None:
        assert 0.0 <= rank_biased_precision(vec, p=0.8) <= 1.0
        assert 0.0 <= expected_reciprocal_rank(vec) <= 1.0
        assert 0.0 <= average_precision(vec) <= 1.0
        assert 0.0 <= r_precision(vec) <= 1.0

    def test_all_relevant_first_gives_one_for_ap_and_rprec(self) -> None:
        assert average_precision([1.0, 1.0, 1.0]) == pytest.approx(1.0)
        assert r_precision([1.0, 1.0, 1.0]) == pytest.approx(1.0)

    def test_no_relevant_is_zero(self) -> None:
        assert average_precision([0.0, 0.0]) == 0.0
        assert r_precision([0.0, 0.0]) == 0.0
        assert expected_reciprocal_rank([0.0, 0.0]) == 0.0
        assert rank_biased_precision([0.0, 0.0], p=0.8) == 0.0


class TestGradedErr:
    def test_graded_relevance_uses_two_to_the_grade(self) -> None:
        # rel = [2], max_grade = 2 -> R = (2^2 - 1)/2^2 = 3/4; ERR = (1/1)*0.75 = 0.75
        assert expected_reciprocal_rank([2.0], max_grade=2.0) == pytest.approx(0.75)

    def test_default_max_grade_is_observed_max(self) -> None:
        # max_grade inferred as 1 -> same as the binary golden first term: ERR([1]) = 0.5
        assert expected_reciprocal_rank([1.0]) == pytest.approx(0.5)


class TestMap:
    def test_map_averages_average_precision(self) -> None:
        # AP([1,0,1,0]) = 5/6 ; AP([1,1]) = 1 ; MAP = (5/6 + 1)/2 = 11/12
        result = mean_average_precision([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0]])
        assert result == pytest.approx((5.0 / 6.0 + 1.0) / 2.0)

    def test_map_empty_is_zero(self) -> None:
        assert mean_average_precision([]) == 0.0


class TestValidation:
    def test_rbp_rejects_out_of_range_p(self) -> None:
        with pytest.raises(ValueError, match="0 < p < 1"):
            rank_biased_precision([1.0], p=1.0)
        with pytest.raises(ValueError, match="0 < p < 1"):
            rank_biased_precision([1.0], p=0.0)

    def test_err_rejects_negative_relevance(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            expected_reciprocal_rank([-1.0])

    def test_err_rejects_non_positive_max_grade(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            expected_reciprocal_rank([1.0], max_grade=0.0)

    def test_r_precision_rejects_negative_n(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            r_precision([1.0], n_relevant=-1)
