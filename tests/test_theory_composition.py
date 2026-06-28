"""Tests for the multi-hop decay composition algebra.

The golden values are hand-computed from the series-reliability law
``R(path) = prod_i p_i`` (Barlow & Proschan, 1975) and cross-checked against the
closed forms in :mod:`membench.theory.recovery`, never produced by running the
code under test.
"""

from __future__ import annotations

import math

import pytest

from membench.theory.composition import (
    Hop,
    HopType,
    compose_log_survival,
    compose_path,
    compose_survival,
    similarity_hop,
    typed_hop,
)
from membench.theory.decay import SimilarityDecayModel
from membench.theory.recovery import similarity_recovery, structured_recovery


class TestComposeSurvival:
    def test_golden_two_hops(self) -> None:
        # Hand-computed: 0.9 * 0.8 = 0.72 (series-system reliability of two hops).
        assert compose_survival([0.9, 0.8]) == pytest.approx(0.72)

    def test_empty_product_is_one(self) -> None:
        # Empty product (a path of no hops) recovers with certainty.
        assert compose_survival([]) == 1.0

    def test_single_hop_is_itself(self) -> None:
        assert compose_survival([0.37]) == pytest.approx(0.37)

    def test_zero_hop_absorbs(self) -> None:
        assert compose_survival([0.9, 0.0, 0.8]) == 0.0

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="survival probability"):
            compose_survival([0.9, 1.5])
        with pytest.raises(ValueError, match="survival probability"):
            compose_survival([-0.1])


class TestComposeLogSurvival:
    def test_matches_log_of_product(self) -> None:
        # log(0.9 * 0.8) = log(0.72) computed independently via math.log.
        assert compose_log_survival([0.9, 0.8]) == pytest.approx(math.log(0.72))

    def test_empty_sum_is_zero(self) -> None:
        assert compose_log_survival([]) == 0.0

    def test_zero_hop_is_neg_inf(self) -> None:
        assert compose_log_survival([0.9, 0.0]) == -math.inf

    def test_exp_of_log_recovers_survival(self) -> None:
        survivals = [0.9, 0.8, 0.7, 0.6]
        assert math.exp(compose_log_survival(survivals)) == pytest.approx(
            compose_survival(survivals)
        )


class TestInvariants:
    def test_associativity(self) -> None:
        # (a*b)*c == a*(b*c): composing intermediates is order-of-grouping free.
        a, b, c = 0.9, 0.8, 0.7
        left = compose_survival([compose_survival([a, b]), c])
        right = compose_survival([a, compose_survival([b, c])])
        flat = compose_survival([a, b, c])
        assert left == pytest.approx(flat)
        assert right == pytest.approx(flat)

    def test_identity_hop_is_neutral(self) -> None:
        # A unit-survival hop is the neutral element: 0.9 * 1.0 * 0.8 == 0.9 * 0.8.
        with_identity = compose_survival([0.9, 1.0, 0.8])
        without = compose_survival([0.9, 0.8])
        assert with_identity == pytest.approx(without)

    def test_monotone_decreasing_in_path_length(self) -> None:
        # Appending any hop with survival < 1 cannot increase whole-path survival.
        base = [0.9, 0.8, 0.7]
        running = 1.0
        for k in range(1, len(base) + 1):
            nxt = compose_survival(base[:k])
            assert nxt <= running
            running = nxt

    def test_monotone_strict_when_hop_below_one(self) -> None:
        prefix = compose_survival([0.9, 0.8])
        extended = compose_survival([0.9, 0.8, 0.95])
        assert extended < prefix


class TestHeterogeneousHops:
    def test_typed_path_matches_structured_recovery(self) -> None:
        # d typed hops at reliability q compose to q**d == structured_recovery(d, q).
        q = 0.95
        hops = [typed_hop(q) for _ in range(3)]
        result = compose_path(hops)
        assert result.survival == pytest.approx(structured_recovery(3, q))  # 0.95**3
        assert result.n_typed == 3
        assert result.n_similarity == 0
        assert result.n_hops == 3

    def test_similarity_path_matches_similarity_recovery(self) -> None:
        # Hops at distances 1..d compose to s0**d * rho**(d(d+1)/2).
        model = SimilarityDecayModel(s0=0.9, rho=0.5)
        hops = [similarity_hop(model, distance=i) for i in range(1, 3)]
        result = compose_path(hops)
        # similarity_recovery(2) = 0.9**2 * 0.5**3 = 0.81 * 0.125 = 0.10125.
        assert result.survival == pytest.approx(similarity_recovery(2, model))
        assert result.survival == pytest.approx(0.10125)
        assert result.n_similarity == 2
        assert result.n_typed == 0

    def test_mixed_path_breakdown_and_product(self) -> None:
        model = SimilarityDecayModel(s0=0.9, rho=0.5)
        hops = [typed_hop(0.95), similarity_hop(model, distance=1)]
        result = compose_path(hops)
        # 0.95 * (0.9 * 0.5) = 0.95 * 0.45 = 0.4275 (hand-computed).
        assert result.survival == pytest.approx(0.4275)
        assert result.n_typed == 1
        assert result.n_similarity == 1
        assert result.n_hops == 2

    def test_empty_path_recovers_with_certainty(self) -> None:
        result = compose_path([])
        assert result.survival == 1.0
        assert result.log_survival == 0.0
        assert result.n_hops == 0


class TestHop:
    def test_rejects_out_of_range_survival(self) -> None:
        with pytest.raises(ValueError, match="survival probability"):
            Hop(survival=1.2, hop_type=HopType.TYPED)

    def test_similarity_hop_rejects_undecayed_over_one(self) -> None:
        # An undecayed model can predict s > 1, which is not a valid survival.
        model = SimilarityDecayModel(s0=0.9, rho=2.0)  # retrievability(1) = 1.8
        with pytest.raises(ValueError, match="survival probability"):
            similarity_hop(model, distance=1)
