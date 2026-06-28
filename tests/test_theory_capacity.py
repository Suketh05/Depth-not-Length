"""Tests for the information-theoretic context-free floor (Bayes / Fano)."""

from __future__ import annotations

import itertools
import math

import pytest

from membench.theory.capacity import (
    ContextFreeFloor,
    binary_entropy,
    bsc_capacity,
    context_free_floor,
    fano_error_lower_bound,
)


class TestBinaryEntropy:
    def test_golden_values(self) -> None:
        # H_b(0) = H_b(1) = 0 by convention; H_b(0.5) = 1 bit (its maximum).
        assert binary_entropy(0.0) == 0.0
        assert binary_entropy(1.0) == 0.0
        assert binary_entropy(0.5) == pytest.approx(1.0)
        # Hand-computed: H_b(0.25) = -0.25*log2(0.25) - 0.75*log2(0.75)
        #              = 0.25*2 + 0.75*0.4150374992788438 = 0.8112781244591328
        assert binary_entropy(0.25) == pytest.approx(0.8112781244591328)

    def test_symmetric(self) -> None:
        for p in (0.1, 0.3, 0.42):
            assert binary_entropy(p) == pytest.approx(binary_entropy(1.0 - p))

    def test_maximised_at_half(self) -> None:
        assert binary_entropy(0.5) >= binary_entropy(0.3)
        assert binary_entropy(0.5) >= binary_entropy(0.9)

    @pytest.mark.parametrize("p", [-0.01, 1.01, 2.0])
    def test_rejects_out_of_range(self, p: float) -> None:
        with pytest.raises(ValueError, match="p must be"):
            binary_entropy(p)


class TestBscCapacity:
    def test_golden_endpoints(self) -> None:
        # C(1/2) = 1 - H_b(1/2) = 1 - 1 = 0 bits (useless channel).
        assert bsc_capacity(0.5) == pytest.approx(0.0)
        # C(0) = 1 - H_b(0) = 1 - 0 = 1 bit (noiseless channel).
        assert bsc_capacity(0.0) == pytest.approx(1.0)
        # C(1) = 1 - H_b(1) = 1 bit (perfectly anti-correlated).
        assert bsc_capacity(1.0) == pytest.approx(1.0)
        # C(0.25) = 1 - 0.8112781244591328 = 0.1887218755408672
        assert bsc_capacity(0.25) == pytest.approx(0.1887218755408672)

    def test_symmetric_about_half(self) -> None:
        for p in (0.1, 0.2, 0.4):
            assert bsc_capacity(p) == pytest.approx(bsc_capacity(1.0 - p))

    def test_in_unit_interval(self) -> None:
        for p in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
            c = bsc_capacity(p)
            assert 0.0 <= c <= 1.0

    @pytest.mark.parametrize("p", [-0.1, 1.5])
    def test_rejects_out_of_range(self, p: float) -> None:
        with pytest.raises(ValueError, match="p must be"):
            bsc_capacity(p)


class TestFanoErrorLowerBound:
    def test_tight_at_max_entropy(self) -> None:
        # At H(X|Y) = log2(K), f(P_e) is maximised at P_e = (K-1)/K where it equals
        # log2(K) exactly, so the Fano bound is tight: P_e = 1 - 1/K.
        # K = 4: log2(4) = 2 bits -> bound = 3/4 = 0.75.
        assert fano_error_lower_bound(2.0, 4) == pytest.approx(0.75)

    def test_binary_channel_full_uncertainty(self) -> None:
        # K = 2: f(p) = H_b(p), maximised at p = 0.5 with H_b(0.5) = 1.
        # So with H(X|Y) = 1 bit the Fano bound is exactly 0.5.
        assert fano_error_lower_bound(1.0, 2) == pytest.approx(0.5)

    def test_zero_entropy_means_zero_error(self) -> None:
        for k in (1, 2, 5, 10):
            assert fano_error_lower_bound(0.0, k) == 0.0

    def test_clamped_above_feasible_max(self) -> None:
        # Entropy above log2(K) is infeasible; bound clamps to (K-1)/K.
        assert fano_error_lower_bound(99.0, 4) == pytest.approx(0.75)

    def test_monotonic_in_conditional_entropy(self) -> None:
        bounds = [fano_error_lower_bound(h, 8) for h in (0.0, 0.5, 1.0, 2.0, 3.0)]
        assert all(b <= c for b, c in itertools.pairwise(bounds))

    def test_never_exceeds_chance_error(self) -> None:
        for k in (2, 4, 8):
            assert fano_error_lower_bound(math.log2(k), k) <= (k - 1) / k + 1e-12

    @pytest.mark.parametrize("k", [0, -3])
    def test_rejects_bad_alphabet(self, k: int) -> None:
        with pytest.raises(ValueError, match="alphabet_size must be"):
            fano_error_lower_bound(1.0, k)

    def test_rejects_negative_entropy(self) -> None:
        with pytest.raises(ValueError, match="conditional_entropy must be"):
            fano_error_lower_bound(-0.5, 4)


class TestContextFreeFloor:
    def test_golden_uniform_quaternary(self) -> None:
        # Uniform K=4, zero mutual information:
        #   H(X) = log2(4) = 2 bits; Bayes accuracy = 1/4 = 0.25;
        #   Bayes error = 0.75; Fano floor coincides at 0.75.
        floor = context_free_floor(4)
        assert isinstance(floor, ContextFreeFloor)
        assert floor.alphabet_size == 4
        assert floor.prior_entropy == pytest.approx(2.0)
        assert floor.bayes_accuracy == pytest.approx(0.25)
        assert floor.bayes_error == pytest.approx(0.75)
        assert floor.fano_error_lower_bound == pytest.approx(0.75)

    def test_bayes_and_fano_floors_coincide(self) -> None:
        # The headline result: at zero information the two derivations agree.
        for k in (2, 3, 5, 16, 100):
            floor = context_free_floor(k)
            assert floor.fano_error_lower_bound == pytest.approx(floor.bayes_error)
            assert floor.bayes_accuracy == pytest.approx(1.0 / k)

    def test_binary_decision(self) -> None:
        # K=2: coin-flip floor -- 50% accuracy, 50% error.
        floor = context_free_floor(2)
        assert floor.bayes_accuracy == pytest.approx(0.5)
        assert floor.fano_error_lower_bound == pytest.approx(0.5)

    def test_singleton_has_no_error(self) -> None:
        floor = context_free_floor(1)
        assert floor.prior_entropy == 0.0
        assert floor.bayes_accuracy == pytest.approx(1.0)
        assert floor.bayes_error == pytest.approx(0.0)
        assert floor.fano_error_lower_bound == 0.0

    def test_accuracy_floor_falls_with_alphabet_size(self) -> None:
        accs = [context_free_floor(k).bayes_accuracy for k in (2, 4, 8, 16)]
        assert all(a > b for a, b in itertools.pairwise(accs))

    def test_is_frozen(self) -> None:
        floor = context_free_floor(4)
        with pytest.raises((AttributeError, TypeError)):
            floor.bayes_accuracy = 0.0  # type: ignore[misc]

    def test_rejects_bad_alphabet(self) -> None:
        with pytest.raises(ValueError, match="alphabet_size must be"):
            context_free_floor(0)
