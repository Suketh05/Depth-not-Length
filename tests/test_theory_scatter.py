"""Tests for the scatter penalty p^sigma model."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from membench.theory.decay import SimilarityDecayModel
from membench.theory.recovery import similarity_recovery
from membench.theory.scatter import (
    ScatterDegradation,
    ScatterModel,
    degrade_recovery,
    scatter_factor,
    scattered_recovery,
)


@pytest.fixture
def decay() -> SimilarityDecayModel:
    return SimilarityDecayModel(s0=0.8, rho=0.5)


class TestScatterFactor:
    def test_sigma_zero_is_unity(self) -> None:
        # p^0 = 1 for any valid p: nothing to assemble, no penalty.
        for p in (0.1, 0.5, 0.92, 1.0):
            assert scatter_factor(p, 0) == 1.0

    def test_sigma_one_is_p(self) -> None:
        # p^1 = p: a single dereference pays exactly the per-area fidelity.
        assert scatter_factor(0.92, 1) == pytest.approx(0.92)

    def test_hand_computed_value(self) -> None:
        # Golden: p^sigma with p=0.5, sigma=3 -> 0.5^3 = 0.125 (hand-computed).
        assert scatter_factor(0.5, 3) == pytest.approx(0.125)

    def test_monotone_decreasing_in_sigma(self) -> None:
        # 0 < p < 1 => p^sigma strictly decreases as sigma grows.
        factors = [scatter_factor(0.7, sigma) for sigma in range(0, 8)]
        assert all(b < a for a, b in itertools.pairwise(factors))

    def test_in_unit_interval(self) -> None:
        for p in (0.05, 0.5, 0.92):
            for sigma in (0.0, 1.0, 2.5, 10.0):
                f = scatter_factor(p, sigma)
                assert 0.0 <= f <= 1.0

    def test_p_one_never_penalizes(self) -> None:
        for sigma in (0, 1, 5, 20):
            assert scatter_factor(1.0, sigma) == 1.0

    @pytest.mark.parametrize("p", [0.0, -0.1, 1.1])
    def test_rejects_bad_p(self, p: float) -> None:
        with pytest.raises(ValueError, match="p must be"):
            scatter_factor(p, 1)

    def test_rejects_negative_sigma(self) -> None:
        with pytest.raises(ValueError, match="sigma must be"):
            scatter_factor(0.5, -1)


class TestScatteredRecovery:
    def test_sigma_zero_leaves_recovery_unchanged(self, decay: SimilarityDecayModel) -> None:
        # Composition with similarity_recovery is the identity at sigma = 0.
        for d in range(1, 6):
            assert scattered_recovery(d, decay, 0.92, 0) == pytest.approx(
                similarity_recovery(d, decay)
            )

    def test_hand_computed_degraded_recovery(self, decay: SimilarityDecayModel) -> None:
        # Golden, fully hand-computed for s0=0.8, rho=0.5, d=2, p=0.5, sigma=2:
        #   P_sim(2) = s0^2 * rho^(2*3/2) = 0.8^2 * 0.5^3 = 0.64 * 0.125 = 0.08
        #   scatter  = p^sigma           = 0.5^2           = 0.25
        #   P_scat   = 0.08 * 0.25       = 0.02
        assert scattered_recovery(2, decay, 0.5, 2) == pytest.approx(0.02)

    def test_equals_product_of_pieces(self, decay: SimilarityDecayModel) -> None:
        d, p, sigma = 3, 0.92, 2.0
        expected = similarity_recovery(d, decay) * (p**sigma)
        assert scattered_recovery(d, decay, p, sigma) == pytest.approx(expected)

    def test_monotone_decreasing_in_sigma(self, decay: SimilarityDecayModel) -> None:
        values = [scattered_recovery(3, decay, 0.7, sigma) for sigma in range(0, 6)]
        assert all(b < a for a, b in itertools.pairwise(values))

    def test_stays_in_unit_interval(self, decay: SimilarityDecayModel) -> None:
        for d in range(1, 6):
            for sigma in (0.0, 1.0, 3.0, 9.0):
                v = scattered_recovery(d, decay, 0.92, sigma)
                assert 0.0 <= v <= 1.0


class TestDegradeRecovery:
    def test_hand_computed(self) -> None:
        # Golden: base 0.5 scattered with p=0.8, sigma=3 -> 0.5 * 0.8^3
        #   0.8^3 = 0.512 ; 0.5 * 0.512 = 0.256
        result = degrade_recovery(0.5, 0.8, 3)
        assert isinstance(result, ScatterDegradation)
        assert result.factor == pytest.approx(0.512)
        assert result.degraded_recovery == pytest.approx(0.256)
        assert result.base_recovery == 0.5
        assert result.sigma == 3.0
        assert result.p == 0.8

    def test_sigma_zero_preserves_base(self) -> None:
        result = degrade_recovery(0.73, 0.6, 0)
        assert result.factor == 1.0
        assert result.degraded_recovery == pytest.approx(0.73)

    def test_never_increases_recovery(self) -> None:
        for base in (0.1, 0.5, 0.9, 1.0):
            for sigma in (0.0, 1.0, 4.0):
                result = degrade_recovery(base, 0.85, sigma)
                assert result.degraded_recovery <= base + 1e-12
                assert 0.0 <= result.degraded_recovery <= 1.0

    @pytest.mark.parametrize("base", [-0.01, 1.01])
    def test_rejects_out_of_range_base(self, base: float) -> None:
        with pytest.raises(ValueError, match="base recovery"):
            degrade_recovery(base, 0.5, 1)


class TestScatterModel:
    def test_factor_matches_function(self) -> None:
        model = ScatterModel(p=0.92)
        for sigma in (0.0, 1.0, 2.0, 5.0):
            assert model.factor(sigma) == pytest.approx(scatter_factor(0.92, sigma))

    def test_scattered_recovery_matches_function(self, decay: SimilarityDecayModel) -> None:
        model = ScatterModel(p=0.92)
        for d in range(1, 5):
            assert model.scattered_recovery(d, decay, 2.0) == pytest.approx(
                scattered_recovery(d, decay, 0.92, 2.0)
            )

    def test_factor_curve_shape_and_values(self) -> None:
        model = ScatterModel(p=0.5)
        curve = model.factor_curve(4)
        # sigma = 0..4 -> [1, 0.5, 0.25, 0.125, 0.0625]
        assert curve.shape == (5,)
        assert curve[0] == 1.0
        np.testing.assert_allclose(curve, [1.0, 0.5, 0.25, 0.125, 0.0625])
        # strictly decreasing
        assert np.all(np.diff(curve) < 0)

    def test_typed_dereference_collapses_scatter(self, decay: SimilarityDecayModel) -> None:
        # A typed store collapses sigma=d to sigma=1: it pays p once, not p^d.
        model = ScatterModel(p=0.7)
        d = 4
        scattered = model.scattered_recovery(d, decay, sigma=float(d))
        dereferenced = model.scattered_recovery(d, decay, sigma=1.0)
        assert dereferenced > scattered

    def test_factor_curve_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="max_sigma"):
            ScatterModel(p=0.5).factor_curve(-1)

    @pytest.mark.parametrize("p", [0.0, -0.1, 1.1])
    def test_rejects_bad_p(self, p: float) -> None:
        with pytest.raises(ValueError, match="p must be"):
            ScatterModel(p=p)


def test_scatter_entropy_assembly_bound_shape() -> None:
    # thm:scatentropy assembly term: scattering across more areas can only lower
    # the assembly probability (assembly <= p^sigma is monotone non-increasing).
    p = 0.92
    sigmas = np.arange(0.0, 35.0)  # paper range sigma in [1, 34]
    assembly = np.asarray([scatter_factor(p, s) for s in sigmas])
    assert math.isclose(assembly[0], 1.0)
    assert np.all(np.diff(assembly) < 0)
    assert np.all((assembly > 0) & (assembly <= 1))
