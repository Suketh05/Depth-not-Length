"""Tests for analytic sensitivities and elasticities of the recovery laws.

Golden values are derived independently of the implementation (by hand from the
closed forms) and cross-checked against central finite differences; nothing here
asserts a value that was produced by running the code under test.
"""

from __future__ import annotations

import pytest

from membench.theory.crossover import find_crossover_depth
from membench.theory.decay import SimilarityDecayModel
from membench.theory.recovery import (
    RecoveryModel,
    similarity_recovery,
    structured_recovery,
)
from membench.theory.sensitivity import (
    central_difference,
    crossover_sensitivity,
    gap_sensitivity,
    similarity_sensitivity,
    structured_sensitivity,
)


@pytest.fixture
def decay() -> SimilarityDecayModel:
    return SimilarityDecayModel(s0=0.9, rho=0.6)


class TestCentralDifference:
    def test_matches_known_derivative_of_cube(self) -> None:
        # d/dx x^3 = 3x^2; at x = 2 the derivative is exactly 12.
        est = central_difference(lambda x: x**3, 2.0, h=1e-5)
        assert est == pytest.approx(12.0, rel=1e-6)

    def test_rejects_nonpositive_step(self) -> None:
        with pytest.raises(ValueError, match="step h"):
            central_difference(lambda x: x, 1.0, h=0.0)


class TestSimilaritySensitivity:
    def test_golden_gradients_and_value(self, decay: SimilarityDecayModel) -> None:
        # Hand-computed at s0=0.9, rho=0.6, d=3, with T = d(d+1)/2 = 6:
        #   P_sim   = 0.9^3 * 0.6^6 = 0.729 * 0.046656            = 0.034012224
        #   dP/ds0  = 3 * 0.9^2 * 0.6^6 = 3 * 0.81 * 0.046656     = 0.11337408
        #   dP/drho = 6 * 0.9^3 * 0.6^5 = 6 * 0.729 * 0.07776     = 0.34012224
        s = similarity_sensitivity(3, decay)
        assert s.value == pytest.approx(0.034012224)
        assert s.grad_s0 == pytest.approx(0.11337408)
        assert s.grad_rho == pytest.approx(0.34012224)
        # P_sim does not depend on q.
        assert s.grad_q == 0.0
        assert s.elasticity_q == 0.0

    def test_elasticities_are_the_exponents(self) -> None:
        # For a monomial s0^d * rho^(d(d+1)/2) the elasticities are exactly the
        # exponents: eps_s0 = d, eps_rho = d(d+1)/2 -- independent of (s0, rho).
        for s0, rho in [(0.9, 0.6), (0.5, 0.8), (0.99, 0.3)]:
            model = SimilarityDecayModel(s0=s0, rho=rho)
            for d in range(1, 6):
                s = similarity_sensitivity(d, model)
                assert s.elasticity_s0 == pytest.approx(float(d))
                assert s.elasticity_rho == pytest.approx(float(d * (d + 1) / 2))

    def test_grad_rho_matches_finite_difference(self, decay: SimilarityDecayModel) -> None:
        for d in (1, 2, 4, 7):
            analytic = similarity_sensitivity(d, decay).grad_rho

            def f(rho: float, d: int = d) -> float:
                return similarity_recovery(d, SimilarityDecayModel(s0=decay.s0, rho=rho))

            assert analytic == pytest.approx(central_difference(f, decay.rho, h=1e-6), rel=1e-5)

    def test_grad_s0_matches_finite_difference(self, decay: SimilarityDecayModel) -> None:
        for d in (1, 2, 4, 7):
            analytic = similarity_sensitivity(d, decay).grad_s0

            def f(s0: float, d: int = d) -> float:
                return similarity_recovery(d, SimilarityDecayModel(s0=s0, rho=decay.rho))

            assert analytic == pytest.approx(central_difference(f, decay.s0, h=1e-6), rel=1e-5)

    def test_recovery_decreases_in_depth(self, decay: SimilarityDecayModel) -> None:
        # Decaying recovery (s0 < 1, rho < 1) strictly falls with depth.
        for d in range(1, 6):
            assert similarity_sensitivity(d, decay).grad_d < 0.0

    def test_rejects_bad_depth(self, decay: SimilarityDecayModel) -> None:
        with pytest.raises(ValueError, match="depth"):
            similarity_sensitivity(0, decay)


class TestStructuredSensitivity:
    def test_golden_gradient_and_elasticity(self) -> None:
        # Hand-computed at q=0.95, d=3:
        #   P_struct = 0.95^3 = 0.857375
        #   dP/dq    = 3 * 0.95^2 = 3 * 0.9025 = 2.7075
        #   eps_q    = (q/P) dP/dq = d = 3
        s = structured_sensitivity(3, 0.95)
        assert s.value == pytest.approx(0.857375)
        assert s.grad_q == pytest.approx(2.7075)
        assert s.elasticity_q == pytest.approx(3.0)
        # Structured recovery uses neither s0 nor rho.
        assert s.grad_s0 == 0.0
        assert s.grad_rho == 0.0

    def test_increases_in_q_decreases_in_depth(self) -> None:
        for d in range(1, 6):
            s = structured_sensitivity(d, 0.9)
            assert s.grad_q > 0.0  # recovery increases in q
            assert s.grad_d < 0.0  # recovery decreases in depth (q < 1)

    def test_grad_q_matches_finite_difference(self) -> None:
        for d in (1, 2, 4, 8):
            analytic = structured_sensitivity(d, 0.9).grad_q
            est = central_difference(lambda q, d=d: structured_recovery(d, q), 0.9, h=1e-6)
            assert analytic == pytest.approx(est, rel=1e-5)

    def test_q_one_is_a_flat_optimum_in_depth(self) -> None:
        # At q = 1 every chain is recovered: value 1, no depth penalty.
        s = structured_sensitivity(4, 1.0)
        assert s.value == 1.0
        assert s.grad_d == 0.0


class TestGapSensitivity:
    def test_partials_decompose_into_the_two_terms(self, decay: SimilarityDecayModel) -> None:
        model = RecoveryModel(decay=decay, q=0.95)
        d = 3
        gap = gap_sensitivity(d, model)
        sim = similarity_sensitivity(d, decay)
        struct = structured_sensitivity(d, 0.95)
        # g = P_struct - P_sim, so each partial is the term-by-term difference.
        assert gap.grad_s0 == pytest.approx(-sim.grad_s0)
        assert gap.grad_rho == pytest.approx(-sim.grad_rho)
        assert gap.grad_q == pytest.approx(struct.grad_q)
        assert gap.value == pytest.approx(struct.value - sim.value)

    def test_grad_q_matches_finite_difference(self, decay: SimilarityDecayModel) -> None:
        model = RecoveryModel(decay=decay, q=0.9)
        d = 4
        analytic = gap_sensitivity(d, model).grad_q

        def f(q: float) -> float:
            rm = RecoveryModel(decay=decay, q=q)
            return rm.p_struct(d) - rm.p_sim(d)

        assert analytic == pytest.approx(central_difference(f, 0.9, h=1e-6), rel=1e-5)

    def test_gap_widens_with_q_narrows_with_similarity(self, decay: SimilarityDecayModel) -> None:
        gap = gap_sensitivity(3, RecoveryModel(decay=decay, q=0.95))
        assert gap.grad_q > 0.0  # stronger structure widens the gap
        assert gap.grad_s0 < 0.0  # stronger similarity narrows it
        assert gap.grad_rho < 0.0


class TestCrossoverSensitivity:
    # Interior-crossover regime: similarity is strong at low depth, so structure
    # does not win until d > 1 and d* moves smoothly with the parameters.
    BASE = RecoveryModel(decay=SimilarityDecayModel(s0=0.95, rho=0.95), q=0.9)
    TAU = 0.05

    def test_base_point_has_interior_crossover(self) -> None:
        res = find_crossover_depth(self.BASE, self.TAU)
        assert res.d_star_continuous is not None
        assert res.d_star_continuous > 1.0  # a genuine interior root, not clamped

    def test_signs_match_economic_intuition(self) -> None:
        cs = crossover_sensitivity(self.BASE, self.TAU)
        # Stronger similarity recovery (larger s0 or rho) delays the crossover;
        # stronger structured recovery (larger q) brings it forward.
        assert cs.grad_s0 > 0.0
        assert cs.grad_rho > 0.0
        assert cs.grad_q < 0.0

    def test_grad_q_matches_independent_central_difference(self) -> None:
        cs = crossover_sensitivity(self.BASE, self.TAU, h=1e-4)

        def d_star(q: float) -> float:
            res = find_crossover_depth(RecoveryModel(decay=self.BASE.decay, q=q), self.TAU)
            assert res.d_star_continuous is not None
            return res.d_star_continuous

        # Re-derive the same derivative with a different step; central differences
        # of an O(h^2)-smooth root agree closely.
        assert cs.grad_q == pytest.approx(central_difference(d_star, self.BASE.q, h=5e-4), rel=1e-2)

    def test_rejects_nonpositive_step(self) -> None:
        with pytest.raises(ValueError, match="step h"):
            crossover_sensitivity(self.BASE, self.TAU, h=0.0)

    def test_raises_when_no_crossover(self) -> None:
        # Overhead so high structured memory never pays: d* is undefined.
        with pytest.raises(ValueError, match="no crossover"):
            crossover_sensitivity(self.BASE, tau=10.0)
