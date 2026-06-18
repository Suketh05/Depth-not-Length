"""Tests for the full-chain recovery bounds."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from membench.theory.decay import SimilarityDecayModel
from membench.theory.recovery import (
    RecoveryModel,
    log_similarity_recovery,
    similarity_recovery,
    structured_recovery,
)


@pytest.fixture
def decay() -> SimilarityDecayModel:
    return SimilarityDecayModel(s0=0.9, rho=0.6)


class TestSimilarityRecovery:
    def test_matches_explicit_product(self, decay: SimilarityDecayModel) -> None:
        for d in range(1, 6):
            expected = math.prod(decay.retrievability(i) for i in range(1, d + 1))
            assert similarity_recovery(d, decay) == pytest.approx(expected)

    def test_closed_form_exponent(self, decay: SimilarityDecayModel) -> None:
        # P_sim(d) = s0^d * rho^(d(d+1)/2)
        d = 4
        expected = decay.s0**d * decay.rho ** (d * (d + 1) / 2)
        assert similarity_recovery(d, decay) == pytest.approx(expected)

    def test_log_matches_log_of_value(self, decay: SimilarityDecayModel) -> None:
        for d in range(1, 8):
            assert log_similarity_recovery(d, decay) == pytest.approx(
                math.log(similarity_recovery(d, decay))
            )

    def test_log_is_finite_where_value_underflows(self) -> None:
        steep = SimilarityDecayModel(s0=0.5, rho=0.3)
        assert similarity_recovery(60, steep) == 0.0  # underflows float64
        assert math.isfinite(log_similarity_recovery(60, steep))

    def test_rejects_bad_depth(self, decay: SimilarityDecayModel) -> None:
        with pytest.raises(ValueError, match="depth"):
            similarity_recovery(0, decay)


class TestStructuredRecovery:
    def test_is_q_to_the_d(self) -> None:
        assert structured_recovery(5, 0.95) == pytest.approx(0.95**5)

    @pytest.mark.parametrize("q", [0.0, -0.1, 1.1])
    def test_rejects_bad_q(self, q: float) -> None:
        with pytest.raises(ValueError, match="q must be"):
            structured_recovery(2, q)


class TestRecoveryModel:
    def test_advantage_grows_with_depth(self, decay: SimilarityDecayModel) -> None:
        model = RecoveryModel(decay=decay, q=0.95)
        ratios = [model.advantage_ratio(d) for d in range(1, 8)]
        assert all(b > a for a, b in itertools.pairwise(ratios))

    def test_log_advantage_consistent_with_components(self, decay: SimilarityDecayModel) -> None:
        model = RecoveryModel(decay=decay, q=0.9)
        d = 3
        expected = math.log(model.p_struct(d)) - math.log(model.p_sim(d))
        assert model.log_advantage(d) == pytest.approx(expected)

    def test_curves_have_expected_shape_and_monotonicity(self, decay: SimilarityDecayModel) -> None:
        model = RecoveryModel(decay=decay, q=0.97)
        sim = model.p_sim_curve(6)
        struct = model.p_struct_curve(6)
        assert sim.shape == struct.shape == (6,)
        # similarity collapses strictly faster than structured memory decays
        assert np.all(np.diff(sim) < 0)
        assert np.all(struct > sim)

    def test_rejects_bad_q(self, decay: SimilarityDecayModel) -> None:
        with pytest.raises(ValueError, match="q must be"):
            RecoveryModel(decay=decay, q=1.5)
