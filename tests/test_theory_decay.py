"""Tests for the similarity-decay model and its log-linear estimator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from membench.theory.decay import SimilarityDecayModel, fit_similarity_decay


class TestSimilarityDecayModel:
    def test_retrievability_is_geometric(self) -> None:
        m = SimilarityDecayModel(s0=0.9, rho=0.5)
        assert m.retrievability(0) == pytest.approx(0.9)
        assert m.retrievability(1) == pytest.approx(0.45)
        assert m.retrievability(3) == pytest.approx(0.9 * 0.5**3)

    def test_retrievabilities_excludes_distance_zero(self) -> None:
        m = SimilarityDecayModel(s0=1.0, rho=0.5)
        vals = m.retrievabilities(3)
        assert vals.shape == (3,)
        np.testing.assert_allclose(vals, [0.5, 0.25, 0.125])

    def test_retrievabilities_requires_positive_distance(self) -> None:
        with pytest.raises(ValueError, match="max_distance"):
            SimilarityDecayModel(s0=1.0, rho=0.5).retrievabilities(0)

    def test_half_life(self) -> None:
        m = SimilarityDecayModel(s0=1.0, rho=0.5)
        assert m.half_life == pytest.approx(1.0)  # halves every hop
        assert SimilarityDecayModel(s0=1.0, rho=0.25).half_life == pytest.approx(0.5)

    def test_no_decay_has_infinite_half_life(self) -> None:
        m = SimilarityDecayModel(s0=0.8, rho=1.5)
        assert not m.is_decaying
        assert math.isinf(m.half_life)

    @pytest.mark.parametrize(("s0", "rho"), [(0.0, 0.5), (-1.0, 0.5), (0.9, 0.0), (0.9, -0.1)])
    def test_rejects_nonpositive_params(self, s0: float, rho: float) -> None:
        with pytest.raises(ValueError):
            SimilarityDecayModel(s0=s0, rho=rho)


class TestFitSimilarityDecay:
    def test_recovers_known_parameters_noise_free(self) -> None:
        true = SimilarityDecayModel(s0=0.85, rho=0.6)
        distances = list(range(0, 8))
        sims = [true.retrievability(i) for i in distances]
        fit = fit_similarity_decay(distances, sims)
        assert fit.model.rho == pytest.approx(0.6, rel=1e-6)
        assert fit.model.s0 == pytest.approx(0.85, rel=1e-6)
        assert fit.r_squared == pytest.approx(1.0, abs=1e-9)
        assert fit.n_points == 8

    def test_recovers_parameters_under_small_noise(self) -> None:
        rng = np.random.default_rng(0)
        true = SimilarityDecayModel(s0=0.9, rho=0.55)
        distances = np.repeat(np.arange(0, 6), 20)
        sims = true.s0 * true.rho**distances * np.exp(rng.normal(0, 0.02, distances.size))
        fit = fit_similarity_decay(distances, sims)
        assert fit.model.rho == pytest.approx(0.55, rel=0.05)
        assert fit.r_squared > 0.97
        assert fit.p_value < 1e-6

    def test_predict_matches_model(self) -> None:
        fit = fit_similarity_decay([0, 1, 2, 3], [1.0, 0.5, 0.25, 0.125])
        np.testing.assert_allclose(fit.predict([0, 2]), [fit.model.s0, fit.model.retrievability(2)])

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="align"):
            fit_similarity_decay([0, 1, 2], [1.0, 0.5])

    def test_rejects_too_few_points(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            fit_similarity_decay([1], [0.5])

    def test_rejects_nonpositive_similarity(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            fit_similarity_decay([0, 1], [0.5, 0.0])

    def test_rejects_constant_distance(self) -> None:
        with pytest.raises(ValueError, match="must vary"):
            fit_similarity_decay([2, 2, 2], [0.4, 0.3, 0.5])
