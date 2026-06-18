"""Tests for assembling the theory from measurements and predicting d*."""

from __future__ import annotations

import numpy as np
import pytest

from membench.theory.decay import fit_similarity_decay
from membench.theory.fit import (
    build_recovery_model,
    compare_crossover,
    estimate_link_reliability,
    predict_crossover,
)
from membench.theory.recovery import structured_recovery


class TestEstimateLinkReliability:
    def test_recovers_known_q(self) -> None:
        q_true = 0.92
        depths = [1, 2, 3, 4, 5]
        rates = [structured_recovery(d, q_true) for d in depths]
        fit = estimate_link_reliability(depths, rates)
        assert fit.q == pytest.approx(q_true, rel=1e-6)
        assert fit.r_squared == pytest.approx(1.0, abs=1e-9)
        assert fit.n_points == 5

    def test_q_clipped_to_one(self) -> None:
        # Perfect recovery at every depth implies q = 1.
        fit = estimate_link_reliability([1, 2, 3], [1.0, 1.0, 1.0])
        assert fit.q == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("depths", "rates", "match"),
        [
            ([1, 2], [0.9], "align"),
            ([], [], "at least one"),
            ([0, 1], [0.9, 0.8], ">= 1"),
            ([1, 2], [0.0, 0.8], r"\(0, 1\]"),
            ([1, 2], [0.9, 1.1], r"\(0, 1\]"),
        ],
    )
    def test_validation(self, depths: list[int], rates: list[float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            estimate_link_reliability(depths, rates)


class TestPredictAndCompare:
    def _decay_fit(self) -> object:
        # Clean geometric decay s0=0.9, rho=0.5.
        distances = list(range(0, 8))
        sims = [0.9 * 0.5**i for i in distances]
        return fit_similarity_decay(distances, sims)

    def test_build_recovery_model_uses_fitted_params(self) -> None:
        decay_fit = self._decay_fit()
        link_fit = estimate_link_reliability([1, 2, 3], [0.95, 0.95**2, 0.95**3])
        model = build_recovery_model(decay_fit, link_fit)  # type: ignore[arg-type]
        assert model.decay.rho == pytest.approx(0.5, rel=1e-6)
        assert model.q == pytest.approx(0.95, rel=1e-6)

    def test_predict_crossover_returns_result(self) -> None:
        decay_fit = self._decay_fit()
        link_fit = estimate_link_reliability([1, 2, 3], [0.99, 0.99**2, 0.99**3])
        res = predict_crossover(decay_fit, link_fit, tau=0.3)  # type: ignore[arg-type]
        assert res.exists
        assert res.d_star is not None and res.d_star >= 1

    def test_compare_crossover_within_tolerance(self) -> None:
        cmp = compare_crossover(predicted=2.3, measured=2.0, tolerance=1.0)
        assert cmp.absolute_error == pytest.approx(0.3)
        assert cmp.agrees

    def test_compare_crossover_outside_tolerance(self) -> None:
        cmp = compare_crossover(predicted=5.0, measured=2.0, tolerance=1.0)
        assert not cmp.agrees

    def test_compare_rejects_negative_tolerance(self) -> None:
        with pytest.raises(ValueError, match="tolerance"):
            compare_crossover(2.0, 2.0, tolerance=-1.0)

    def test_end_to_end_prediction_is_finite(self) -> None:
        decay_fit = self._decay_fit()
        link_fit = estimate_link_reliability([1, 2, 3, 4], [0.97**d for d in (1, 2, 3, 4)])
        res = predict_crossover(decay_fit, link_fit, tau=0.2, max_depth=40)
        assert np.isfinite(res.g_curve).all()
