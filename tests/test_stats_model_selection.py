"""Tests for decay-law model selection."""

from __future__ import annotations

import numpy as np
import pytest

from membench.stats.model_selection import aic, bic, compare_decay_models


class TestInformationCriteria:
    def test_more_params_penalised(self) -> None:
        # same fit, more params -> higher AIC/BIC
        assert aic(1.0, 50, 3) > aic(1.0, 50, 2)
        assert bic(1.0, 50, 3) > bic(1.0, 50, 2)


class TestCompareDecayModels:
    def test_geometric_wins_on_noisy_geometric_data(self) -> None:
        # Noisy geometric data: geometric and the (nesting) stretched-exp fit equally
        # well, so the AIC parsimony penalty makes the 2-param geometric the winner.
        rng = np.random.default_rng(0)
        i = np.arange(0, 12, dtype=float)
        s = 0.9 * 0.6**i * np.exp(rng.normal(0, 0.03, i.size))
        fits = compare_decay_models(i, s)
        best = min(fits.values(), key=lambda f: f.aic)
        assert best.name == "geometric"
        assert fits["geometric"].params["rho"] == pytest.approx(0.6, rel=0.05)
        assert fits["geometric"].aic <= fits["stretched_exp"].aic  # parsimony

    def test_power_law_competitive_on_power_data(self) -> None:
        i = np.arange(0, 10, dtype=float)
        s = 0.8 * (i + 1.0) ** -1.5  # exactly power-law
        fits = compare_decay_models(i, s)
        best = min(fits.values(), key=lambda f: f.aic)
        assert best.name == "power_law"

    def test_all_three_models_present(self) -> None:
        i = np.arange(0, 8, dtype=float)
        s = 0.9 * 0.7**i
        fits = compare_decay_models(i, s)
        assert set(fits) == {"geometric", "power_law", "stretched_exp"}

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="four"):
            compare_decay_models([0, 1], [1.0, 0.5])
        with pytest.raises(ValueError, match="positive"):
            compare_decay_models([0, 1, 2, 3], [1.0, 0.5, 0.0, 0.1])
