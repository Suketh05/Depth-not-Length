"""Tests for linear quantile regression (Koenker & Bassett 1978).

Golden values are derived independently (by hand or from the textbook definition),
never by running the implementation under test.
"""

from __future__ import annotations

import pytest

from membench.stats.quantile_regression import (
    pinball_loss,
    quantile_regression,
    quantile_regression_path,
)


class TestPinballLoss:
    def test_hand_computed_value(self) -> None:
        # residuals u = [2, -1, 3], tau = 0.7. rho(u) = u*(tau - 1{u<0}):
        #   u=2  (>=0): 0.7 * 2          = 1.4
        #   u=-1 (<0):  (0.7 - 1) * -1   = 0.3
        #   u=3  (>=0): 0.7 * 3          = 2.1
        # mean = (1.4 + 0.3 + 2.1) / 3 = 3.8 / 3 = 1.2666...
        assert pinball_loss([2.0, -1.0, 3.0], 0.7) == pytest.approx(3.8 / 3.0)

    def test_median_equals_half_mean_abs(self) -> None:
        # At tau = 0.5 the tilted loss is exactly 0.5*|u|, so the mean is
        # 0.5 * mean|residual|. For [2, -1, 3]: 0.5 * (2 + 1 + 3) / 3 = 1.0.
        residuals = [2.0, -1.0, 3.0]
        assert pinball_loss(residuals, 0.5) == pytest.approx(1.0)
        assert pinball_loss(residuals, 0.5) == pytest.approx(
            0.5 * (abs(2.0) + abs(-1.0) + abs(3.0)) / 3.0
        )

    def test_rejects_bad_tau(self) -> None:
        with pytest.raises(ValueError, match="tau must be in"):
            pinball_loss([1.0], 0.0)
        with pytest.raises(ValueError, match="tau must be in"):
            pinball_loss([1.0], 1.0)

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            pinball_loss([], 0.5)


class TestQuantileRegression:
    def test_median_recovers_exact_line(self) -> None:
        # Exactly linear data y = 2*x + 1; the median fit must recover (slope, intercept)
        # = (2, 1) with zero residuals, hence zero pinball loss.
        x = [1.0, 2.0, 3.0, 4.0]
        y = [3.0, 5.0, 7.0, 9.0]
        fit = quantile_regression(x, y, tau=0.5)
        assert fit.slope == pytest.approx(2.0, abs=1e-7)
        assert fit.intercept == pytest.approx(1.0, abs=1e-7)
        assert fit.pinball_loss == pytest.approx(0.0, abs=1e-9)
        assert fit.n_obs == 4
        assert fit.tau == 0.5
        assert fit.predict(5.0) == pytest.approx(11.0, abs=1e-7)

    def test_fit_loss_matches_definition_at_median(self) -> None:
        # The stored pinball_loss at tau=0.5 must equal 0.5*mean|residual|.
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [0.1, 1.4, 1.8, 3.5, 3.9]
        fit = quantile_regression(x, y, tau=0.5)
        residuals = [yi - fit.predict(xi) for xi, yi in zip(x, y, strict=True)]
        mean_abs = sum(abs(r) for r in residuals) / len(residuals)
        assert fit.pinball_loss == pytest.approx(0.5 * mean_abs)

    def test_extreme_taus_shift_intercept(self) -> None:
        # Two parallel "lines" of points: y = x (lower) and y = x + 2 (upper). The high
        # quantile tracks the upper points (intercept ~ 2), the low quantile the lower
        # points (intercept ~ 0), so the upper-quantile intercept is strictly larger.
        x = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
        y = [0.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0]
        low = quantile_regression(x, y, tau=0.1)
        high = quantile_regression(x, y, tau=0.9)
        assert high.intercept > low.intercept
        assert low.intercept == pytest.approx(0.0, abs=1e-6)
        assert high.intercept == pytest.approx(2.0, abs=1e-6)
        # both quantile lines share the common slope of the parallel structure
        assert low.slope == pytest.approx(1.0, abs=1e-6)
        assert high.slope == pytest.approx(1.0, abs=1e-6)

    def test_path_returns_one_fit_per_tau_increasing(self) -> None:
        x = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
        y = [0.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0]
        fits = quantile_regression_path(x, y, taus=(0.1, 0.5, 0.9))
        assert len(fits) == 3
        assert [f.tau for f in fits] == [0.1, 0.5, 0.9]
        # quantile intercepts are non-crossing here: lower tau, lower intercept
        assert fits[0].intercept <= fits[1].intercept <= fits[2].intercept

    def test_rejects_bad_tau(self) -> None:
        with pytest.raises(ValueError, match="tau must be in"):
            quantile_regression([0.0, 1.0], [0.0, 1.0], tau=1.5)

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            quantile_regression([0.0, 1.0, 2.0], [0.0, 1.0])

    def test_rejects_too_few_observations(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            quantile_regression([1.0], [1.0])
