"""Tests for split (inductive) conformal prediction intervals.

Golden values are derived by hand from the split-conformal definition
``q = R_(k)`` with ``k = ceil((n + 1)(1 - alpha))`` (Lei et al. 2018), never by
running the implementation under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from membench.stats.conformal import (
    ConformalInterval,
    conformal_quantile,
    split_conformal_interval,
)

# Fixed calibration residual set used for the hand-computed goldens below.
# Sorted ascending: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  (n = 9).
RESIDUALS = [0.2, 0.5, 0.1, 0.9, 0.4, 0.7, 0.3, 0.8, 0.6]


class TestConformalQuantileGolden:
    def test_interior_order_statistic(self) -> None:
        # n = 9, alpha = 0.2 -> k = ceil(10 * 0.8) = ceil(8.0) = 8
        # -> 8th smallest residual = 0.8.
        assert conformal_quantile(RESIDUALS, alpha=0.2) == pytest.approx(0.8)

    def test_top_order_statistic(self) -> None:
        # n = 9, alpha = 0.1 -> k = ceil(10 * 0.9) = ceil(9.0) = 9
        # -> 9th smallest (the maximum) = 0.9.
        assert conformal_quantile(RESIDUALS, alpha=0.1) == pytest.approx(0.9)

    def test_infinite_when_n_too_small(self) -> None:
        # n = 9, alpha = 0.05 -> k = ceil(10 * 0.95) = ceil(9.5) = 10 > 9
        # -> level unattainable in finite samples -> +inf.
        assert conformal_quantile(RESIDUALS, alpha=0.05) == math.inf

    def test_single_residual_repeats_it(self) -> None:
        # n = 1, alpha = 0.1 -> k = ceil(2 * 0.9) = 2 > 1 -> +inf.
        assert conformal_quantile([3.5], alpha=0.1) == math.inf
        # n = 1, alpha = 0.6 -> k = ceil(2 * 0.4) = 1 -> the only residual.
        assert conformal_quantile([3.5], alpha=0.6) == pytest.approx(3.5)


class TestConformalQuantileInvariants:
    def test_monotone_nonincreasing_in_alpha(self) -> None:
        # Smaller alpha (higher coverage) -> wider band -> larger quantile.
        widths = [conformal_quantile(RESIDUALS, alpha=a) for a in (0.5, 0.3, 0.2, 0.1)]
        finite = [w for w in widths if math.isfinite(w)]
        assert finite == sorted(finite)
        assert all(0.0 <= w <= max(RESIDUALS) for w in finite)

    def test_validates_alpha(self) -> None:
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="alpha must be in"):
                conformal_quantile(RESIDUALS, alpha=bad)

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            conformal_quantile([])


class TestSplitConformalInterval:
    def test_symmetric_band_around_prediction(self) -> None:
        # Residuals all derived from |y - yhat|; with alpha=0.2 the half-width is 0.8.
        y_true = [1.2, 0.5, 1.1, 0.1, 0.6, 0.3, 0.7, 0.2, 0.4]
        y_pred = [1.0] * 9
        # |y - yhat| = [0.2,0.5,0.1,0.9,0.4,0.7,0.3,0.8,0.6] == RESIDUALS.
        interval = split_conformal_interval(y_true, y_pred, prediction=5.0, alpha=0.2)
        assert isinstance(interval, ConformalInterval)
        assert interval.half_width == pytest.approx(0.8)
        assert interval.point == pytest.approx(5.0)
        assert interval.low == pytest.approx(4.2)
        assert interval.high == pytest.approx(5.8)
        assert interval.coverage == pytest.approx(0.8)
        assert interval.low < interval.point < interval.high

    def test_infinite_bounds_when_underdetermined(self) -> None:
        interval = split_conformal_interval([0.0, 1.0], [0.0, 0.0], 2.0, alpha=0.1)
        assert interval.half_width == math.inf
        assert interval.low == -math.inf
        assert interval.high == math.inf

    def test_rejects_misaligned_calibration(self) -> None:
        with pytest.raises(ValueError, match="align"):
            split_conformal_interval([1.0, 2.0], [1.0], 0.0)

    def test_rejects_non_1d(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            split_conformal_interval([[1.0, 2.0]], [[1.0, 2.0]], 0.0)


class TestCoverageInvariant:
    def test_marginal_coverage_at_least_one_minus_alpha(self) -> None:
        """On exchangeable synthetic data, empirical coverage >= 1 - alpha.

        Split conformal guarantees ``E[coverage] = ceil((n+1)(1-alpha))/(n+1)``, which
        for n=18, alpha=0.1 equals 18/19 ~= 0.947 -- comfortably above 0.9. We estimate
        this expectation over many seeded calibration/test resplits (a fixed seed makes
        the whole test deterministic) and assert the marginal coverage clears 1 - alpha.
        """
        rng = np.random.default_rng(0)
        alpha = 0.1
        n_cal, n_test, trials = 18, 500, 400
        slope, sigma = 2.0, 1.0
        coverages: list[float] = []
        for _ in range(trials):
            x_cal = rng.normal(size=n_cal)
            y_cal = slope * x_cal + rng.normal(0.0, sigma, size=n_cal)
            q = conformal_quantile(np.abs(y_cal - slope * x_cal), alpha=alpha)
            x_test = rng.normal(size=n_test)
            y_test = slope * x_test + rng.normal(0.0, sigma, size=n_test)
            covered = np.abs(y_test - slope * x_test) <= q
            coverages.append(float(np.mean(covered)))
        marginal = float(np.mean(coverages))
        assert marginal >= 1.0 - alpha
