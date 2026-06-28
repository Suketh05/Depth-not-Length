"""Tests for isotonic regression (PAVA) and the monotone trend test.

Golden values are derived by hand from the classic PAVA example, independently of
the implementation under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from membench.stats.isotonic import (
    isotonic_regression,
    monotone_trend_statistic,
    monotone_trend_test,
)


class TestIsotonicRegression:
    def test_classic_increasing_golden(self) -> None:
        # Hand-computed PAVA on the textbook input [1, 2, 3, 2, 1], non-decreasing fit.
        # Scan L->R pooling on violations:
        #   blocks [1],[2],[3]                       (1<=2<=3, no violation)
        #   add 2: 3>2 -> pool (3+2)/2 = 2.5         -> [1],[2],[2.5(w2)]
        #   add 1: 2.5>1 -> pool (2.5*2+1)/3 = 2.0   -> [1],[2],[2.0(w3)]
        # Expanded fit = [1, 2, 2, 2, 2].
        fit = isotonic_regression([1, 2, 3, 2, 1])
        np.testing.assert_array_equal(fit, np.array([1.0, 2.0, 2.0, 2.0, 2.0]))

    def test_classic_decreasing_golden(self) -> None:
        # Non-increasing fit of the same input is the mirror image: [2, 2, 2, 2, 1].
        fit = isotonic_regression([1, 2, 3, 2, 1], increasing=False)
        np.testing.assert_array_equal(fit, np.array([2.0, 2.0, 2.0, 2.0, 1.0]))

    def test_fit_is_monotone(self) -> None:
        fit = isotonic_regression([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
        assert np.all(np.diff(fit) >= -1e-12)

    def test_identity_on_already_monotone(self) -> None:
        # A non-decreasing input is its own isotonic fit.
        y = [0.0, 0.5, 0.5, 1.0, 2.0]
        np.testing.assert_array_equal(isotonic_regression(y), np.array(y))

    def test_idempotent(self) -> None:
        y = [5.0, 1.0, 3.0, 2.0, 8.0, 4.0]
        once = isotonic_regression(y)
        twice = isotonic_regression(once)
        np.testing.assert_allclose(twice, once)

    def test_preserves_weighted_mean(self) -> None:
        # The L2 projection preserves the (weighted) total: sum of fit == sum of data.
        y = [4.0, 1.0, 2.0, 3.0]
        fit = isotonic_regression(y)
        assert fit.sum() == pytest.approx(sum(y))

    def test_weighted_pooling(self) -> None:
        # Weights [1, 3] on [2, 0] -> pooled mean (2*1 + 0*3)/4 = 0.5 for both.
        fit = isotonic_regression([2.0, 0.0], weights=[1.0, 3.0])
        np.testing.assert_allclose(fit, np.array([0.5, 0.5]))

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            isotonic_regression([])
        with pytest.raises(ValueError, match="weights must match"):
            isotonic_regression([1.0, 2.0], weights=[1.0])
        with pytest.raises(ValueError, match="strictly positive"):
            isotonic_regression([1.0, 2.0], weights=[1.0, 0.0])


class TestMonotoneTrendStatistic:
    def test_golden_partial_fit(self) -> None:
        # For [1, 2, 3, 2, 1] increasing, fit = [1, 2, 2, 2, 2], mean = 1.8.
        # SS_total = 0.64+0.04+1.44+0.04+0.64 = 2.8 ; SS_resid = 0+0+1+0+1 = 2.0.
        # R^2_iso = 1 - 2.0/2.8 = 2/7 ~= 0.2857142857.
        stat = monotone_trend_statistic([1, 2, 3, 2, 1])
        assert stat == pytest.approx(2.0 / 7.0)

    def test_perfectly_monotone_is_one(self) -> None:
        assert monotone_trend_statistic([3.0, 2.0, 1.0], increasing=False) == pytest.approx(1.0)

    def test_constant_is_one(self) -> None:
        assert monotone_trend_statistic([0.7, 0.7, 0.7]) == pytest.approx(1.0)

    def test_opposite_trend_is_zero(self) -> None:
        # A strictly increasing series has no non-increasing structure: best fit is flat.
        assert monotone_trend_statistic([0.1, 0.5, 0.9], increasing=False) == pytest.approx(0.0)


class TestMonotoneTrendTest:
    def test_perfect_decrease_golden(self) -> None:
        # Depth means: d1=1.0 (n=2), d2=0.5 (n=2), d3=0.0 (n=2); already non-increasing.
        depths = [1, 1, 2, 2, 3, 3]
        compliance = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        res = monotone_trend_test(depths, compliance)
        assert res.depths == (1, 2, 3)
        assert res.n_by_depth == (2, 2, 2)
        assert res.fitted == pytest.approx((1.0, 0.5, 0.0))
        assert res.statistic == pytest.approx(1.0)
        assert res.decreasing is True

    def test_pooling_when_not_monotone_golden(self) -> None:
        # One obs per depth: means [0.2, 0.8, 0.5]. Non-increasing PAVA:
        #   0.2 then 0.8 violates -> pool (0.2+0.8)/2 = 0.5 ; then 0.5 ties -> [0.5,0.5,0.5].
        # Weighted mean = 0.5, fit == mean -> R^2_iso = 0.
        res = monotone_trend_test([1, 2, 3], [0.2, 0.8, 0.5])
        assert res.fitted == pytest.approx((0.5, 0.5, 0.5))
        assert res.statistic == pytest.approx(0.0)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="align"):
            monotone_trend_test([1, 2], [1.0])
        with pytest.raises(ValueError, match="at least one"):
            monotone_trend_test([], [])
