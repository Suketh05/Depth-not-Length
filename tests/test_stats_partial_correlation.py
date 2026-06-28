"""Tests for partial and semipartial correlation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from membench.stats.partial_correlation import (
    partial_correlation,
    partial_correlation_matrix,
)


class TestPrecisionMatrix:
    def test_golden_three_variable_partial(self) -> None:
        # Hand-computed golden. Pairwise correlations r_xy=0.6, r_xz=0.5, r_yz=0.4.
        #   r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))
        #          = (0.6 - 0.5 * 0.4) / sqrt((1 - 0.25)(1 - 0.16))
        #          = 0.40 / sqrt(0.75 * 0.84)
        #          = 0.40 / sqrt(0.63)
        #          = 0.40 / 0.7937253933
        #          = 0.5039526306
        corr = np.array(
            [
                [1.0, 0.6, 0.5],
                [0.6, 1.0, 0.4],
                [0.5, 0.4, 1.0],
            ]
        )
        pcorr = partial_correlation_matrix(corr)
        assert pcorr[0, 1] == pytest.approx(0.5039526306, abs=1e-9)
        assert pcorr[0, 1] == pytest.approx(pcorr[1, 0], abs=1e-12)  # symmetric
        assert pcorr[0, 0] == pytest.approx(1.0)  # unit diagonal

    def test_two_variables_recovers_pearson(self) -> None:
        # With no other variable to control for, partial corr == the raw correlation.
        r = 0.37
        corr = np.array([[1.0, r], [r, 1.0]])
        assert partial_correlation_matrix(corr)[0, 1] == pytest.approx(r)

    def test_singular_matrix_raises(self) -> None:
        with pytest.raises(ValueError, match="singular"):
            partial_correlation_matrix(np.array([[1.0, 1.0], [1.0, 1.0]]))

    def test_non_square_raises(self) -> None:
        with pytest.raises(ValueError, match="square"):
            partial_correlation_matrix(np.array([[1.0, 0.5, 0.5]]))


class TestPartialCorrelation:
    def test_formula_golden_partial_and_semipartial(self) -> None:
        # numpy's corrcoef is the independent oracle for the pairwise r's; the partial
        # and semipartial values are then the textbook closed forms of those r's.
        rng = np.random.default_rng(5)
        n = 300
        z = rng.normal(size=n)
        x = z + rng.normal(size=n)
        y = 0.5 * z + rng.normal(size=n)
        res = partial_correlation(x, y, [z])

        r_xy = float(np.corrcoef(x, y)[0, 1])
        r_xz = float(np.corrcoef(x, z)[0, 1])
        r_yz = float(np.corrcoef(y, z)[0, 1])
        # partial: z removed from both x and y.
        expected_partial = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        # semipartial: z removed from x only.
        expected_semi = (r_xy - r_xz * r_yz) / np.sqrt(1 - r_xz**2)
        assert res.partial == pytest.approx(expected_partial, abs=1e-10)
        assert res.semipartial == pytest.approx(expected_semi, abs=1e-10)
        assert res.n == n
        assert res.n_covariates == 1
        assert res.df == n - 3

    def test_uncorrelated_covariate_leaves_correlation_unchanged(self) -> None:
        # Invariant: controlling for a variable uncorrelated with both x and y does not
        # change the correlation between x and y.
        rng = np.random.default_rng(3)
        n = 5000
        x = rng.normal(size=n)
        y = 0.7 * x + rng.normal(size=n)
        z = rng.normal(size=n)  # independent of x and y
        raw = float(np.corrcoef(x, y)[0, 1])
        res = partial_correlation(x, y, [z])
        assert res.partial == pytest.approx(raw, abs=0.02)
        assert res.semipartial == pytest.approx(raw, abs=0.02)

    def test_no_covariates_reduces_to_pearson(self) -> None:
        # With k = 0 the result is the ordinary Pearson correlation and its t-test.
        rng = np.random.default_rng(11)
        n = 80
        x = rng.normal(size=n)
        y = 0.5 * x + rng.normal(size=n)
        res = partial_correlation(x, y)
        r, p = stats.pearsonr(x, y)
        assert res.partial == pytest.approx(r)
        assert res.semipartial == pytest.approx(r)
        assert res.df == n - 2
        assert res.p_value == pytest.approx(p, rel=1e-9, abs=1e-12)

    def test_partial_equals_residual_correlation_multi_covariate(self) -> None:
        # The precision-matrix partial must equal the correlation of the OLS residuals
        # of x and y on the covariates (an independent computation of the same quantity).
        rng = np.random.default_rng(7)
        n = 200
        z1 = rng.normal(size=n)
        z2 = rng.normal(size=n)
        x = z1 + 0.5 * z2 + rng.normal(size=n)
        y = z1 - 0.3 * z2 + rng.normal(size=n)
        res = partial_correlation(x, y, [z1, z2])

        design = np.column_stack([np.ones(n), z1, z2])
        bx, *_ = np.linalg.lstsq(design, x, rcond=None)
        by, *_ = np.linalg.lstsq(design, y, rcond=None)
        ex = x - design @ bx
        ey = y - design @ by
        expected = float(np.corrcoef(ex, ey)[0, 1])
        assert res.partial == pytest.approx(expected, abs=1e-10)
        assert res.n_covariates == 2
        assert res.df == n - 4

    def test_significant_partial_has_small_p(self) -> None:
        rng = np.random.default_rng(13)
        n = 500
        z = rng.normal(size=n)
        x = z + rng.normal(size=n)
        y = 0.6 * x + z + rng.normal(size=n)  # x->y over and above z
        res = partial_correlation(x, y, [z])
        assert res.partial > 0.3
        assert res.p_value < 1e-6


class TestValidation:
    def test_misaligned_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            partial_correlation([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_constant_variable(self) -> None:
        with pytest.raises(ValueError, match="non-zero variance"):
            partial_correlation([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0])

    def test_too_few_observations(self) -> None:
        with pytest.raises(ValueError, match="enough observations"):
            partial_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [[1.0, 0.0, 2.0]])

    def test_misaligned_covariate(self) -> None:
        with pytest.raises(ValueError, match="align"):
            partial_correlation([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], [[1.0, 2.0]])
