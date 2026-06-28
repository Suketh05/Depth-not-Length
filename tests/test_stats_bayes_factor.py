"""Tests for Bayes factors on binomial proportions.

Golden values are derived independently of the implementation: the single-proportion
Bayes factor has an exact closed form under a uniform prior, and the two-proportion BIC
factor is recomputed here from the raw Schwarz formula.
"""

from __future__ import annotations

import math

import pytest

from membench.stats.bayes_factor import (
    bayes_factor_proportion,
    bayes_factor_two_proportion_bic,
)


class TestProportionBayesFactor:
    def test_uniform_prior_closed_form_golden(self) -> None:
        # Beta(1,1) prior, s=9, n=10, p0=0.5.
        # m1 = B(1+9, 1+1) / B(1,1) = B(10, 2) = 9! * 1! / 11! = 1/110.
        # m0 = p0^s (1-p0)^(n-s) = 0.5^10 = 1/1024.
        # BF10 = m1 / m0 = (1/110) / (1/1024) = 1024/110 = 9.30909...
        result = bayes_factor_proportion(9, 10, p0=0.5, prior=(1.0, 1.0))
        assert result.bf10 == pytest.approx(1024.0 / 110.0)
        assert result.log_bf == pytest.approx(math.log(1024.0 / 110.0))
        assert result.interpretation == "moderate evidence for H1"  # 3 <= 9.31 < 10

    def test_log_bf_matches_bf10(self) -> None:
        result = bayes_factor_proportion(7, 12, p0=0.3, prior=(2.0, 3.0))
        assert result.log_bf == pytest.approx(math.log(result.bf10))

    def test_symmetry_under_success_failure_swap(self) -> None:
        # With p0=0.5 and a symmetric Beta(a, a) prior, swapping s <-> n-s leaves the
        # Beta marginal and the null likelihood unchanged, so BF10 is invariant.
        lo = bayes_factor_proportion(3, 10, p0=0.5, prior=(1.0, 1.0))
        hi = bayes_factor_proportion(7, 10, p0=0.5, prior=(1.0, 1.0))
        assert lo.bf10 == pytest.approx(hi.bf10)

    def test_monotone_in_distance_from_null(self) -> None:
        # Evidence for H1 grows as the observed rate moves away from p0=0.5.
        bf = [bayes_factor_proportion(s, 20, p0=0.5, prior=(1.0, 1.0)).bf10 for s in (10, 15, 19)]
        assert bf[0] < bf[1] < bf[2]

    def test_data_at_null_favours_null(self) -> None:
        # s/n == p0 exactly -> evidence should favour H0 (BF10 < 1).
        result = bayes_factor_proportion(10, 20, p0=0.5, prior=(1.0, 1.0))
        assert result.bf10 < 1.0
        assert "for H0" in result.interpretation

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="successes"):
            bayes_factor_proportion(5, 3)
        with pytest.raises(ValueError, match="p0"):
            bayes_factor_proportion(1, 2, p0=0.0)
        with pytest.raises(ValueError, match="prior"):
            bayes_factor_proportion(1, 2, prior=(0.0, 1.0))


class TestTwoProportionBicBayesFactor:
    def test_bic_closed_form_golden(self) -> None:
        # n_a=n_b=100, s_a=60, s_b=40. Independently recompute the Schwarz factor:
        #   pooled p = 100/200 = 0.5
        #   ll0 = 200 * ln(0.5)
        #   ll1 = 60 ln.6 + 40 ln.4 + 40 ln.4 + 60 ln.6
        #   BIC0 = 1*ln(200) - 2 ll0 ;  BIC1 = 2*ln(200) - 2 ll1
        #   BF10 = exp((BIC0 - BIC1)/2)  ~= 3.9667
        n = 200
        ll0 = 200 * math.log(0.5)
        ll1 = 60 * math.log(0.6) + 40 * math.log(0.4) + 40 * math.log(0.4) + 60 * math.log(0.6)
        bic0 = 1 * math.log(n) - 2 * ll0
        bic1 = 2 * math.log(n) - 2 * ll1
        expected = math.exp((bic0 - bic1) / 2)
        result = bayes_factor_two_proportion_bic(60, 100, 40, 100)
        assert result.bf10 == pytest.approx(expected)
        assert result.bf10 == pytest.approx(3.9667, abs=1e-3)
        assert result.log_bf == pytest.approx(math.log(expected))

    def test_equal_rates_favour_null(self) -> None:
        # Identical fit -> only the BIC parameter penalty differs.
        # ll1 == ll0 so BF10 = exp(-ln(N)/2) = N^-0.5 < 1, with N = n_a + n_b = 160.
        result = bayes_factor_two_proportion_bic(50, 100, 30, 60)  # both 0.5
        assert result.bf10 == pytest.approx(160.0**-0.5)
        assert result.bf10 < 1.0
        assert "for H0" in result.interpretation

    def test_group_swap_symmetry(self) -> None:
        ab = bayes_factor_two_proportion_bic(60, 100, 40, 100)
        ba = bayes_factor_two_proportion_bic(40, 100, 60, 100)
        assert ab.bf10 == pytest.approx(ba.bf10)

    def test_monotone_in_separation(self) -> None:
        near = bayes_factor_two_proportion_bic(52, 100, 48, 100).bf10
        far = bayes_factor_two_proportion_bic(70, 100, 30, 100).bf10
        assert far > near

    def test_boundary_counts_finite(self) -> None:
        # All-success vs all-failure: MLEs at 0 and 1 must not blow up (0*log0 = 0).
        result = bayes_factor_two_proportion_bic(100, 100, 0, 100)
        assert math.isfinite(result.bf10)
        assert result.bf10 > 1.0

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="trial"):
            bayes_factor_two_proportion_bic(1, 0, 1, 5)
        with pytest.raises(ValueError, match="successes"):
            bayes_factor_two_proportion_bic(6, 5, 1, 5)
