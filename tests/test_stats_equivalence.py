"""Tests for Schuirmann's TOST equivalence tests.

The headline golden is hand-computed from first principles (closed-form t
statistics and degrees of freedom), then turned into reference p-values via the
t-distribution function evaluated on those *independently derived* statistics --
never by reading the module under test back to itself.
"""

from __future__ import annotations

import math

import pytest
from scipy import stats

from membench.stats.equivalence import (
    TOSTResult,
    tost_independent_samples,
    tost_two_means,
    tost_two_proportions,
)


class TestTwoMeansGolden:
    r"""Textbook two-sample TOST worked example.

    Group A: mean = 10.0, sd = 2.0, n = 50.
    Group B: mean = 10.5, sd = 2.0, n = 50.   margin delta = 1.0, alpha = 0.05.

    Hand computation (equal sd & n => Welch df collapses to 2*(n-1)):
        diff = 10.0 - 10.5            = -0.5
        SE   = sqrt(2^2/50 + 2^2/50)  = sqrt(0.16) = 0.40
        df   = (0.16)^2 / (2 * (0.08^2 / 49)) = 98.0
        t_lower = (diff + delta)/SE = (-0.5 + 1.0)/0.4 =  1.25
        t_upper = (diff - delta)/SE = (-0.5 - 1.0)/0.4 = -3.75
    Reference p-values from the t(98) distribution evaluated on the hand t stats:
        lower_p = P(T >  1.25) = 0.1071381...   (the binding, larger one)
        upper_p = P(T < -3.75) = 0.0001496...
        p_value = max(...) = 0.1071381 > 0.05  => NOT equivalent.
    90% (= 1 - 2*alpha) CI: -0.5 +/- t_{.95,98}*0.4 = (-1.1642, 0.1642),
    whose lower bound -1.1642 < -1.0, i.e. it pokes outside [-1, 1] -- the same
    not-equivalent verdict by the interval-inclusion rule.
    """

    MEAN_A, SD_A, N_A = 10.0, 2.0, 50
    MEAN_B, SD_B, N_B = 10.5, 2.0, 50
    DELTA = 1.0

    def _golden(self) -> tuple[float, float]:
        # Derived independently of the module: hand t stats -> t(98) tail areas.
        return float(stats.t.sf(1.25, 98)), float(stats.t.cdf(-3.75, 98))

    def test_one_sided_p_values_match_hand_computation(self) -> None:
        res = tost_two_means(
            self.MEAN_A,
            self.SD_A,
            self.N_A,
            self.MEAN_B,
            self.SD_B,
            self.N_B,
            delta=self.DELTA,
        )
        exp_lower, exp_upper = self._golden()
        # Literal golden values (independent of this code base).
        assert exp_lower == pytest.approx(0.1071381, abs=1e-6)
        assert exp_upper == pytest.approx(0.0001496, abs=1e-6)
        assert res.lower_p == pytest.approx(exp_lower, rel=1e-9)
        assert res.upper_p == pytest.approx(exp_upper, rel=1e-9)

    def test_overall_p_is_max_and_not_equivalent(self) -> None:
        res = tost_two_means(
            self.MEAN_A,
            self.SD_A,
            self.N_A,
            self.MEAN_B,
            self.SD_B,
            self.N_B,
            delta=self.DELTA,
        )
        assert res.p_value == pytest.approx(0.1071381, abs=1e-6)
        assert res.p_value == max(res.lower_p, res.upper_p)
        assert res.equivalent is False

    def test_confidence_interval_matches_hand_computation(self) -> None:
        res = tost_two_means(
            self.MEAN_A,
            self.SD_A,
            self.N_A,
            self.MEAN_B,
            self.SD_B,
            self.N_B,
            delta=self.DELTA,
        )
        crit = float(stats.t.ppf(0.95, 98))  # 1.6605512...
        assert res.ci[0] == pytest.approx(-0.5 - crit * 0.4, rel=1e-9)
        assert res.ci[1] == pytest.approx(-0.5 + crit * 0.4, rel=1e-9)
        assert res.ci[0] == pytest.approx(-1.1642205, abs=1e-6)
        # interval-inclusion rule agrees with the p-value verdict
        inside = res.ci[1] <= self.DELTA and res.ci[0] >= -self.DELTA
        assert inside is res.equivalent

    def test_pooled_matches_welch_for_equal_design(self) -> None:
        # Equal sd & n => pooled and Welch are identical here.
        welch = tost_two_means(
            self.MEAN_A,
            self.SD_A,
            self.N_A,
            self.MEAN_B,
            self.SD_B,
            self.N_B,
            delta=self.DELTA,
        )
        pooled = tost_two_means(
            self.MEAN_A,
            self.SD_A,
            self.N_A,
            self.MEAN_B,
            self.SD_B,
            self.N_B,
            delta=self.DELTA,
            pooled=True,
        )
        assert pooled.p_value == pytest.approx(welch.p_value, rel=1e-12)
        assert pooled.ci == pytest.approx(welch.ci, rel=1e-12)


class TestMarginExtremes:
    MEAN_A, SD_A, N_A = 10.0, 2.0, 50
    MEAN_B, SD_B, N_B = 10.5, 2.0, 50

    def test_huge_margin_is_equivalent(self) -> None:
        res = tost_two_means(
            self.MEAN_A,
            self.SD_A,
            self.N_A,
            self.MEAN_B,
            self.SD_B,
            self.N_B,
            delta=100.0,
        )
        assert res.equivalent is True
        assert res.p_value < 1e-6

    def test_tiny_margin_is_not_equivalent(self) -> None:
        res = tost_two_means(
            self.MEAN_A,
            self.SD_A,
            self.N_A,
            self.MEAN_B,
            self.SD_B,
            self.N_B,
            delta=0.001,
        )
        assert res.equivalent is False
        assert res.p_value > 0.5


class TestProportions:
    def test_symmetric_equal_proportions_lower_equals_upper(self) -> None:
        # p_a == p_b => diff = 0 => the two one-sided tests are mirror images.
        res = tost_two_proportions(40, 100, 40, 100, delta=0.2)
        assert res.lower_p == pytest.approx(res.upper_p, rel=1e-12)
        assert res.equivalent is True

    def test_golden_proportion_z_statistics(self) -> None:
        # successes 45/100 vs 40/100, delta = 0.2, alpha = 0.05.
        # p_a=0.45, p_b=0.40, diff=0.05
        # SE = sqrt(.45*.55/100 + .40*.60/100) = sqrt(0.002475 + 0.0024) = sqrt(0.004875)
        se = math.sqrt(0.45 * 0.55 / 100 + 0.40 * 0.60 / 100)
        z_lower = (0.05 + 0.2) / se
        z_upper = (0.05 - 0.2) / se
        exp_lower = float(stats.norm.sf(z_lower))
        exp_upper = float(stats.norm.cdf(z_upper))
        res = tost_two_proportions(45, 100, 40, 100, delta=0.2)
        assert res.lower_p == pytest.approx(exp_lower, rel=1e-9)
        assert res.upper_p == pytest.approx(exp_upper, rel=1e-9)
        assert res.p_value == pytest.approx(max(exp_lower, exp_upper), rel=1e-9)

    def test_huge_margin_proportions_equivalent(self) -> None:
        assert tost_two_proportions(45, 100, 40, 100, delta=0.5).equivalent is True


class TestSampleWrapper:
    def test_wrapper_matches_summary_stats(self) -> None:
        import numpy as np

        rng = np.random.default_rng(0)
        a = rng.normal(0.0, 1.0, 80)
        b = rng.normal(0.05, 1.0, 80)
        from_samples = tost_independent_samples(a, b, delta=0.5)
        from_summary = tost_two_means(
            float(a.mean()),
            float(a.std(ddof=1)),
            a.size,
            float(b.mean()),
            float(b.std(ddof=1)),
            b.size,
            delta=0.5,
        )
        assert from_samples.p_value == pytest.approx(from_summary.p_value, rel=1e-12)
        assert from_samples.ci == pytest.approx(from_summary.ci, rel=1e-12)


class TestStructureAndValidation:
    def test_result_is_frozen_dataclass(self) -> None:
        res = tost_two_means(10.0, 2.0, 50, 10.5, 2.0, 50, delta=1.0)
        assert isinstance(res, TOSTResult)
        assert isinstance(res.equivalent, bool)
        assert len(res.ci) == 2
        with pytest.raises(AttributeError):
            res.p_value = 0.0  # type: ignore[misc]

    def test_p_values_are_probabilities(self) -> None:
        res = tost_two_means(10.0, 2.0, 50, 10.5, 2.0, 50, delta=1.0)
        for p in (res.p_value, res.lower_p, res.upper_p):
            assert 0.0 <= p <= 1.0

    def test_negative_margin_rejected(self) -> None:
        with pytest.raises(ValueError, match="delta"):
            tost_two_means(10.0, 2.0, 50, 10.5, 2.0, 50, delta=-1.0)

    def test_bad_alpha_rejected(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            tost_two_means(10.0, 2.0, 50, 10.5, 2.0, 50, delta=1.0, alpha=0.9)

    def test_too_few_observations_rejected(self) -> None:
        with pytest.raises(ValueError, match="two observations"):
            tost_two_means(10.0, 2.0, 1, 10.5, 2.0, 50, delta=1.0)

    def test_proportion_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="successes"):
            tost_two_proportions(150, 100, 40, 100, delta=0.2)
