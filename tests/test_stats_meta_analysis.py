"""Tests for the DerSimonian-Laird random-effects meta-analysis.

The golden values for the 3-study example are derived by hand (not by running the
implementation) from the DerSimonian-Laird (1986) formulae.

Example: effects y = [0.1, 0.3, 0.5], variances v = [0.02, 0.03, 0.05].
Fixed-effect weights w = 1/v = [50, 100/3, 20], so sum_w = 310/3.

    fixed_effect = sum(w*y)/sum_w = 25 / (310/3) = 75/310 = 15/62 = 0.241935484

    Q = sum w*(y - fixed)^2
      = 50*(22/155)^2 + (100/3)*(9/155)^2 + 20*(8/31)^2
      = 24200/24025 + 8100/72075 + 1280/961
      = 76/31 = 2.451612903

    C = sum_w - sum(w^2)/sum_w
      = 310/3 - (2500 + 10000/9 + 400)/(310/3) = 2000/31 = 64.516129

    tau^2 = (Q - (k-1)) / C = (76/31 - 2) / (2000/31)
          = (14/31) / (2000/31) = 14/2000 = 0.007   (exactly)

    I^2 = (Q - (k-1)) / Q = (14/31) / (76/31) = 14/76 = 7/38 = 0.184210526

Random-effects weights w* = 1/(v + 0.007) = [1/0.027, 1/0.037, 1/0.057]:

    pooled = sum(w*  y)/sum(w*) = 20.583741636 / 81.607923713 = 0.252227243
"""

from __future__ import annotations

import math

import pytest

from membench.stats.meta_analysis import MetaAnalysisResult, dersimonian_laird

EFFECTS = [0.1, 0.3, 0.5]
VARIANCES = [0.02, 0.03, 0.05]

# Hand-computed golden values (see module docstring above).
GOLD_FIXED = 15.0 / 62.0  # 0.241935484
GOLD_Q = 76.0 / 31.0  # 2.451612903
GOLD_TAU2 = 0.007  # exact
GOLD_I2 = 7.0 / 38.0  # 0.184210526
GOLD_POOLED = 0.2522272434


class TestGoldenThreeStudy:
    def test_matches_hand_computed_values(self) -> None:
        r = dersimonian_laird(EFFECTS, VARIANCES)
        assert r.fixed_effect == pytest.approx(GOLD_FIXED, abs=1e-9)
        assert r.q == pytest.approx(GOLD_Q, abs=1e-9)
        assert r.tau_squared == pytest.approx(GOLD_TAU2, abs=1e-9)
        assert r.i_squared == pytest.approx(GOLD_I2, abs=1e-9)
        assert r.pooled_effect == pytest.approx(GOLD_POOLED, abs=1e-7)
        assert r.k == 3

    def test_confidence_interval_is_normal_around_pooled(self) -> None:
        r = dersimonian_laird(EFFECTS, VARIANCES, confidence=0.95)
        # se = sqrt(1 / sum(1/(v+tau^2))) = sqrt(1/81.607923713) = 0.110696486
        assert r.standard_error == pytest.approx(0.110696486, abs=1e-7)
        # z_{0.975} = 1.959963985 -> half-width = z * se = 0.216962
        z = 1.959963985
        assert r.ci_low == pytest.approx(GOLD_POOLED - z * r.standard_error, abs=1e-9)
        assert r.ci_high == pytest.approx(GOLD_POOLED + z * r.standard_error, abs=1e-7)
        assert r.ci_low < r.pooled_effect < r.ci_high


class TestInvariants:
    def test_identical_studies_have_zero_heterogeneity(self) -> None:
        # When every study reports the same effect, Q = 0 so tau^2 and I^2 are 0,
        # and the pooled estimate equals that common effect.
        r = dersimonian_laird([0.4, 0.4, 0.4], [0.01, 0.02, 0.05])
        assert r.q == pytest.approx(0.0, abs=1e-12)
        assert r.tau_squared == 0.0
        assert r.i_squared == 0.0
        assert r.pooled_effect == pytest.approx(0.4)
        assert r.fixed_effect == pytest.approx(0.4)

    def test_homogeneous_pool_reduces_to_fixed_effect(self) -> None:
        # With tau^2 = 0 the random-effects pool coincides with the fixed-effect pool.
        r = dersimonian_laird([0.2, 0.2, 0.2], [0.01, 0.01, 0.01])
        assert r.tau_squared == 0.0
        assert r.pooled_effect == pytest.approx(r.fixed_effect)

    def test_heterogeneity_widens_interval_vs_fixed(self) -> None:
        # Random-effects SE >= fixed-effect SE when tau^2 > 0.
        r = dersimonian_laird(EFFECTS, VARIANCES)
        assert r.tau_squared > 0.0
        fixed_se = math.sqrt(1.0 / sum(1.0 / v for v in VARIANCES))
        assert r.standard_error > fixed_se

    def test_returns_frozen_dataclass(self) -> None:
        r = dersimonian_laird(EFFECTS, VARIANCES)
        assert isinstance(r, MetaAnalysisResult)
        with pytest.raises((AttributeError, TypeError)):
            r.pooled_effect = 0.0  # type: ignore[misc]


class TestValidation:
    def test_rejects_single_study(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            dersimonian_laird([0.1], [0.02])

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            dersimonian_laird([0.1, 0.2], [0.02])

    def test_rejects_nonpositive_variance(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            dersimonian_laird([0.1, 0.2, 0.3], [0.02, 0.0, 0.05])

    def test_rejects_bad_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            dersimonian_laird(EFFECTS, VARIANCES, confidence=1.5)
