"""Tests for g-computation / standardization (the parametric g-formula)."""

from __future__ import annotations

import numpy as np
import pytest

from membench.stats.g_computation import g_computation


class TestGoldenExactLinearOutcome:
    def test_ate_equals_true_coefficient(self) -> None:
        # Outcome is an EXACT linear function of A and L: Y = 2 + 3*A + 1.5*L.
        # The g-formula contrast at any unit i is
        #   E[Y|A=1, L_i] - E[Y|A=0, L_i] = (2 + 3 + 1.5*L_i) - (2 + 1.5*L_i) = 3,
        # so the standardized ATE = mean_i(3) = 3.0 regardless of the L distribution.
        # OLS recovers the coefficients exactly on noise-free linear data, hence the
        # golden value is the true A coefficient, 3.0 (derived by hand, not from code).
        a = [0, 0, 0, 1, 1, 1]
        L = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        y = [2.0 + 3.0 * ai + 1.5 * li for ai, li in zip(a, L, strict=True)]

        res = g_computation(a, y, L)

        assert res.ate == pytest.approx(3.0)
        assert res.mean_y1 - res.mean_y0 == pytest.approx(3.0)
        assert res.n == 6

    def test_interaction_model_matches_on_overlapping_support(self) -> None:
        # With full L overlap across arms the interaction term is identified and,
        # since the true model is additive (no effect modification), the estimate
        # is still exactly the true A coefficient 3.0.
        a = [0, 0, 0, 1, 1, 1]
        L = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        y = [2.0 + 3.0 * ai + 1.5 * li for ai, li in zip(a, L, strict=True)]

        res = g_computation(a, y, L, interaction=True)

        assert res.ate == pytest.approx(3.0)


class TestInvariants:
    def test_no_confounding_equals_crude_difference(self) -> None:
        # Invariant: with no confounding (L balanced across arms => A independent of
        # L) the g-formula ATE equals the crude difference of group means. L is the
        # same multiset in both arms, so cov(A, L) = 0 and the adjusted A coefficient
        # equals the unadjusted one even on noisy outcomes.
        rng = np.random.default_rng(0)
        L = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        a = [0] * 6 + [1] * 6
        y = rng.normal(size=12).tolist()

        res = g_computation(a, y, L)

        ya = np.asarray(y)
        aa = np.asarray(a)
        crude = float(ya[aa == 1].mean() - ya[aa == 0].mean())
        assert res.ate == pytest.approx(crude)

    def test_adjustment_corrects_confounding(self) -> None:
        # When L confounds (treated units have systematically higher L) and L raises
        # Y, the crude difference is biased upward but the g-formula recovers the true
        # additive effect. Y = 1 + 2*A + 5*L exactly => true ATE = 2.
        a = [0, 0, 0, 1, 1, 1]
        L = [1.0, 2.0, 3.0, 3.0, 4.0, 5.0]
        y = [1.0 + 2.0 * ai + 5.0 * li for ai, li in zip(a, L, strict=True)]

        res = g_computation(a, y, L)

        ya = np.asarray(y)
        aa = np.asarray(a)
        crude = float(ya[aa == 1].mean() - ya[aa == 0].mean())
        assert res.ate == pytest.approx(2.0)
        assert crude == pytest.approx(12.0)  # confounded, far from the truth
        assert res.ate != pytest.approx(crude)

    def test_counterfactual_means_bracket_ate(self) -> None:
        a = [0, 0, 1, 1]
        L = [1.0, 2.0, 1.0, 2.0]
        y = [0.0, 1.0, 1.0, 1.0]

        res = g_computation(a, y, L)

        assert res.ate == pytest.approx(res.mean_y1 - res.mean_y0)
        assert res.n == 4


class TestValidation:
    def test_rejects_non_binary_treatment(self) -> None:
        with pytest.raises(ValueError, match="binary"):
            g_computation([0, 1, 2], [0.0, 1.0, 1.0], [1.0, 2.0, 3.0])

    def test_requires_both_groups(self) -> None:
        with pytest.raises(ValueError, match="positivity"):
            g_computation([1, 1, 1], [0.0, 1.0, 1.0], [1.0, 2.0, 3.0])

    def test_rejects_misaligned(self) -> None:
        with pytest.raises(ValueError, match="aligned"):
            g_computation([0, 1], [0.0, 1.0], [1.0])

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            g_computation([], [], [])

    def test_supports_multiple_covariates(self) -> None:
        # Two confounders; additive truth Y = 1 + 2*A + 3*L1 - L2 => ATE = 2.
        a = [0, 0, 0, 1, 1, 1]
        l1 = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        l2 = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        cov = [[a1, b1] for a1, b1 in zip(l1, l2, strict=True)]
        y = [1.0 + 2.0 * ai + 3.0 * c[0] - c[1] for ai, c in zip(a, cov, strict=True)]

        res = g_computation(a, y, cov)

        assert res.ate == pytest.approx(2.0)
