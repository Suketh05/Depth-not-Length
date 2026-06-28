"""Tests for compliance-vs-depth curve fitting and crossover extraction.

Goldens are derived independently of the module under test:

* The logistic recovery test generates data from an EXACT four-parameter logistic
  written out explicitly here (not via the module), then asserts the fit inverts
  the generator back to those known parameters with R^2 == 1.0.
* The crossover test uses two analytic straight lines whose intersection is solved
  by hand, so the expected depth is a closed-form constant, not a fitted value.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from membench.analysis.effect_curves import (
    CrossoverPoint,
    ExponentialDecayFit,
    LogisticFit,
    find_curve_crossover,
    fit_exponential_decay,
    fit_logistic,
)

# --- Known generative parameters for the logistic golden ----------------------
# f(d) = LOWER + (UPPER - LOWER) / (1 + exp(GROWTH * (d - MIDPOINT)))
LOWER = 0.1
UPPER = 0.9
MIDPOINT = 4.0
GROWTH = 1.0


def _exact_logistic(depths: np.ndarray) -> np.ndarray:
    return LOWER + (UPPER - LOWER) / (1.0 + np.exp(GROWTH * (depths - MIDPOINT)))


class TestFitLogistic:
    def test_recovers_known_parameters_with_perfect_r2(self) -> None:
        depths = np.arange(1.0, 8.0)  # 7 points, depths 1..7
        compliance = _exact_logistic(depths)
        # Sanity anchor (hand-computed): at d == MIDPOINT the curve is the midpoint
        # of the asymptotes, (0.1 + 0.9) / 2 = 0.5.
        assert math.isclose(float(_exact_logistic(np.array([4.0]))[0]), 0.5)

        fit = fit_logistic(depths, compliance)

        assert isinstance(fit, LogisticFit)
        assert fit.n_points == 7
        # GOLDEN: the fit must recover the exact generative parameters.
        assert fit.lower == pytest.approx(LOWER, abs=1e-3)
        assert fit.upper == pytest.approx(UPPER, abs=1e-3)
        assert fit.midpoint == pytest.approx(MIDPOINT, abs=1e-3)
        assert fit.growth_rate == pytest.approx(GROWTH, abs=1e-3)
        # GOLDEN: exact data => R^2 == 1.0.
        assert fit.r_squared == pytest.approx(1.0, abs=1e-9)

    def test_predict_matches_exact_curve(self) -> None:
        depths = np.arange(1.0, 8.0)
        fit = fit_logistic(depths, _exact_logistic(depths))
        # predict(MIDPOINT) == 0.5 (hand-computed midpoint of asymptotes).
        assert fit.predict(MIDPOINT) == pytest.approx(0.5, abs=1e-3)
        # predict(1) matches the exact generator at d = 1.
        assert fit.predict(1.0) == pytest.approx(
            float(_exact_logistic(np.array([1.0]))[0]), abs=1e-3
        )

    def test_rejects_too_few_points(self) -> None:
        with pytest.raises(ValueError, match="at least 4"):
            fit_logistic([1.0, 2.0, 3.0], [0.9, 0.5, 0.1])

    def test_rejects_misaligned_inputs(self) -> None:
        with pytest.raises(ValueError, match="align"):
            fit_logistic([1.0, 2.0, 3.0, 4.0], [0.9, 0.5, 0.1])

    def test_rejects_constant_depth(self) -> None:
        with pytest.raises(ValueError, match="must vary"):
            fit_logistic([2.0, 2.0, 2.0, 2.0], [0.9, 0.5, 0.3, 0.1])


class TestFitExponentialDecay:
    def test_recovers_known_parameters(self) -> None:
        # Exact exponential: f(d) = 0.8 * exp(-0.5 d) + 0.1, generated explicitly.
        amplitude, decay_rate, offset = 0.8, 0.5, 0.1
        depths = np.arange(0.0, 7.0)  # 7 points
        compliance = amplitude * np.exp(-decay_rate * depths) + offset
        # Hand anchor: at d = 0, f(0) = 0.8 + 0.1 = 0.9.
        assert math.isclose(float(compliance[0]), 0.9)

        fit = fit_exponential_decay(depths, compliance)

        assert isinstance(fit, ExponentialDecayFit)
        assert fit.amplitude == pytest.approx(amplitude, abs=1e-3)
        assert fit.decay_rate == pytest.approx(decay_rate, abs=1e-3)
        assert fit.offset == pytest.approx(offset, abs=1e-3)
        assert fit.r_squared == pytest.approx(1.0, abs=1e-9)
        # predict(0) == 0.9 (hand-computed).
        assert fit.predict(0.0) == pytest.approx(0.9, abs=1e-3)

    def test_rejects_too_few_points(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            fit_exponential_decay([1.0, 2.0], [0.5, 0.1])


class TestFindCurveCrossover:
    def test_analytic_line_crossing(self) -> None:
        # Two straight lines:
        #   a(d) = 0.9 - 0.1 d   (a falling arm)
        #   b(d) = 0.1 + 0.1 d   (a rising arm)
        # Crossing: 0.9 - 0.1 d = 0.1 + 0.1 d  =>  0.8 = 0.2 d  =>  d = 4.0,
        # common value a(4) = 0.9 - 0.4 = 0.5. (Hand-computed golden.)
        def curve_a(depth: float) -> float:
            return 0.9 - 0.1 * depth

        def curve_b(depth: float) -> float:
            return 0.1 + 0.1 * depth

        result = find_curve_crossover(curve_a, curve_b, depth_min=1.0, depth_max=7.0)

        assert isinstance(result, CrossoverPoint)
        assert result.exists
        assert result.depth == pytest.approx(4.0, abs=1e-9)  # GOLDEN d* = 4
        assert result.compliance == pytest.approx(0.5, abs=1e-9)  # GOLDEN value = 0.5

    def test_no_crossover_when_one_arm_dominates(self) -> None:
        def high(depth: float) -> float:
            return 0.9 - 0.01 * depth

        def low(depth: float) -> float:
            return 0.2 - 0.01 * depth

        result = find_curve_crossover(high, low, depth_min=1.0, depth_max=7.0)
        assert not result.exists
        assert math.isnan(result.depth)
        assert math.isnan(result.compliance)

    def test_crossover_between_two_fitted_logistics(self) -> None:
        # End-to-end: fit two arms, then find where the fitted curves cross.
        depths = np.arange(1.0, 8.0)
        falling = _exact_logistic(depths)  # high then low (midpoint 4)
        rising = _exact_logistic(8.0 - depths)  # mirror: low then high (midpoint 4)
        fit_a = fit_logistic(depths, falling)
        fit_b = fit_logistic(depths, rising)
        result = find_curve_crossover(fit_a.predict, fit_b.predict, depth_min=1.0, depth_max=7.0)
        assert result.exists
        # By the symmetry of the two mirrored logistics the crossing sits at the
        # shared midpoint depth 4.0.
        assert result.depth == pytest.approx(4.0, abs=1e-2)

    def test_rejects_inverted_interval(self) -> None:
        with pytest.raises(ValueError, match="depth_min < depth_max"):
            find_curve_crossover(lambda d: d, lambda d: 1.0 - d, depth_min=5.0, depth_max=1.0)
