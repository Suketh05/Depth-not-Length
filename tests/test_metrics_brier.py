"""Tests for the Murphy (1973) Brier-score decomposition.

The golden numbers below are hand-derived with exact fractions (see comments), NOT
read back from the implementation, so the assertions are non-circular.
"""

from __future__ import annotations

import math

import pytest

from membench.metrics.brier import BrierDecomposition, brier_decomposition

# --- Hand-computed golden ----------------------------------------------------
# Forecasts take two DISTINCT values, so each lands in its own equal-width bin and
# the forecast is constant within every bin -> binned Brier == raw Brier exactly.
#
#   probs   = [0.2, 0.2, 0.8, 0.8, 0.8]
#   outcomes= [  0,   1,   1,   1,   0]
#   N = 5,  base rate  o_bar = 3/5 = 0.6
#
# Bin A (p = 0.2): n=2, mean forecast f_A = 0.2, observed o_A = 1/2
# Bin B (p = 0.8): n=3, mean forecast f_B = 0.8, observed o_B = 2/3
#
# reliability = (1/5)[ 2*(0.2 - 1/2)^2 + 3*(0.8 - 2/3)^2 ] = (1/5)(7/30) = 7/150
# resolution  = (1/5)[ 2*(1/2 - 3/5)^2 + 3*(2/3 - 3/5)^2 ] = (1/5)(1/30) = 1/150
# uncertainty = (3/5)*(2/5) = 6/25
# brier = reliability - resolution + uncertainty = 7/150 - 1/150 + 6/25 = 7/25 = 0.28
# cross-check raw Brier: (0.04+0.64+0.04+0.04+0.64)/5 = 1.40/5 = 0.28  (matches)
_PROBS = [0.2, 0.2, 0.8, 0.8, 0.8]
_OUTCOMES = [False, True, True, True, False]
_GOLD_RELIABILITY = 7.0 / 150.0
_GOLD_RESOLUTION = 1.0 / 150.0
_GOLD_UNCERTAINTY = 6.0 / 25.0
_GOLD_BRIER = 7.0 / 25.0


def test_golden_components_and_sum() -> None:
    """Components match the hand-computed values and sum to the Brier score."""
    result = brier_decomposition(_PROBS, _OUTCOMES, n_bins=10)
    assert isinstance(result, BrierDecomposition)
    assert math.isclose(result.reliability, _GOLD_RELIABILITY, abs_tol=1e-12)
    assert math.isclose(result.resolution, _GOLD_RESOLUTION, abs_tol=1e-12)
    assert math.isclose(result.uncertainty, _GOLD_UNCERTAINTY, abs_tol=1e-12)
    assert math.isclose(result.brier, _GOLD_BRIER, abs_tol=1e-12)
    # The Murphy identity, asserted independently of how `brier` was computed.
    reconstructed = result.reliability - result.resolution + result.uncertainty
    assert math.isclose(result.brier, reconstructed, abs_tol=1e-12)


def test_metadata_fields() -> None:
    """The result reports the requested bin count and observation count."""
    result = brier_decomposition(_PROBS, _OUTCOMES, n_bins=10)
    assert result.n_bins == 10
    assert result.n_observations == 5
    assert math.isclose(result.base_rate, 0.6, abs_tol=1e-12)


def test_perfect_calibration_zero_reliability() -> None:
    """When every bin's mean forecast equals its observed frequency, reliability is 0."""
    # Bin 0.5: forecasts 0.5, outcomes [1, 0] -> observed 0.5 == forecast.
    # Bin 1.0: forecasts 1.0, outcomes [1, 1] -> observed 1.0 == forecast.
    probs = [0.5, 0.5, 1.0, 1.0]
    outcomes = [True, False, True, True]
    result = brier_decomposition(probs, outcomes, n_bins=10)
    assert math.isclose(result.reliability, 0.0, abs_tol=1e-12)


def test_components_non_negative() -> None:
    """Reliability, resolution, and uncertainty are all non-negative."""
    probs = [0.05, 0.3, 0.3, 0.55, 0.7, 0.7, 0.95]
    outcomes = [False, True, False, True, True, False, True]
    result = brier_decomposition(probs, outcomes, n_bins=10)
    assert result.reliability >= 0.0
    assert result.resolution >= 0.0
    assert result.uncertainty >= 0.0


def test_uncertainty_equals_base_rate_variance() -> None:
    """Uncertainty equals base-rate variance p*(1-p) for the observed outcomes."""
    probs = [0.1, 0.4, 0.6, 0.9, 0.3]
    outcomes = [True, False, True, True, False]
    base_rate = sum(outcomes) / len(outcomes)  # 3/5 = 0.6
    result = brier_decomposition(probs, outcomes, n_bins=10)
    assert math.isclose(result.uncertainty, base_rate * (1.0 - base_rate), abs_tol=1e-12)


def test_single_bin_zero_resolution() -> None:
    """With one bin every observation shares the base rate, so resolution is 0."""
    probs = [0.2, 0.8, 0.5]
    outcomes = [True, False, True]
    result = brier_decomposition(probs, outcomes, n_bins=1)
    assert math.isclose(result.resolution, 0.0, abs_tol=1e-12)
    # Brier collapses to reliability + uncertainty here.
    assert math.isclose(result.brier, result.reliability + result.uncertainty, abs_tol=1e-12)


@pytest.mark.parametrize(
    ("probs", "outcomes", "n_bins", "match"),
    [
        ([0.5], [True], 0, "n_bins must be >= 1"),
        ([], [], 10, "at least one observation"),
        ([0.5, 0.5], [True], 10, "must align"),
        ([1.5], [True], 10, "must be in"),
    ],
)
def test_validation_errors(
    probs: list[float], outcomes: list[bool], n_bins: int, match: str
) -> None:
    """Invalid inputs raise informative ValueErrors."""
    with pytest.raises(ValueError, match=match):
        brier_decomposition(probs, outcomes, n_bins=n_bins)
