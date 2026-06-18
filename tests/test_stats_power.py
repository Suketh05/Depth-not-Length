"""Tests for power and minimum-detectable-effect."""

from __future__ import annotations

import pytest

from membench.stats.power import (
    minimum_detectable_effect,
    required_sample_size,
    two_proportion_power,
)


class TestPower:
    def test_power_increases_with_n(self) -> None:
        small = two_proportion_power(0.9, 0.6, 20)
        large = two_proportion_power(0.9, 0.6, 200)
        assert large > small
        assert 0.0 <= small <= large <= 1.0

    def test_large_gap_high_power(self) -> None:
        assert two_proportion_power(0.95, 0.4, 50) > 0.95

    def test_equal_proportions_power_is_alpha(self) -> None:
        assert two_proportion_power(0.5, 0.5, 100, alpha=0.05) == pytest.approx(0.05)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            two_proportion_power(1.2, 0.5, 10)


class TestRequiredSampleSize:
    def test_round_trips_to_target_power(self) -> None:
        n = required_sample_size(0.9, 0.6, power=0.8)
        # plugging the computed n back in should reach ~the target power
        assert two_proportion_power(0.9, 0.6, n) >= 0.79
        assert n > 0

    def test_smaller_effect_needs_more_samples(self) -> None:
        assert required_sample_size(0.9, 0.85) > required_sample_size(0.9, 0.6)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="must differ"):
            required_sample_size(0.5, 0.5)


class TestMDE:
    def test_mde_shrinks_with_n(self) -> None:
        assert minimum_detectable_effect(0.5, 25) > minimum_detectable_effect(0.5, 400)

    def test_positive(self) -> None:
        assert minimum_detectable_effect(0.5, 100) > 0
