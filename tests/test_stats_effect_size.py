"""Tests for effect-size measures."""

from __future__ import annotations

import math

import pytest

from membench.stats.effect_size import (
    cliffs_delta,
    cohens_d,
    cohens_h,
    common_language_effect_size,
    odds_ratio,
)


class TestCliffsDelta:
    def test_complete_dominance(self) -> None:
        r = cliffs_delta([5, 6, 7], [1, 2, 3])
        assert r.delta == pytest.approx(1.0) and r.magnitude == "large"

    def test_no_difference(self) -> None:
        r = cliffs_delta([1, 2, 3], [1, 2, 3])
        assert r.delta == pytest.approx(0.0) and r.magnitude == "negligible"

    def test_reverse_dominance(self) -> None:
        assert cliffs_delta([1, 2], [5, 6]).delta == pytest.approx(-1.0)


class TestCLES:
    def test_relationship_to_cliffs_delta(self) -> None:
        # CLES = (delta + 1) / 2
        a, b = [3, 5, 7, 2], [1, 4, 6, 0]
        delta = cliffs_delta(a, b).delta
        assert common_language_effect_size(a, b) == pytest.approx((delta + 1) / 2)

    def test_full_superiority(self) -> None:
        assert common_language_effect_size([10, 11], [1, 2]) == 1.0


class TestParametric:
    def test_cohens_d_sign_and_scale(self) -> None:
        d = cohens_d([10, 11, 12, 13], [0, 1, 2, 3])
        assert d > 2.0  # large standardized gap

    def test_cohens_d_zero_when_equal(self) -> None:
        assert cohens_d([1, 2, 3], [1, 2, 3]) == 0.0

    def test_cohens_h(self) -> None:
        assert cohens_h(0.5, 0.5) == 0.0
        assert cohens_h(0.9, 0.5) == pytest.approx(
            2 * math.asin(math.sqrt(0.9)) - 2 * math.asin(math.sqrt(0.5))
        )

    def test_cohens_h_validation(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            cohens_h(1.2, 0.5)


class TestOddsRatio:
    def test_greater_than_one_when_a_better(self) -> None:
        assert odds_ratio(19, 20, 8, 20) > 1.0

    def test_haldane_correction_avoids_zero_div(self) -> None:
        assert odds_ratio(20, 20, 0, 20) > 1.0  # no division by zero despite extremes

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="successes"):
            odds_ratio(21, 20, 5, 20)
