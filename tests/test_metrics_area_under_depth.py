"""Tests for area-under-depth-curve (AUDC) and depth half-life.

Golden values are hand-derived from the trapezoidal rule and linear interpolation in
the comments below -- never read back from the implementation.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from membench.analysis.tables import ScoredRow
from membench.metrics.area_under_depth import (
    area_under_depth_curve,
    depth_half_life,
    depth_value_curve,
)


def _row(depth: int, compliance: float) -> ScoredRow:
    return ScoredRow(
        dataset="synthetic",
        arm="brief_graph_3hop",
        model="claude",
        depth=depth,
        spec_variant="stripped",
        compliance_rate=compliance,
        chain_recovered=compliance >= 1.0,
        recall=compliance,
        precision=compliance,
        correct=compliance >= 1.0,
        total_tokens=100,
        dollars=0.01,
    )


class TestGolden:
    # Curve (1, 1.0), (2, 0.5), (3, 0.0).
    # Trapezoid area = 1*(1.0+0.5)/2 + 1*(0.5+0.0)/2 = 0.75 + 0.25 = 1.0.
    # Depth span = 3 - 1 = 2, so normalized AUDC = 1.0 / 2 = 0.5.
    # Half its d=1 value (1.0) is 0.5, reached exactly at depth 2 -> half-life = 2.0.
    GOLDEN: ClassVar[dict[int, float]] = {1: 1.0, 2: 0.5, 3: 0.0}

    def test_raw_area(self) -> None:
        assert area_under_depth_curve(self.GOLDEN, normalize=False) == pytest.approx(1.0)

    def test_normalized_audc(self) -> None:
        assert area_under_depth_curve(self.GOLDEN) == pytest.approx(0.5)

    def test_half_life(self) -> None:
        assert depth_half_life(self.GOLDEN) == pytest.approx(2.0)

    def test_from_scored_rows(self) -> None:
        # Two rows per depth, averaged: d1 -> 1.0, d2 -> (0.6+0.4)/2 = 0.5, d3 -> 0.0.
        # Identical to the golden curve, so the same goldens must hold.
        rows = [
            _row(1, 1.0),
            _row(1, 1.0),
            _row(2, 0.6),
            _row(2, 0.4),
            _row(3, 0.0),
            _row(3, 0.0),
        ]
        assert depth_value_curve(rows) == {1: 1.0, 2: 0.5, 3: 0.0}
        assert area_under_depth_curve(rows, normalize=False) == pytest.approx(1.0)
        assert area_under_depth_curve(rows) == pytest.approx(0.5)
        assert depth_half_life(rows) == pytest.approx(2.0)


class TestInvariants:
    def test_flat_curve_normalized_equals_constant(self) -> None:
        # Mean-value identity: a flat curve at c has normalized AUDC == c.
        assert area_under_depth_curve({1: 0.7, 2: 0.7, 3: 0.7}) == pytest.approx(0.7)

    def test_flat_curve_raw_area(self) -> None:
        # Raw area = c * span = 0.7 * (3 - 1) = 1.4.
        assert area_under_depth_curve({1: 0.7, 2: 0.7, 3: 0.7}, normalize=False) == pytest.approx(
            1.4
        )

    def test_flat_curve_never_halves(self) -> None:
        assert depth_half_life({1: 0.7, 2: 0.7, 3: 0.7}) is None

    def test_monotone_interpolated_half_life(self) -> None:
        # Curve (1,1.0),(2,0.8),(3,0.6),(4,0.2). Baseline 1.0, target 0.5.
        # Crossing in [3, 4]: frac = (0.6-0.5)/(0.6-0.2) = 0.25 -> 3 + 0.25 = 3.25.
        curve = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.2}
        assert depth_half_life(curve) == pytest.approx(3.25)

    def test_monotone_raw_and_normalized_area(self) -> None:
        # Raw = (1.0+0.8)/2 + (0.8+0.6)/2 + (0.6+0.2)/2 = 0.9+0.7+0.4 = 2.0.
        # Normalized = 2.0 / (4 - 1) = 0.6666...
        curve = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.2}
        assert area_under_depth_curve(curve, normalize=False) == pytest.approx(2.0)
        assert area_under_depth_curve(curve) == pytest.approx(2.0 / 3.0)

    def test_nonunit_depth_spacing(self) -> None:
        # Trapezoid respects width: (1,1.0),(3,0.0) -> area = (3-1)*(1.0+0.0)/2 = 1.0.
        # Normalized = 1.0 / 2 = 0.5. Half-life: target 0.5 at frac 0.5 -> 1 + 0.5*2 = 2.0.
        curve = {1: 1.0, 3: 0.0}
        assert area_under_depth_curve(curve, normalize=False) == pytest.approx(1.0)
        assert area_under_depth_curve(curve) == pytest.approx(0.5)
        assert depth_half_life(curve) == pytest.approx(2.0)

    def test_single_depth(self) -> None:
        assert area_under_depth_curve({2: 0.4}, normalize=False) == pytest.approx(0.0)
        assert area_under_depth_curve({2: 0.4}) == pytest.approx(0.4)
        assert depth_half_life({2: 0.4}) is None

    def test_nonpositive_baseline_half_life(self) -> None:
        assert depth_half_life({1: 0.0, 2: 0.0}) is None

    def test_exact_crossing_at_knot(self) -> None:
        # Value hits target exactly at depth 2: baseline 1.0, target 0.5 at d=2.
        assert depth_half_life({1: 1.0, 2: 0.5, 3: 0.4}) == pytest.approx(2.0)


class TestErrors:
    def test_empty_mapping_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one point"):
            area_under_depth_curve({})

    def test_empty_rows_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one scored row"):
            depth_value_curve([])
