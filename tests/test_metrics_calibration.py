"""Tests for calibration metrics."""

from __future__ import annotations

import pytest

from membench.metrics.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_curve,
)


class TestBrier:
    def test_perfect(self) -> None:
        assert brier_score([1.0, 0.0, 1.0], [True, False, True]) == 0.0

    def test_worst(self) -> None:
        assert brier_score([0.0, 1.0], [True, False]) == pytest.approx(1.0)

    def test_known_value(self) -> None:
        # (0.7-1)^2 + (0.2-0)^2 = 0.09 + 0.04 = 0.13 -> mean 0.065
        assert brier_score([0.7, 0.2], [True, False]) == pytest.approx(0.065)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="align"):
            brier_score([0.5], [True, False])
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            brier_score([1.5], [True])


class TestReliabilityAndECE:
    def test_perfectly_calibrated_has_zero_ece(self) -> None:
        # 10 preds at 1.0 all correct, 10 at 0.0 all wrong -> ECE 0
        probs = [1.0] * 10 + [0.0] * 10
        outcomes = [True] * 10 + [False] * 10
        assert expected_calibration_error(probs, outcomes) == pytest.approx(0.0)

    def test_overconfident_has_positive_ece(self) -> None:
        # asserts 0.9 but only half correct
        probs = [0.9] * 10
        outcomes = [True] * 5 + [False] * 5
        ece = expected_calibration_error(probs, outcomes)
        assert ece == pytest.approx(0.4)  # |0.9 - 0.5|

    def test_reliability_bins_report_counts(self) -> None:
        report = reliability_curve([0.1, 0.2, 0.9, 0.95], [False, False, True, True], n_bins=2)
        total = sum(b.count for b in report.bins)
        assert total == 4
        # the high-confidence bin should have accuracy 1.0
        high = max(report.bins, key=lambda b: b.mean_confidence)
        assert high.accuracy == 1.0
