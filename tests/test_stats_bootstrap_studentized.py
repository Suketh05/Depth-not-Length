"""Tests for studentized (bootstrap-t) and subsampling confidence intervals.

The headline golden is the hand-computed studentized pivot ``t*`` on one
resample (see ``test_pivot_matches_hand_computation``); coverage and structural
invariants are checked separately.
"""

from __future__ import annotations

import numpy as np
import pytest

from membench.stats.bootstrap_studentized import (
    ConfidenceInterval,
    studentized_ci,
    studentized_t_statistic,
    subsampling_ci,
)


class TestStudentizedPivot:
    def test_pivot_matches_hand_computation(self) -> None:
        # Original sample x = [1, 2, 3, 4, 5]: theta_hat = mean = 3.0.
        # Take ONE resample x* = [1, 1, 2, 3, 5] (drawn with replacement).
        #   mean(x*)          = 12 / 5            = 2.4
        #   sum sq dev (2.4)  = 1.96+1.96+0.16+0.36+6.76 = 11.2
        #   var (ddof=1)      = 11.2 / 4          = 2.8
        #   SE(x*)            = sqrt(2.8 / 5)     = sqrt(0.56)
        #   t* = (2.4 - 3.0) / sqrt(0.56)         = -0.6 / sqrt(0.56)
        #      = -0.80178372573727...
        golden = -0.6 / np.sqrt(0.56)  # = -0.8017837257372732, derived by hand above
        got = studentized_t_statistic([1.0, 1.0, 2.0, 3.0, 5.0], point_estimate=3.0)
        assert got == pytest.approx(golden, rel=1e-12)
        assert got == pytest.approx(-0.8017837257372732, abs=1e-12)

    def test_pivot_rejects_constant_resample(self) -> None:
        with pytest.raises(ValueError, match="standard error"):
            studentized_t_statistic([2.0, 2.0, 2.0], point_estimate=2.0)


class TestStudentizedCI:
    def test_brackets_estimate_and_is_deterministic(self) -> None:
        rng = np.random.default_rng(1)
        data = rng.normal(5.0, 1.0, size=200).tolist()
        a = studentized_ci(data, seed=0, n_resamples=1000)
        b = studentized_ci(data, seed=0, n_resamples=1000)
        assert isinstance(a, ConfidenceInterval)
        assert a.low < a.point < a.high
        assert a.confidence == 0.95
        assert (a.low, a.high) == (b.low, b.high)  # seeded -> reproducible

    def test_rejects_too_few(self) -> None:
        with pytest.raises(ValueError, match="two observations"):
            studentized_ci([1.0])

    def test_coverage_at_least_nominal(self) -> None:
        # Validity check: over many seeded normal samples (mu = 0), the 90%
        # bootstrap-t interval must bracket the TRUE mean about >= 90% of the
        # time. Threshold 0.85 = nominal 0.90 minus a Monte-Carlo allowance
        # (~3 binomial SEs over 200 trials, SE = sqrt(.9*.1/200) ~= 0.021); a
        # broken procedure would land near 0.6-0.7, not 0.85.
        n_trials = 200
        hits = 0
        for trial in range(n_trials):
            sample = np.random.default_rng(trial).normal(0.0, 1.0, size=40)
            ci = studentized_ci(sample.tolist(), confidence=0.90, n_resamples=400, seed=trial)
            if ci.low <= 0.0 <= ci.high:
                hits += 1
        coverage = hits / n_trials
        assert coverage >= 0.85


class TestSubsamplingCI:
    def test_brackets_true_mean(self) -> None:
        # A single representative (non-pathological) seeded normal sample; the
        # validity-over-repeated-seeds claim is the coverage test below.
        rng = np.random.default_rng(0)
        data = rng.normal(10.0, 2.0, size=400).tolist()
        ci = subsampling_ci(data, seed=0, n_subsamples=1500)
        assert ci.low < 10.0 < ci.high
        assert ci.low < ci.point < ci.high

    def test_deterministic(self) -> None:
        rng = np.random.default_rng(2)
        data = rng.normal(0.0, 1.0, size=300).tolist()
        a = subsampling_ci(data, seed=3, n_subsamples=500)
        b = subsampling_ci(data, seed=3, n_subsamples=500)
        assert (a.low, a.high) == (b.low, b.high)

    def test_rejects_bad_block_size(self) -> None:
        with pytest.raises(ValueError, match="1 < b < n"):
            subsampling_ci([1.0, 2.0, 3.0, 4.0], subsample_size=1)
        with pytest.raises(ValueError, match="1 < b < n"):
            subsampling_ci([1.0, 2.0, 3.0, 4.0], subsample_size=4)

    def test_rejects_too_few(self) -> None:
        with pytest.raises(ValueError, match="two observations"):
            subsampling_ci([1.0])

    def test_coverage_at_least_nominal(self) -> None:
        # Subsampling with b = floor(sqrt(n)) and the sqrt(n) rate gives a
        # rate-normalised root whose dispersion matches sigma; the inverted 90%
        # interval should bracket the true mean about >= 90% of the time.
        # Threshold 0.82 keeps a wider Monte-Carlo margin (subsampling is more
        # variable than the bootstrap-t at finite n).
        n_trials = 200
        hits = 0
        for trial in range(n_trials):
            sample = np.random.default_rng(1000 + trial).normal(0.0, 1.0, size=400)
            ci = subsampling_ci(sample.tolist(), confidence=0.90, n_subsamples=400, seed=trial)
            if ci.low <= 0.0 <= ci.high:
                hits += 1
        coverage = hits / n_trials
        assert coverage >= 0.82
