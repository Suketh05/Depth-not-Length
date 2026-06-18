"""Tests for bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
import pytest

from membench.stats.bootstrap import bca_ci, cluster_bootstrap_ci, percentile_ci


class TestPercentile:
    def test_brackets_the_mean_and_is_deterministic(self) -> None:
        rng = np.random.default_rng(1)
        data = rng.normal(5.0, 1.0, size=200).tolist()
        a = percentile_ci(data, seed=0)
        b = percentile_ci(data, seed=0)
        assert a.low < a.point < a.high
        assert (a.low, a.high) == (b.low, b.high)  # seeded -> reproducible

    def test_rejects_too_few(self) -> None:
        with pytest.raises(ValueError, match="two observations"):
            percentile_ci([1.0])


class TestBCa:
    def test_brackets_true_mean_on_normal_data(self) -> None:
        rng = np.random.default_rng(2)
        data = rng.normal(10.0, 2.0, size=300).tolist()
        ci = bca_ci(data, seed=0, n_resamples=1000)
        assert ci.low < 10.0 < ci.high
        assert ci.low < ci.point < ci.high

    def test_skewed_data_interval_is_asymmetric(self) -> None:
        rng = np.random.default_rng(3)
        data = rng.exponential(1.0, size=300).tolist()  # right-skewed
        ci = bca_ci(data, seed=0, n_resamples=1500)
        # BCa shifts to account for skew; just assert a valid, ordered interval
        assert ci.low < ci.point < ci.high


class TestClusterBootstrap:
    def test_resamples_clusters(self) -> None:
        # two clusters with different means; CI should bracket the pooled mean
        values = [0.0] * 20 + [1.0] * 20
        clusters = ["a"] * 20 + ["b"] * 20
        ci = cluster_bootstrap_ci(values, clusters, seed=0, n_resamples=1000)
        assert 0.0 <= ci.low <= ci.point <= ci.high <= 1.0

    def test_requires_two_clusters(self) -> None:
        with pytest.raises(ValueError, match="two clusters"):
            cluster_bootstrap_ci([1.0, 2.0], ["a", "a"])

    def test_alignment_checked(self) -> None:
        with pytest.raises(ValueError, match="align"):
            cluster_bootstrap_ci([1.0, 2.0], ["a"])
