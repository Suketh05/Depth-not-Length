"""Tests for the empirical crossover-depth estimator."""

from __future__ import annotations

import numpy as np
import pytest

from membench.stats.changepoint import bootstrap_crossover_ci, interpolated_crossing


class TestInterpolatedCrossing:
    def test_simple_upward_crossing(self) -> None:
        # advantage goes -0.2, -0.1, 0.1 over depths 1,2,3 -> crosses between 2 and 3
        x = interpolated_crossing([1, 2, 3], [-0.2, -0.1, 0.1])
        assert x == pytest.approx(2.5)

    def test_already_above(self) -> None:
        assert interpolated_crossing([1, 2], [0.3, 0.4]) == 1.0

    def test_never_crosses(self) -> None:
        assert interpolated_crossing([1, 2, 3], [-0.3, -0.2, -0.1]) is None

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="aligned"):
            interpolated_crossing([1, 2], [0.1])


class TestBootstrapCrossover:
    def test_locates_crossover_with_ci(self) -> None:
        rng = np.random.default_rng(0)
        depths = [1, 2, 3]
        # structured ~0.95 flat; similarity decays 0.9 -> 0.5 -> 0.2 -> crosses ~d2
        structured = [rng.binomial(1, 0.95, 40).astype(float) for _ in depths]
        similarity = [rng.binomial(1, p, 40).astype(float) for p in (0.9, 0.5, 0.2)]
        est = bootstrap_crossover_ci(depths, structured, similarity, seed=0, n_resamples=500)
        assert est.d_star is not None
        assert 1.0 <= est.d_star <= 3.0
        assert est.ci_low is not None and est.ci_low <= est.d_star <= est.ci_high
        assert est.exists_fraction > 0.9

    def test_no_crossing_when_structured_always_worse(self) -> None:
        depths = [1, 2, 3]
        structured = [[0.1] * 20 for _ in depths]
        similarity = [[0.9] * 20 for _ in depths]
        est = bootstrap_crossover_ci(depths, structured, similarity, seed=0, n_resamples=200)
        assert est.d_star is None and est.exists_fraction == 0.0

    def test_requires_two_depths(self) -> None:
        with pytest.raises(ValueError, match="two depths"):
            bootstrap_crossover_ci([1], [[1.0]], [[0.0]])
