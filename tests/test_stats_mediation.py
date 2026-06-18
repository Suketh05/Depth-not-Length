"""Tests for mediation analysis and mutual information."""

from __future__ import annotations

import numpy as np
import pytest

from membench.stats.mediation import mediation_analysis, mutual_information


class TestMutualInformation:
    def test_independent_is_zero(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.integers(0, 2, 2000)
        y = rng.integers(0, 2, 2000)
        assert mutual_information(x, y) == pytest.approx(0.0, abs=0.02)

    def test_identical_equals_entropy(self) -> None:
        x = [0, 1] * 50
        assert mutual_information(x, x) == pytest.approx(1.0)  # 1 bit for a fair binary

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="aligned"):
            mutual_information([1, 2], [1])


class TestMediation:
    def test_fully_mediated(self) -> None:
        # treatment drives the mediator; outcome is the mediator -> ~100% mediated
        rng = np.random.default_rng(1)
        t = np.array([0] * 100 + [1] * 100)
        m = np.where(t == 1, rng.binomial(1, 0.95, 200), rng.binomial(1, 0.1, 200)).astype(float)
        y = m.copy()  # outcome determined entirely by the mediator
        res = mediation_analysis(t, m, y)
        assert res.total_effect > 0.5
        assert res.proportion_mediated == pytest.approx(1.0, abs=0.05)
        assert abs(res.direct_effect) < 0.05

    def test_no_mediation_direct_effect(self) -> None:
        # outcome depends on treatment directly, mediator is noise
        rng = np.random.default_rng(2)
        t = np.array([0] * 100 + [1] * 100)
        m = rng.binomial(1, 0.5, 200).astype(float)
        y = t.astype(float)  # purely direct
        res = mediation_analysis(t, m, y)
        assert res.proportion_mediated == pytest.approx(0.0, abs=0.1)
        assert res.direct_effect == pytest.approx(1.0, abs=0.05)

    def test_requires_both_groups(self) -> None:
        with pytest.raises(ValueError, match="treated and control"):
            mediation_analysis([1, 1, 1], [0.0, 1.0, 1.0], [1, 1, 0])
