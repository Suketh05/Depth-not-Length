"""Tests for the analytic crossover-depth solver."""

from __future__ import annotations

import pytest

from membench.theory.crossover import (
    find_crossover_depth,
    g,
    net_value_gain,
    overhead_ratio,
)
from membench.theory.decay import SimilarityDecayModel
from membench.theory.recovery import RecoveryModel


def _model(q: float, s0: float = 0.9, rho: float = 0.5) -> RecoveryModel:
    return RecoveryModel(decay=SimilarityDecayModel(s0=s0, rho=rho), q=q)


class TestOverheadRatio:
    def test_ratio(self) -> None:
        assert overhead_ratio(2.0, 8.0) == pytest.approx(0.25)

    def test_rejects_nonpositive_value(self) -> None:
        with pytest.raises(ValueError, match="task_value"):
            overhead_ratio(1.0, 0.0)

    def test_rejects_negative_cost(self) -> None:
        with pytest.raises(ValueError, match="maintenance_cost"):
            overhead_ratio(-1.0, 1.0)


class TestG:
    def test_g_is_struct_minus_sim(self) -> None:
        m = _model(q=0.95)
        assert g(3, m) == pytest.approx(m.p_struct(3) - m.p_sim(3))

    def test_net_value_gain(self) -> None:
        m = _model(q=0.95)
        assert net_value_gain(3, m, task_value=10.0, maintenance_cost=2.0) == pytest.approx(
            10.0 * g(3, m) - 2.0
        )


class TestCrossover:
    def test_q_one_is_monotone_with_well_defined_dstar(self) -> None:
        # q = 1 => g(d) = 1 - P_sim(d), strictly increasing => half-line win.
        m = _model(q=1.0)
        res = find_crossover_depth(m, tau=0.5, max_depth=32)
        assert res.exists
        assert res.d_star is not None
        # g just below d_star must be <= tau, at d_star must exceed tau.
        assert g(res.d_star, m) > 0.5
        assert g(res.d_star - 1, m) <= 0.5 or res.d_star == 1
        # win region runs to the end of the scan for the monotone case.
        assert res.win_region is not None and res.win_region[1] == 32

    def test_continuous_root_brackets_integer_dstar(self) -> None:
        # tau chosen strictly between g(1) and g(2) so the crossover falls between
        # integer depths and the continuous root is a genuine sub-integer value.
        m = _model(q=1.0)
        assert g(1, m) < 0.8 < g(2, m)
        res = find_crossover_depth(m, tau=0.8)
        assert res.d_star == 2
        assert res.d_star_continuous is not None
        assert 1.0 <= res.d_star_continuous <= 2.0
        assert g(res.d_star_continuous, m) == pytest.approx(0.8, abs=1e-6)

    def test_qsub1_win_region_is_an_interval(self) -> None:
        # q < 1: both curves vanish, so the advantage is a bounded interval.
        m = _model(q=0.8, s0=0.9, rho=0.5)
        res = find_crossover_depth(m, tau=0.2, max_depth=64)
        assert res.exists
        assert res.win_region is not None
        lo, hi = res.win_region
        assert 1 <= lo <= hi
        # Inside the region g exceeds tau; just past hi it has fallen back.
        assert g(lo, m) > 0.2
        if hi < 64:
            assert g(hi + 1, m) <= 0.2

    def test_no_crossover_when_overhead_exceeds_peak(self) -> None:
        m = _model(q=0.8)
        res = find_crossover_depth(m, tau=0.99, max_depth=64)
        assert not res.exists
        assert res.d_star is None and res.win_region is None
        # peak_depth is still reported as a diagnostic.
        assert 1 <= res.peak_depth <= 64

    def test_rejects_bad_inputs(self) -> None:
        m = _model(q=0.9)
        with pytest.raises(ValueError, match="tau"):
            find_crossover_depth(m, tau=-0.1)
        with pytest.raises(ValueError, match="max_depth"):
            find_crossover_depth(m, tau=0.1, max_depth=0)
