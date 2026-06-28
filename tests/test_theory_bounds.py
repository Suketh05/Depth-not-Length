"""Tests for the closed-form crossover-depth bounds.

The golden case is hand-derived two independent ways and is *not* read back from
``bracket_crossover_depth`` itself (that would be circular):

    s0 = 0.9, rho = 0.95, q = 0.6, tau = 0

1. Closed form (depth ceiling). The zero-overhead crossover is the nonzero root
   of the log-advantage parabola,

       d0 = 2 * ln(q / s0) / ln(rho) - 1
          = 2 * ln(0.6 / 0.9) / ln(0.95) - 1
          = 2 * (-0.4054651) / (-0.0512933) - 1
          = 15.80963 - 1                      [note: 2*ln(q/s0)/ln(rho) = 15.80963]
          = 14.80963

   so the smallest winning integer depth is floor(14.80963) + 1 = 15.

2. Direct sign change of g(d) = q^d - s0^d * rho^(d(d+1)/2) at consecutive ints:

       g(14) = 0.6^14 - 0.9^14 * 0.95^105
             = 7.8322e-4 - (0.228768 * 0.0045673)
             = 7.8322e-4 - 1.04490e-3 = -2.617e-4   (< 0  -> similarity wins)
       g(15) = 0.6^15 - 0.9^15 * 0.95^120
             = 4.6993e-4 - (0.205891 * 0.0021261)
             = 4.6993e-4 - 4.3775e-4 =  3.218e-5   (> 0  -> structure wins)

   The sign flips between d = 14 and d = 15, so d* = 15, matching (1).

Hence d_lower == d_upper == 15 at tau = 0, and (with max_depth = 64) the win
window is (15, 64).
"""

from __future__ import annotations

import math

import pytest

from membench.theory.bounds import (
    CrossoverBounds,
    bracket_crossover_depth,
    tau_zero_crossover,
)
from membench.theory.crossover import find_crossover_depth, g
from membench.theory.decay import SimilarityDecayModel
from membench.theory.recovery import RecoveryModel

# Golden model (see module docstring for the hand derivation).
GOLDEN_S0 = 0.9
GOLDEN_RHO = 0.95
GOLDEN_Q = 0.6
GOLDEN_DSTAR = 15


def _model(q: float, s0: float, rho: float) -> RecoveryModel:
    return RecoveryModel(decay=SimilarityDecayModel(s0=s0, rho=rho), q=q)


class TestGolden:
    def test_tau_zero_crossover_matches_hand_value(self) -> None:
        m = _model(GOLDEN_Q, GOLDEN_S0, GOLDEN_RHO)
        # Hand value 14.80963 from the closed form, independent of the solver.
        assert tau_zero_crossover(m) == pytest.approx(14.80963, abs=1e-4)

    def test_hand_sign_change_brackets_dstar(self) -> None:
        # Independent confirmation of the golden via the (separate) g() function:
        # similarity wins at 14, structure wins at 15.
        m = _model(GOLDEN_Q, GOLDEN_S0, GOLDEN_RHO)
        assert g(GOLDEN_DSTAR - 1, m) < 0.0
        assert g(GOLDEN_DSTAR, m) > 0.0

    def test_bounds_pin_dstar_exactly_at_tau_zero(self) -> None:
        m = _model(GOLDEN_Q, GOLDEN_S0, GOLDEN_RHO)
        b = bracket_crossover_depth(m, tau=0.0, max_depth=64)
        assert b.d_lower == GOLDEN_DSTAR
        assert b.d_upper == GOLDEN_DSTAR
        assert b.win_window == (GOLDEN_DSTAR, 64)

    def test_bounds_agree_with_numeric_solver_on_golden(self) -> None:
        m = _model(GOLDEN_Q, GOLDEN_S0, GOLDEN_RHO)
        res = find_crossover_depth(m, tau=0.0, max_depth=64)
        # The hand golden, the solver, and the closed form all give 15.
        assert res.d_star == GOLDEN_DSTAR
        b = bracket_crossover_depth(m, tau=0.0, max_depth=64)
        assert b.d_lower <= res.d_star <= b.d_upper


class TestValidation:
    def test_rejects_negative_tau(self) -> None:
        with pytest.raises(ValueError, match="tau"):
            bracket_crossover_depth(_model(0.6, 0.9, 0.95), tau=-0.1)

    def test_rejects_bad_max_depth(self) -> None:
        with pytest.raises(ValueError, match="max_depth"):
            bracket_crossover_depth(_model(0.6, 0.9, 0.95), tau=0.0, max_depth=0)

    def test_rejects_non_decaying_model(self) -> None:
        with pytest.raises(ValueError, match="rho"):
            bracket_crossover_depth(_model(0.6, 0.9, 1.0), tau=0.0)


class TestStructure:
    def test_no_win_when_tau_exceeds_q(self) -> None:
        # q^d <= q < tau for all d >= 1 -> structure never clears the overhead.
        m = _model(0.6, 0.9, 0.95)
        b = bracket_crossover_depth(m, tau=0.7, max_depth=64)
        assert b.win_window is None
        assert not find_crossover_depth(m, tau=0.7, max_depth=64).exists

    def test_q_one_is_monotone_half_line(self) -> None:
        # q = 1 => advantage is monotone => win window runs to the scan edge.
        m = _model(1.0, 0.9, 0.95)
        b = bracket_crossover_depth(m, tau=0.3, max_depth=40)
        assert b.win_window is not None
        assert b.win_window[1] == 40
        assert b.d_upper == 40
        res = find_crossover_depth(m, tau=0.3, max_depth=40)
        assert res.exists and res.d_star is not None
        assert b.d_lower <= res.d_star <= b.d_upper

    def test_dataclass_is_frozen(self) -> None:
        b = bracket_crossover_depth(_model(0.6, 0.9, 0.95), tau=0.0)
        assert isinstance(b, CrossoverBounds)
        with pytest.raises((AttributeError, TypeError)):
            b.d_lower = 1  # type: ignore[misc]


class TestInvariantGrid:
    """Closed-form bounds must stay consistent with the numeric solver."""

    @pytest.mark.parametrize("q", [0.6, 0.7, 0.85])
    @pytest.mark.parametrize("rho", [0.9, 0.95])
    @pytest.mark.parametrize("s0", [0.8, 0.95])
    @pytest.mark.parametrize("tau", [0.0, 1e-4, 1e-3, 0.01, 0.05])
    def test_bracket_consistent_with_solver(
        self, q: float, rho: float, s0: float, tau: float
    ) -> None:
        m = _model(q, s0, rho)
        b = bracket_crossover_depth(m, tau, max_depth=64)
        res = find_crossover_depth(m, tau, max_depth=64)

        assert b.d_lower >= 1
        if b.win_window is not None:
            win_lo, win_hi = b.win_window
            assert win_lo == b.d_lower
            assert win_lo <= win_hi
        else:
            # An empty closed-form window means a win is provably impossible.
            assert not res.exists

        if res.exists:
            assert res.d_star is not None
            # The bracket must contain the solver's crossover depth.
            assert b.d_lower <= res.d_star <= b.d_upper
            # The numeric win region must sit inside the closed-form window.
            assert b.win_window is not None
            assert res.win_region is not None
            assert b.win_window[0] <= res.win_region[0]
            assert res.win_region[1] <= b.win_window[1]

    def test_lower_bound_is_first_possible_winner(self) -> None:
        # Every integer depth strictly below d_lower must lose at tau = 0
        # (g(d) <= 0), confirming the lower bound is not over-tight.
        m = _model(0.6, 0.9, 0.95)
        b = bracket_crossover_depth(m, tau=0.0, max_depth=64)
        for d in range(1, b.d_lower):
            assert g(d, m) <= 0.0
        assert g(b.d_lower, m) > 0.0
        assert math.floor(tau_zero_crossover(m)) + 1 == b.d_lower
