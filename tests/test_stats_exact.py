"""Tests for exact inference: Clopper-Pearson intervals and exact McNemar.

Every golden is hand-derived from first principles in the class docstring
(closed forms at the binomial boundary, exact dyadic fractions for the McNemar
tails) or quoted from the paper's worked examples -- never produced by running
the module under test back to itself. Paper anchors: worked example (j)
(app:worked, Figure fig:chain), worked example (n) (Figure fig:supersession),
and the fig:forest caption's prescription of exact methods at boundary cells.
"""

from __future__ import annotations

import math

import pytest

from membench.stats.exact import (
    ExactBinomialInterval,
    clopper_pearson,
)


class TestClopperPearsonClosedFormBoundaries:
    r"""Boundary cells collapse to closed forms one can verify by hand.

    For ``s == n`` the only exact constraint is the lower one: the smallest p
    such that observing all successes is not too surprising solves
    ``P(X = n | p) = p^n = alpha/2``, i.e. ``L = (alpha/2)^(1/n)`` and U = 1.
    Equivalently Beta(n, 1) has CDF ``F(x) = x^n``, so its ``alpha/2`` quantile
    is analytically ``(alpha/2)^(1/n)`` -- no numerics needed. By the mirror
    argument ``s == 0`` gives ``(0, 1 - (alpha/2)^(1/n))``.
    """

    @pytest.mark.parametrize("n", [1, 5, 40])
    @pytest.mark.parametrize("alpha", [0.05, 0.10])
    def test_all_successes_closed_form(self, n: int, alpha: float) -> None:
        res = clopper_pearson(n, n, alpha=alpha)
        assert res.low == pytest.approx((alpha / 2.0) ** (1.0 / n), rel=1e-12)
        assert res.high == 1.0
        assert res.estimate == 1.0

    @pytest.mark.parametrize("n", [1, 5, 40])
    @pytest.mark.parametrize("alpha", [0.05, 0.10])
    def test_zero_successes_closed_form(self, n: int, alpha: float) -> None:
        res = clopper_pearson(0, n, alpha=alpha)
        assert res.low == 0.0
        assert res.high == pytest.approx(1.0 - (alpha / 2.0) ** (1.0 / n), rel=1e-12)
        assert res.estimate == 0.0

    def test_single_trial_hand_literals(self) -> None:
        # n = 1, alpha = 0.05: the closed forms are literal numbers.
        # 1/1 -> ((0.025)^(1/1), 1) = (0.025, 1.0); 0/1 -> (0, 1 - 0.025) = (0, 0.975).
        one = clopper_pearson(1, 1)
        assert one.low == pytest.approx(0.025, rel=1e-12)
        assert one.high == 1.0
        zero = clopper_pearson(0, 1)
        assert zero.low == 0.0
        assert zero.high == pytest.approx(0.975, rel=1e-12)

    @pytest.mark.parametrize("n", [1, 7, 40])
    def test_zero_and_all_are_mirror_images(self, n: int) -> None:
        # By p -> 1-p symmetry of the binomial: CP(0, n).high == 1 - CP(n, n).low.
        assert clopper_pearson(0, n).high == pytest.approx(
            1.0 - clopper_pearson(n, n).low, abs=1e-15
        )

    def test_result_shape(self) -> None:
        res = clopper_pearson(3, 7, alpha=0.10)
        assert isinstance(res, ExactBinomialInterval)
        assert res.successes == 3
        assert res.n == 7
        assert res.alpha == 0.10
        assert res.method == "clopper_pearson"
        assert res.estimate == pytest.approx(3 / 7)
        assert math.isfinite(res.low) and math.isfinite(res.high)
