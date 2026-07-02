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
    McNemarExactResult,
    clopper_pearson,
    mcnemar_exact,
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


class TestPaperWorkedExampleJ:
    """Paper worked example (j): chain recovery at synthetic d=3 (fig:chain).

    Quoting app:worked (j): "The typed store recovers 40/40 chains at synthetic
    d=3; the Clopper--Pearson exact 95% interval for 40/40 is
    [(0.025)^{1/40}, 1] = [0.912, 1.000] (lower limit = 0.025^{1/40} from
    inverting the binomial tail), whereas bm25 at 0.70 (28/40) gives
    [0.534, 0.834], disjoint."

    The 40/40 lower endpoint is the closed form 0.025**(1/40) evaluated in
    float64; 0.912 is its 3-dp rounding. For 28/40 the exact limits are
    (0.53468..., 0.83437...): the paper's printed upper endpoint 0.834 is the
    3-dp rounding, while its printed lower endpoint 0.534 is the 3-dp
    TRUNCATION of 0.53468 (round-half-up would give 0.535) -- we assert both
    published digits via the rounding rule that reproduces them, and pin
    correctness itself to the exact tail-inversion identity in the class below.
    """

    def test_typed_store_40_of_40(self) -> None:
        res = clopper_pearson(40, 40, alpha=0.05)
        assert res.low == pytest.approx(0.025 ** (1.0 / 40.0), rel=1e-12)
        assert round(res.low, 3) == 0.912  # the paper's printed lower limit
        assert res.high == 1.0  # the paper's printed upper limit
        assert res.low <= 1.0 and res.high <= 1.0  # never leaves [0, 1]

    def test_bm25_28_of_40(self) -> None:
        res = clopper_pearson(28, 40, alpha=0.05)
        assert res.estimate == pytest.approx(0.70)  # "bm25 at 0.70"
        assert round(res.high, 3) == 0.834  # paper's printed upper endpoint
        # Paper prints 0.534: the truncation of the exact 0.53468... (see class
        # docstring); assert the printed digits via floor-to-3-dp.
        assert math.floor(res.low * 1000.0) / 1000.0 == 0.534

    def test_intervals_disjoint(self) -> None:
        # The worked example's punchline: the two exact intervals do not overlap.
        typed = clopper_pearson(40, 40, alpha=0.05)
        bm25 = clopper_pearson(28, 40, alpha=0.05)
        assert typed.low > bm25.high


class TestBetaQuantileBinomialTailIdentity:
    r"""The defining property of Clopper--Pearson, checked independently.

    By construction the exact limits invert the binomial tail tests:

        P(X >= s | n, L) = alpha/2   and   P(X <= s | n, U) = alpha/2.

    The module computes L and U through Beta quantiles; here we plug them back
    into the *binomial* distribution (a different SciPy distribution object and
    code path), so agreement verifies the Beta-quantile identity
    ``P(X >= s | p) = I_p(s, n - s + 1)`` numerically rather than assuming it.
    This is the textbook derivation of the interval (Clopper & Pearson 1934;
    Brown, Cai & DasGupta 2001, eq. 6).
    """

    @pytest.mark.parametrize(
        ("successes", "n"),
        [(1, 5), (3, 10), (7, 20), (28, 40), (39, 40)],
    )
    @pytest.mark.parametrize("alpha", [0.05, 0.10])
    def test_limits_invert_the_binomial_tails(self, successes: int, n: int, alpha: float) -> None:
        from scipy import stats

        res = clopper_pearson(successes, n, alpha=alpha)
        upper_tail_at_low = float(stats.binom.sf(successes - 1, n, res.low))
        lower_tail_at_high = float(stats.binom.cdf(successes, n, res.high))
        assert upper_tail_at_low == pytest.approx(alpha / 2.0, abs=1e-10)
        assert lower_tail_at_high == pytest.approx(alpha / 2.0, abs=1e-10)


class TestClopperPearsonProperties:
    """Structural guarantees the paper leans on, plus input validation."""

    @pytest.mark.parametrize(
        ("successes", "n"),
        [(0, 5), (1, 5), (4, 8), (6, 8), (28, 40), (40, 40)],
    )
    def test_interval_contains_point_estimate(self, successes: int, n: int) -> None:
        res = clopper_pearson(successes, n)
        assert res.low <= res.estimate <= res.high
        assert 0.0 <= res.low <= res.high <= 1.0  # never leaves the unit interval

    def test_contains_the_wilson_interval_on_the_pooled_kappa_case(self) -> None:
        # 6/8 is the repo's pooled-kappa Wilson CI case (metrics/factorization):
        # Wilson 95% = (0.40927543031016883, 0.9285207872478909). Clopper-Pearson
        # is conservative by construction, so the exact interval must enclose the
        # score interval here.
        res = clopper_pearson(6, 8, alpha=0.05)
        assert res.low < 0.40927543031016883
        assert res.high > 0.9285207872478909

    def test_narrower_alpha_gives_wider_interval(self) -> None:
        # Coverage 99% > 95% > 80% must nest.
        wide = clopper_pearson(28, 40, alpha=0.01)
        mid = clopper_pearson(28, 40, alpha=0.05)
        narrow = clopper_pearson(28, 40, alpha=0.20)
        assert wide.low < mid.low < narrow.low
        assert narrow.high < mid.high < wide.high

    def test_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="trials"):
            clopper_pearson(0, 0)
        with pytest.raises(ValueError, match="successes"):
            clopper_pearson(9, 8)
        with pytest.raises(ValueError, match="successes"):
            clopper_pearson(-1, 8)
        with pytest.raises(ValueError, match="alpha"):
            clopper_pearson(4, 8, alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            clopper_pearson(4, 8, alpha=1.0)


class TestMcNemarExactGolden:
    """Hand-computed exact McNemar golden on b=2, c=8 discordant pairs.

    Under the null each of the 10 discordant pairs favours either arm with
    probability 1/2, so min(b, c) = 2 is a Binomial(10, 1/2) tail event:

        p = 2 * sum_{k<=2} C(10, k) / 2^10
          = 2 * (C(10,0) + C(10,1) + C(10,2)) / 1024
          = 2 * (1 + 10 + 45) / 1024
          = 112 / 1024
          = 0.109375  exactly.

    The denominator is a power of two, so 112/1024 = 7/64 is a dyadic rational
    representable exactly in float64 -- asserted with ==, no tolerance.
    """

    def test_exact_two_sided_p(self) -> None:
        res = mcnemar_exact(2, 8)
        assert res.p_value == 112 / 1024
        assert res.p_value == 0.109375  # exact float equality, not approx

    def test_symmetric_in_b_and_c(self) -> None:
        # Swapping the arms flips the sign of the statistic, not the p-value.
        forward = mcnemar_exact(2, 8)
        backward = mcnemar_exact(8, 2)
        assert forward.p_value == backward.p_value == 0.109375
        assert forward.mid_p == backward.mid_p
        assert forward.statistic == -6.0
        assert backward.statistic == 6.0

    def test_counts_and_method_recorded(self) -> None:
        res = mcnemar_exact(2, 8)
        assert isinstance(res, McNemarExactResult)
        assert (res.b, res.c) == (2, 8)
        assert res.method == "mcnemar_exact"


class TestMcNemarMidP:
    """Mid-p variant golden, hand-computed on the same b=2, c=8 table.

    The mid-p correction counts only half the probability of the observed
    count (Lancaster 1961):

        mid_p = 2 * (P(X < 2) + 0.5 * P(X = 2))
              = 2 * (C(10,0) + C(10,1) + 0.5 * C(10,2)) / 1024
              = 2 * (1 + 10 + 22.5) / 1024
              = 67 / 1024
              = 0.0654296875  exactly (dyadic, so == with no tolerance).

    Equivalently mid_p = p_exact - P(X = 2) = 112/1024 - 45/1024 = 67/1024.
    """

    def test_mid_p_golden(self) -> None:
        res = mcnemar_exact(2, 8)
        assert res.mid_p == 67 / 1024
        assert res.mid_p == 0.0654296875  # exact float equality

    def test_mid_p_is_exact_minus_observed_pmf(self) -> None:
        # Second derivation of the same number: subtract pmf(2) = 45/1024.
        res = mcnemar_exact(2, 8)
        assert res.p_value - res.mid_p == 45 / 1024

    @pytest.mark.parametrize(("b", "c"), [(0, 5), (1, 9), (2, 8), (3, 7), (5, 5), (10, 30)])
    def test_mid_p_never_exceeds_exact_p(self, b: int, c: int) -> None:
        res = mcnemar_exact(b, c)
        assert 0.0 < res.mid_p <= res.p_value <= 1.0
