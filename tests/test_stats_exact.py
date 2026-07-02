r"""Golden tests for exact small-sample inference (``membench.stats.exact``).

Every asserted number is either (a) hand-derived from a closed form and quoted in the
comment above the assertion, (b) transcribed from the paper's worked examples with the
label cited, or (c) checked against the defining Beta-quantile / binomial-tail identity
directly -- never a value produced by running the module under test.

Paper anchors: App. ``app:worked`` worked examples (j) chain-recovery ``40/40 ->
[0.912, 1.000]`` and (n) supersession; Fig. ``fig:forest`` (F024) exact intervals /
McNemar at the ``p_hat = 1`` boundary cells.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from membench.stats.exact import (
    ClopperPearsonInterval,
    McNemarExactResult,
    clopper_pearson,
    mcnemar_exact,
)


class TestClopperPearsonClosedForms:
    def test_zero_successes_lower_is_zero_upper_is_closed_form(self) -> None:
        # x = 0/n has an exact upper limit 1 - (alpha/2)^(1/n) (Beta(1, n) CDF is
        # 1 - (1-x)^n, so ppf(1 - alpha/2) solves (1-x)^n = alpha/2). n = 20, alpha 0.05:
        # upper = 1 - 0.025^(1/20).
        ci = clopper_pearson(0, 20, 0.05)
        assert isinstance(ci, ClopperPearsonInterval)
        assert ci.low == 0.0
        assert ci.high == pytest.approx(1.0 - 0.025 ** (1.0 / 20), abs=1e-12)
        assert ci.point == 0.0

    def test_all_successes_upper_is_one_lower_is_closed_form(self) -> None:
        # x = n/n has an exact lower limit (alpha/2)^(1/n) (Beta(n, 1) CDF is x^n, so
        # ppf(alpha/2) solves x^n = alpha/2) and upper limit exactly 1.
        ci = clopper_pearson(20, 20, 0.05)
        assert ci.high == 1.0
        assert ci.low == pytest.approx(0.025 ** (1.0 / 20), abs=1e-12)
        assert ci.point == 1.0

    def test_closed_form_tracks_alpha(self) -> None:
        # Same closed form at a 99% interval (alpha = 0.01): lower = 0.005^(1/10).
        ci = clopper_pearson(10, 10, 0.01)
        assert ci.low == pytest.approx(0.005 ** (1.0 / 10), abs=1e-12)
        assert ci.high == 1.0


class TestClopperPearsonPaperWorkedExample:
    def test_chain_recovery_40_of_40_boundary_cell(self) -> None:
        # Paper App. app:worked worked example (j): a typed store recovers the chain at
        # every one of n = 40 paired queries (40/40), reported as the exact interval
        # [0.912, 1.000]. This is the n/n saturation cell: upper = 1 exactly, and
        # lower = (alpha/2)^(1/n) = 0.025^(1/40), which rounds to 0.912 at 3 dp.
        ci = clopper_pearson(40, 40, 0.05)
        assert ci.high == 1.0
        # Paper-published 3-dp value (worked example (j)).
        assert ci.low == pytest.approx(0.912, abs=1e-3)
        # Exact closed form the paper value is a rounding of.
        assert ci.low == pytest.approx(0.025 ** (1.0 / 40), abs=1e-12)
        assert round(ci.low, 3) == 0.912


class TestClopperPearsonBetaIdentity:
    def test_interior_cell_matches_beta_quantiles(self) -> None:
        # Textbook interior cell 8/10 at 95%. The Clopper-Pearson limits ARE the Beta
        # quantiles by definition: low = B^{-1}(alpha/2; x, n-x+1),
        # high = B^{-1}(1-alpha/2; x+1, n-x). Assert against the defining identity, not
        # a magic literal. (This 8/10 cell is the same one the Wilson golden in C.6
        # uses, so the two interval families can be compared side by side.)
        ci = clopper_pearson(8, 10, 0.05)
        assert ci.low == pytest.approx(float(stats.beta.ppf(0.025, 8, 3)), abs=1e-12)
        assert ci.high == pytest.approx(float(stats.beta.ppf(0.975, 9, 2)), abs=1e-12)
        # The exact interval is strictly wider than (contains) the observed proportion.
        assert ci.low < ci.point < ci.high

    def test_lower_limit_is_upper_limit_of_the_complement(self) -> None:
        # Symmetry of the equal-tailed construction: the lower limit for x successes is
        # one minus the upper limit for n-x successes (relabel success<->failure).
        ci = clopper_pearson(3, 10, 0.05)
        comp = clopper_pearson(7, 10, 0.05)
        assert ci.low == pytest.approx(1.0 - comp.high, abs=1e-12)
        assert ci.high == pytest.approx(1.0 - comp.low, abs=1e-12)

    def test_higher_confidence_widens_the_interval(self) -> None:
        # Nesting: a 99% interval (alpha 0.01) contains the 95% interval (alpha 0.05).
        wide = clopper_pearson(8, 10, 0.01)
        narrow = clopper_pearson(8, 10, 0.05)
        assert wide.low <= narrow.low
        assert wide.high >= narrow.high


class TestClopperPearsonValidation:
    def test_single_trial(self) -> None:
        # Smallest valid n. 1/1: lower = 0.025^(1/1) = 0.025 exactly, upper = 1.
        ci = clopper_pearson(1, 1, 0.05)
        assert ci.low == pytest.approx(0.025, abs=1e-12)
        assert ci.high == 1.0
        # 0/1: lower = 0, upper = 1 - 0.025 = 0.975 exactly.
        ci0 = clopper_pearson(0, 1, 0.05)
        assert ci0.low == 0.0
        assert ci0.high == pytest.approx(0.975, abs=1e-12)

    def test_rejects_out_of_range_successes(self) -> None:
        with pytest.raises(ValueError, match="successes"):
            clopper_pearson(11, 10)
        with pytest.raises(ValueError, match="successes"):
            clopper_pearson(-1, 10)

    def test_rejects_non_positive_n(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            clopper_pearson(0, 0)

    def test_rejects_out_of_range_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            clopper_pearson(5, 10, 0.0)
        with pytest.raises(ValueError, match="alpha"):
            clopper_pearson(5, 10, 1.0)


class TestMcNemarExactGolden:
    def test_b2_c8_two_sided_p_is_112_over_1024(self) -> None:
        # Golden-pack C / gap G13: b = 2, c = 8 discordant pairs (n = 10). The exact
        # two-sided p-value is twice the lower Binomial(10, 1/2) tail through k = 2:
        #   p = 2 * (C(10,0) + C(10,1) + C(10,2)) / 2^10
        #     = 2 * (1 + 10 + 45) / 1024
        #     = 112 / 1024
        #     = 0.109375  (exact dyadic rational).
        res = mcnemar_exact(2, 8)
        assert isinstance(res, McNemarExactResult)
        assert res.p_value == 0.109375
        assert res.p_value == 112 / 1024
        assert res.statistic == -6.0  # b - c = 2 - 8
        assert res.n_discordant == 10
        assert res.b == 2 and res.c == 8
        assert res.method == "mcnemar_exact"

    def test_symmetric_in_b_and_c(self) -> None:
        # The two-sided p-value depends only on {b, c} as a set; swapping flips the sign
        # of the statistic but leaves the p-value identical (still 112/1024).
        res = mcnemar_exact(8, 2)
        assert res.p_value == 0.109375
        assert res.statistic == 6.0


class TestMcNemarMidP:
    def test_b2_c8_mid_p_is_67_over_1024(self) -> None:
        # The Lancaster mid-p excludes half the point mass at the observed count k = 2
        # from each tail. Hand-derived from the same n = 10 tail:
        #   p_mid = (2 * (C(10,0)+C(10,1)+C(10,2)) - C(10,2)) / 2^10
        #         = (2 * 56 - 45) / 1024
        #         = 67 / 1024
        #         = 0.0654296875  (exact dyadic rational).
        res = mcnemar_exact(2, 8)
        assert res.mid_p_value == 0.0654296875
        assert res.mid_p_value == 67 / 1024

    def test_mid_p_is_strictly_less_than_exact_p(self) -> None:
        # Mid-p is uniformly less conservative than the exact two-sided p-value whenever
        # there is a non-empty discordant tail.
        res = mcnemar_exact(2, 8)
        assert res.mid_p_value < res.p_value
        # The gap is exactly the excluded point mass C(n, min(b,c)) / 2^n = 45/1024.
        assert res.p_value - res.mid_p_value == pytest.approx(45 / 1024, abs=1e-15)


class TestMcNemarEdges:
    def test_no_discordant_pairs_returns_unity(self) -> None:
        # b = c = 0: the test is undefined; the convention returns p = 1.0 (no evidence).
        res = mcnemar_exact(0, 0)
        assert res.p_value == 1.0
        assert res.mid_p_value == 1.0
        assert res.statistic == 0.0
        assert res.n_discordant == 0

    def test_even_small_split_clips_to_one(self) -> None:
        # b = c = 1 (n = 2): 2 * (C(2,0)+C(2,1)) / 4 = 2 * 3 / 4 = 1.5, clipped to 1.0.
        # Mid-p: (2*3 - C(2,1)) / 4 = (6 - 2) / 4 = 1.0.
        res = mcnemar_exact(1, 1)
        assert res.p_value == 1.0
        assert res.mid_p_value == 1.0

    def test_fully_discordant_is_strongly_significant(self) -> None:
        # b = 10, c = 0 (all discordances favour one arm), n = 10, min = 0:
        #   p     = 2 * C(10,0) / 2^10 = 2/1024 = 1/512 = 0.001953125.
        #   p_mid = (2 * C(10,0) - C(10,0)) / 2^10 = 1/1024 = 0.0009765625.
        res = mcnemar_exact(10, 0)
        assert res.p_value == 1 / 512
        assert res.p_value == 0.001953125
        assert res.mid_p_value == 1 / 1024
        assert res.statistic == 10.0

    def test_rejects_negative_counts(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            mcnemar_exact(-1, 3)
        with pytest.raises(ValueError, match="non-negative"):
            mcnemar_exact(3, -2)


def _discordant(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    """Return the McNemar discordant counts (a-only-correct, b-only-correct)."""
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    return a_only, b_only


class TestSupersessionScenario:
    """End-to-end use on the paper's paired supersession / forest cells."""

    def test_forest_boundary_cell_brief_vs_dense_d3(self) -> None:
        # Fig. fig:forest (F024) / tab:slope: Brief vs dense at d=3, 40/40 vs 27/40. Set
        # up the paired outcomes deterministically -- Brief right everywhere, dense right
        # on the first 27 tasks -- so the 2x2 table is exact.
        brief = np.ones(40, dtype=bool)
        dense = np.array([True] * 27 + [False] * 13, dtype=bool)
        b, c = _discordant(brief, dense)
        assert (b, c) == (13, 0)  # every discordance favours Brief; none favour dense

        res = mcnemar_exact(b, c)
        assert res.statistic == 13.0
        # n=13, min=0: p = 2 * C(13,0) / 2^13 = 2/8192 = 1/4096 = 0.000244140625.
        assert res.p_value == 1 / 4096
        # The paper cell is z=3.94, p<1e-4 by the asymptotic z-test; the exact test is
        # more conservative but still decisively rejects at the 1e-3 level.
        assert res.p_value < 1e-3

        # The two arms' exact intervals separate: Brief's lower limit (0.912, the 40/40
        # worked example (j)) sits above dense's point estimate 27/40 = 0.675.
        brief_ci = clopper_pearson(int(brief.sum()), 40)
        dense_ci = clopper_pearson(int(dense.sum()), 40)
        assert brief_ci.low == pytest.approx(0.912, abs=1e-3)
        assert dense_ci.low < 0.675 < dense_ci.high
        assert brief_ci.low > dense_ci.point

    def test_supersession_typed_vs_similarity_seeded(self) -> None:
        # tab:super: a typed store ranks the current decision first ~92% of the time vs a
        # 64-69% similarity band over n=40 paired queries (chance 50%). Draw paired
        # per-query outcomes from a fixed seed and run the exact machinery on them.
        rng = np.random.default_rng(2)
        n = 40
        typed = rng.random(n) < 0.923  # Brief current-ranked-first rate (tab:super)
        similarity = rng.random(n) < 0.66  # mid of the 64-69% band

        b, c = _discordant(typed, similarity)
        res = mcnemar_exact(b, c)
        # Net discordances favour the typed store, and the exact test rejects at 0.05.
        assert res.statistic > 0
        assert res.p_value < 0.05
        # Mid-p never exceeds the exact p-value.
        assert res.mid_p_value <= res.p_value

        # The typed store's exact interval clears the 50% chance floor; the confidence
        # interval brackets its own point estimate on both sides.
        typed_ci = clopper_pearson(int(typed.sum()), n)
        assert typed_ci.low > 0.5
        assert typed_ci.low <= typed_ci.point <= typed_ci.high

    def test_chance_floor_both_arms_at_random_no_rejection(self) -> None:
        # Control: two arms both drawn at the 50% chance floor produce a balanced
        # discordant split, and the exact McNemar does NOT reject (large p-value).
        rng = np.random.default_rng(7)
        n = 40
        arm_a = rng.random(n) < 0.5
        arm_b = rng.random(n) < 0.5
        res = mcnemar_exact(*_discordant(arm_a, arm_b))
        assert res.p_value > 0.05
