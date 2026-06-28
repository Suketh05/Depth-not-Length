"""Tests for inter-arm agreement metrics (Fleiss' kappa, Krippendorff's alpha)."""

from __future__ import annotations

import pytest

from membench.metrics.agreement import (
    FleissKappa,
    fleiss_kappa,
    krippendorff_alpha,
)

# ---------------------------------------------------------------------------
# Hand-computed golden, 4 subjects x 3 raters x 2 categories ("A", "B").
#
#   S1: A A A  -> counts [3, 0]
#   S2: A A B  -> counts [2, 1]
#   S3: B B B  -> counts [0, 3]
#   S4: A B B  -> counts [1, 2]
#
# Category totals: A = 3+2+0+1 = 6, B = 0+1+3+2 = 6, total = N*n = 12.
#   p_A = p_B = 0.5  ->  P_e = 0.5^2 + 0.5^2 = 0.5
# Per-subject agreement P_i = (sum n_ij^2 - n) / (n(n-1)) = (sum_sq - 3) / 6:
#   S1: (9-3)/6 = 1,  S2: (5-3)/6 = 1/3,  S3: (9-3)/6 = 1,  S4: (5-3)/6 = 1/3
#   P_bar = (1 + 1/3 + 1 + 1/3) / 4 = (8/3)/4 = 2/3
#   kappa = (P_bar - P_e)/(1 - P_e) = (2/3 - 1/2)/(1/2) = (1/6)/(1/2) = 1/3
#
# Krippendorff (nominal) on the same table, each unit weighted by 1/(m-1)=1/2:
#   o_AA = (6 + 2 + 0 + 0)/2 = 4,  o_BB = (0 + 0 + 6 + 2)/2 = 4
#   o_AB = o_BA = (0 + 2 + 0 + 2)/2 = 2
#   n_A = 4+2 = 6, n_B = 2+4 = 6, n = 12
#   D_o = o_AB + o_BA = 4
#   D_e = (n_A*n_B + n_B*n_A)/(n-1) = (36+36)/11 = 72/11
#   alpha = 1 - D_o/D_e = 1 - 44/72 = 7/18 ~= 0.388889
# ---------------------------------------------------------------------------
GOLDEN_TABLE = [
    ["A", "A", "A"],
    ["A", "A", "B"],
    ["B", "B", "B"],
    ["A", "B", "B"],
]


class TestFleissKappa:
    def test_golden_textbook_value(self) -> None:
        result = fleiss_kappa(GOLDEN_TABLE)
        assert isinstance(result, FleissKappa)
        assert result.p_bar == pytest.approx(2 / 3)
        assert result.p_e == pytest.approx(0.5)
        assert result.kappa == pytest.approx(1 / 3)
        assert result.n_subjects == 4
        assert result.n_raters == 3
        assert result.n_categories == 2

    def test_perfect_agreement_is_one(self) -> None:
        # Every rater agrees on every subject -> kappa = 1.0.
        table = [["A", "A", "A"], ["B", "B", "B"], ["A", "A", "A"]]
        assert fleiss_kappa(table).kappa == pytest.approx(1.0)

    def test_chance_agreement_is_zero(self) -> None:
        # Two raters, balanced categories, agreeing on exactly half the subjects.
        #   S1: A A (agree), S2: B B (agree), S3: A B, S4: B A (disagree)
        # totals A=4, B=4 -> P_e = 0.5; P_bar = 2/4 = 0.5 -> kappa = 0 exactly.
        table = [["A", "A"], ["B", "B"], ["A", "B"], ["B", "A"]]
        assert fleiss_kappa(table).kappa == pytest.approx(0.0)

    def test_complete_disagreement_is_negative(self) -> None:
        # Two raters always split -> P_bar = 0 < P_e -> kappa < 0.
        table = [["A", "B"], ["B", "A"], ["A", "B"]]
        assert fleiss_kappa(table).kappa < 0.0

    def test_single_category_is_perfect(self) -> None:
        # Degenerate 1 - P_e == 0 case: trivially perfect agreement.
        result = fleiss_kappa([["A", "A"], ["A", "A"]])
        assert result.kappa == pytest.approx(1.0)
        assert result.n_categories == 1

    def test_kappa_in_unit_interval_upper_bound(self) -> None:
        result = fleiss_kappa(GOLDEN_TABLE)
        assert -1.0 <= result.kappa <= 1.0
        assert 0.0 <= result.p_bar <= 1.0
        assert 0.0 <= result.p_e <= 1.0

    def test_requires_subjects(self) -> None:
        with pytest.raises(ValueError, match="at least one subject"):
            fleiss_kappa([])

    def test_requires_two_raters(self) -> None:
        with pytest.raises(ValueError, match="two raters"):
            fleiss_kappa([["A"], ["B"]])

    def test_requires_constant_raters(self) -> None:
        with pytest.raises(ValueError, match="same number of raters"):
            fleiss_kappa([["A", "B"], ["A", "B", "B"]])

    def test_rejects_missing(self) -> None:
        with pytest.raises(ValueError, match="missing ratings"):
            fleiss_kappa([["A", None], ["A", "B"]])


class TestKrippendorffAlpha:
    def test_golden_value(self) -> None:
        # Hand-computed above: alpha = 7/18.
        assert krippendorff_alpha(GOLDEN_TABLE) == pytest.approx(7 / 18)

    def test_identical_raters_is_one(self) -> None:
        # Required golden: perfect reliability on identical raters.
        table = [["A", "A", "A"], ["B", "B", "B"], ["C", "C", "C"]]
        assert krippendorff_alpha(table) == pytest.approx(1.0)

    def test_no_variation_is_one(self) -> None:
        # Single category everywhere -> D_e == 0 -> defined as 1.0.
        assert krippendorff_alpha([["A", "A"], ["A", "A"]]) == pytest.approx(1.0)

    def test_handles_missing_values(self) -> None:
        # None entries are skipped; the second unit still agrees perfectly.
        table = [["A", "A", None], ["B", "B", "B"]]
        assert krippendorff_alpha(table) == pytest.approx(1.0)

    def test_unit_with_one_value_is_ignored(self) -> None:
        # A unit with a single pairable value contributes nothing.
        table = [["A", None], ["B", "B"], ["C", "C"]]
        assert krippendorff_alpha(table) == pytest.approx(1.0)

    def test_alpha_upper_bounded_by_one(self) -> None:
        assert krippendorff_alpha(GOLDEN_TABLE) <= 1.0

    def test_requires_pairable_values(self) -> None:
        with pytest.raises(ValueError, match="pairable"):
            krippendorff_alpha([["A", None], ["B", None]])

    def test_numeric_categories(self) -> None:
        # Categories may be any hashable (ints here), matching float-array inputs.
        assert krippendorff_alpha([[1, 1], [0, 0]]) == pytest.approx(1.0)
