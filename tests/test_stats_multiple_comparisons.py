"""Tests for multiple-comparison control and the Demšar protocol."""

from __future__ import annotations

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from membench.stats.multiple_comparisons import (
    average_ranks,
    benjamini_hochberg_adjust,
    friedman_test,
    holm_adjust,
    nemenyi_critical_difference,
)

_PVALS = [0.001, 0.013, 0.02, 0.04, 0.3, 0.8]


class TestAdjustments:
    def test_holm_matches_statsmodels(self) -> None:
        mine = holm_adjust(_PVALS)
        ref = multipletests(_PVALS, method="holm")[1]
        np.testing.assert_allclose(mine, ref, atol=1e-9)

    def test_bh_matches_statsmodels(self) -> None:
        mine = benjamini_hochberg_adjust(_PVALS)
        ref = multipletests(_PVALS, method="fdr_bh")[1]
        np.testing.assert_allclose(mine, ref, atol=1e-9)

    def test_adjusted_ge_raw(self) -> None:
        assert all(adj >= raw - 1e-12 for adj, raw in zip(holm_adjust(_PVALS), _PVALS, strict=True))

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            holm_adjust([])


class TestFriedmanAndRanks:
    def test_friedman_detects_difference(self) -> None:
        # arm C consistently best, A worst across 6 blocks
        rng = np.random.default_rng(0)
        a = rng.normal(0.2, 0.02, 6)
        b = rng.normal(0.5, 0.02, 6)
        c = rng.normal(0.9, 0.02, 6)
        scores = np.column_stack([a, b, c])
        res = friedman_test(scores)
        assert res.p_value < 0.05

    def test_average_ranks_best_arm_lowest_rank(self) -> None:
        scores = np.array([[0.2, 0.5, 0.9], [0.1, 0.6, 0.95], [0.3, 0.4, 0.8]])
        ranks = average_ranks(scores, higher_is_better=True)
        assert ranks[2] == pytest.approx(1.0)  # column 2 always best -> rank 1
        assert ranks[0] == pytest.approx(3.0)  # column 0 always worst -> rank 3

    def test_friedman_requires_three_treatments(self) -> None:
        with pytest.raises(ValueError, match=">= 3"):
            friedman_test(np.array([[0.1, 0.2], [0.3, 0.4]]))


class TestNemenyi:
    def test_cd_positive_and_shrinks_with_blocks(self) -> None:
        cd_small = nemenyi_critical_difference(5, 10)
        cd_large = nemenyi_critical_difference(5, 100)
        assert cd_small > cd_large > 0

    def test_known_value(self) -> None:
        # k=5 (q=2.728), N=10: 2.728*sqrt(5*6/60) = 2.728*sqrt(0.5)
        assert nemenyi_critical_difference(5, 10) == pytest.approx(2.728 * np.sqrt(0.5), abs=1e-6)

    def test_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="tabulated range"):
            nemenyi_critical_difference(25, 10)
