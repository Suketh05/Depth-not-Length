"""Tests for paired significance tests."""

from __future__ import annotations

import numpy as np
import pytest

from membench.stats.hypothesis import (
    mcnemar_test,
    paired_permutation_test,
    wilcoxon_signed_rank,
)


class TestPairedPermutation:
    def test_clear_difference_is_significant(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(1.0, 0.1, 50)  # consistently higher
        b = rng.normal(0.0, 0.1, 50)
        assert paired_permutation_test(a, b, seed=0).p_value < 0.01

    def test_no_difference_not_significant(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(0.0, 1.0, 50)
        res = paired_permutation_test(x, x.copy(), seed=0)
        assert res.statistic == pytest.approx(0.0)
        assert res.p_value == pytest.approx(1.0)

    def test_deterministic(self) -> None:
        a, b = [1.0, 2.0, 3.0, 1.5], [0.5, 1.0, 2.5, 1.0]
        assert paired_permutation_test(a, b, seed=0).p_value == (
            paired_permutation_test(a, b, seed=0).p_value
        )

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="paired"):
            paired_permutation_test([1.0], [1.0, 2.0])


class TestWilcoxon:
    def test_identical_is_one(self) -> None:
        assert wilcoxon_signed_rank([1.0, 2.0], [1.0, 2.0]).p_value == 1.0

    def test_shift_is_significant(self) -> None:
        a = list(range(1, 21))
        b = [x - 3 for x in a]  # consistent positive difference
        assert wilcoxon_signed_rank(a, b).p_value < 0.01


class TestMcNemar:
    def test_one_sided_dominance(self) -> None:
        # a right & b wrong on many tasks, never the reverse
        a = [True] * 12 + [True] * 8
        b = [False] * 12 + [True] * 8
        res = mcnemar_test(a, b)
        assert res.statistic == 12.0 and res.p_value < 0.01

    def test_no_discordance_is_one(self) -> None:
        a = [True, False, True]
        assert mcnemar_test(a, a).p_value == 1.0

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="paired"):
            mcnemar_test([True], [True, False])
