"""Tests for the Aligned Rank Transform (Wobbrock et al. 2011).

The golden values below are hand-computed on a tiny balanced 2x2 design with two
replicates per cell (eight observations), NOT read back from this implementation.

Dataset (arm, depth) -> responses:
    (0, 0): [1, 3]    cell mean 2.0
    (0, 1): [2, 4]    cell mean 3.0
    (1, 0): [5, 7]    cell mean 6.0
    (1, 1): [6, 10]   cell mean 8.0

Grand mean      = 38 / 8 = 4.75
Arm marginals   = {0: 2.5, 1: 7.0}
Depth marginals = {0: 4.0, 1: 5.5}

Align for the ARM main effect:  aligned = (Y - cellmean) + (armmean - grand)
    (0,0) Y=1 : 1-2 + 2.5-4.75 = -3.25      (0,0) Y=3 : 3-2 + 2.5-4.75 = -1.25
    (0,1) Y=2 : 2-3 + 2.5-4.75 = -3.25      (0,1) Y=4 : 4-3 + 2.5-4.75 = -1.25
    (1,0) Y=5 : 5-6 + 7.0-4.75 =  1.25      (1,0) Y=7 : 7-6 + 7.0-4.75 =  3.25
    (1,1) Y=6 : 6-8 + 7.0-4.75 =  0.25      (1,1) Y=10: 10-8+ 7.0-4.75 =  4.25
  -> aligned = [-3.25, -1.25, -3.25, -1.25, 1.25, 3.25, 0.25, 4.25]

Average ranks of that column (sorted -3.25,-3.25,-1.25,-1.25,0.25,1.25,3.25,4.25):
    -3.25 -> 1.5 (tie of ranks 1,2)
    -1.25 -> 3.5 (tie of ranks 3,4)
     0.25 -> 5,  1.25 -> 6,  3.25 -> 7,  4.25 -> 8
  -> ranks (input order) = [1.5, 3.5, 1.5, 3.5, 6, 7, 5, 8]   (sum = 36 = 8*9/2)

Two-way ANOVA on those ranks (n=2, a=b=2, N=8, grand=4.5):
    arm marginals   = {0: 2.5, 1: 6.5};  depth marginals = {0: 4.5, 1: 4.5}
    SS_arm   = 2*2 * [(2.5-4.5)^2 + (6.5-4.5)^2] = 4*8 = 32   (df 1)
    SS_depth = 0
    SS_cells = 2 * 4*(2.0^2) = 32  ;  SS_AB = 32-32-0 = 0
    SS_total = 41  ;  SS_error = 41-32 = 9   (df 4)  ->  MS_error = 2.25
    F_arm    = (32/1) / (9/4) = 32 / 2.25 = 14.2222...
"""

from __future__ import annotations

import numpy as np
import pytest

from membench.stats.aligned_rank_transform import (
    ARTResult,
    Effect,
    EffectResult,
    align_for_effect,
    aligned_rank_transform,
    aligned_ranks,
)

# Balanced 2x2 design, two replicates per cell.
VALUES = [1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 10.0]
ARM = [0, 0, 0, 0, 1, 1, 1, 1]
DEPTH = [0, 0, 1, 1, 0, 0, 1, 1]


class TestAlignment:
    def test_arm_alignment_matches_hand_computed(self) -> None:
        expected = [-3.25, -1.25, -3.25, -1.25, 1.25, 3.25, 0.25, 4.25]
        aligned = align_for_effect(VALUES, ARM, DEPTH, Effect.ARM)
        np.testing.assert_allclose(aligned, expected)

    def test_aligning_for_arm_zeroes_the_depth_effect(self) -> None:
        # Invariant: the column aligned for ARM carries no DEPTH (or interaction) signal.
        aligned = align_for_effect(VALUES, ARM, DEPTH, Effect.ARM)
        depth = np.asarray(DEPTH)
        depth0 = aligned[depth == 0].mean()
        depth1 = aligned[depth == 1].mean()
        assert depth0 == pytest.approx(0.0, abs=1e-12)
        assert depth1 == pytest.approx(0.0, abs=1e-12)

    def test_aligning_for_depth_zeroes_the_arm_effect(self) -> None:
        aligned = align_for_effect(VALUES, ARM, DEPTH, Effect.DEPTH)
        arm = np.asarray(ARM)
        assert aligned[arm == 0].mean() == pytest.approx(0.0, abs=1e-12)
        assert aligned[arm == 1].mean() == pytest.approx(0.0, abs=1e-12)

    def test_aligning_for_interaction_zeroes_both_main_effects(self) -> None:
        aligned = align_for_effect(VALUES, ARM, DEPTH, Effect.INTERACTION)
        arm, depth = np.asarray(ARM), np.asarray(DEPTH)
        for code in (0, 1):
            assert aligned[arm == code].mean() == pytest.approx(0.0, abs=1e-12)
            assert aligned[depth == code].mean() == pytest.approx(0.0, abs=1e-12)


class TestAlignedRanks:
    def test_arm_ranks_match_hand_computed(self) -> None:
        expected = [1.5, 3.5, 1.5, 3.5, 6.0, 7.0, 5.0, 8.0]
        ranks = aligned_ranks(VALUES, ARM, DEPTH, Effect.ARM)
        np.testing.assert_allclose(ranks, expected)

    def test_ranks_sum_to_constant(self) -> None:
        # n(n+1)/2 = 36 regardless of which effect is aligned (sum-to-constant invariant).
        for effect in Effect:
            ranks = aligned_ranks(VALUES, ARM, DEPTH, effect)
            assert ranks.sum() == pytest.approx(36.0)


class TestART:
    def test_arm_f_statistic_matches_hand_computed(self) -> None:
        result = aligned_rank_transform(VALUES, ARM, DEPTH)
        # F_arm = 32 / 2.25 = 14.2222... (derived in the module docstring above).
        assert result.arm.f_statistic == pytest.approx(32.0 / 2.25)
        assert result.arm.df_effect == 1
        assert result.arm.df_error == 4

    def test_strong_arm_effect_is_significant(self) -> None:
        result = aligned_rank_transform(VALUES, ARM, DEPTH)
        assert result.arm.p_value < 0.05

    def test_return_shape(self) -> None:
        result = aligned_rank_transform(VALUES, ARM, DEPTH)
        assert isinstance(result, ARTResult)
        for effect in (result.arm, result.depth, result.interaction):
            assert isinstance(effect, EffectResult)
            assert effect.df_effect == 1
            assert effect.df_error == 4
        assert result.arm.name == "arm"
        assert result.interaction.name == "interaction"

    def test_p_value_matches_scipy_survival_function(self) -> None:
        from scipy import stats

        result = aligned_rank_transform(VALUES, ARM, DEPTH)
        expected_p = float(stats.f.sf(32.0 / 2.25, 1, 4))
        assert result.arm.p_value == pytest.approx(expected_p)


class TestValidation:
    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            aligned_rank_transform([1.0, 2.0], [0], [0, 1])

    def test_unbalanced_design_rejected(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        arm = [0, 0, 0, 1, 1]
        depth = [0, 1, 1, 0, 1]  # cell (0,1) has two, others have one
        with pytest.raises(ValueError, match="balanced"):
            aligned_rank_transform(values, arm, depth)

    def test_single_replicate_rejected(self) -> None:
        # One observation per cell -> no error degrees of freedom.
        values = [1.0, 2.0, 3.0, 4.0]
        arm = [0, 0, 1, 1]
        depth = [0, 1, 0, 1]
        with pytest.raises(ValueError, match="replicate"):
            aligned_rank_transform(values, arm, depth)

    def test_single_level_factor_rejected(self) -> None:
        with pytest.raises(ValueError, match="two levels"):
            aligned_rank_transform([1.0, 2.0, 3.0, 4.0], [0, 0, 0, 0], [0, 1, 0, 1])
