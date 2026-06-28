r"""Aligned Rank Transform (ART) for a two-factor (arm x depth) design.

Parametric factorial ANOVA needs normal, homoscedastic residuals -- an assumption the
benchmark's bounded, often-binary outcomes routinely violate. The classic Kruskal-Wallis
/ Friedman rank tests are non-parametric but cannot test an *interaction*. The Aligned
Rank Transform fills that gap: it is a non-parametric procedure that supports main
effects **and** interactions in a fully factorial design by *aligning* the response for
one effect at a time, ranking the aligned response, then running an ordinary ANOVA on
those ranks and reading off the F statistic for that one effect.

Algorithm (Wobbrock, Findlater, Gergle & Higgins 2011, for two factors A and B)
------------------------------------------------------------------------------
Write each response as ``Y = grand + a_i + b_j + ab_ij + residual`` where ``a_i`` /
``b_j`` are the main-effect deviations, ``ab_ij`` the cell interaction, and
``residual = Y - cellmean_ij``. To test one effect, *strip out every other effect* --
equivalently add back only the effect of interest to the residual:

.. math::

    Y^{\mathrm{aligned}}_{A}  &= \text{residual} + (\bar{A}_i - \bar{Y})           \\
    Y^{\mathrm{aligned}}_{B}  &= \text{residual} + (\bar{B}_j - \bar{Y})           \\
    Y^{\mathrm{aligned}}_{AB} &= \text{residual}
        + (\overline{AB}_{ij} - \bar{A}_i - \bar{B}_j + \bar{Y})

By construction the aligned column for an effect contains *only* that effect plus error:
a full-factorial ANOVA on the (un-ranked) aligned column yields an F of zero for every
**other** effect -- the correctness check Wobbrock et al. recommend. The aligned column
is then converted to average ranks (ties shared) and a standard balanced two-way ANOVA is
run on the ranks; ``F`` for the aligned-for effect is the ART statistic for that effect.

This implementation requires a *balanced* design (equal observations per cell), the
setting in which the sums-of-squares decomposition is orthogonal and Type I/II/III
ANOVA coincide, so the reported F is unambiguous.

References
----------
Wobbrock, J. O., Findlater, L., Gergle, D., & Higgins, J. J. (2011). The Aligned Rank
Transform for nonparametric factorial analyses using only ANOVA procedures. *Proceedings
of the ACM Conference on Human Factors in Computing Systems (CHI '11)*, 143-146.
https://doi.org/10.1145/1978942.1979005
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy import stats

__all__ = [
    "ARTResult",
    "Effect",
    "EffectResult",
    "align_for_effect",
    "aligned_rank_transform",
    "aligned_ranks",
]


class Effect(str, Enum):
    """A term in the two-factor model: the two main effects and their interaction."""

    ARM = "arm"
    DEPTH = "depth"
    INTERACTION = "interaction"


@dataclass(frozen=True, slots=True)
class EffectResult:
    """ANOVA-on-ranks outcome for a single aligned effect."""

    name: str
    f_statistic: float
    df_effect: int
    df_error: int
    p_value: float


@dataclass(frozen=True, slots=True)
class ARTResult:
    """F statistics for both main effects and their interaction under the ART."""

    arm: EffectResult
    depth: EffectResult
    interaction: EffectResult


def _factorize(labels: Sequence[Hashable]) -> tuple[NDArray[np.intp], int]:
    """Map labels to integer codes by first appearance; return codes and level count."""
    levels = list(dict.fromkeys(labels))
    index = {label: i for i, label in enumerate(levels)}
    codes = np.array([index[label] for label in labels], dtype=np.intp)
    return codes, len(levels)


def _prepare(
    values: Sequence[float],
    arm: Sequence[Hashable],
    depth: Sequence[Hashable],
) -> tuple[NDArray[np.float64], NDArray[np.intp], NDArray[np.intp], int, int, int]:
    """Validate inputs and return ``(y, arm_codes, depth_codes, a, b, n_per_cell)``."""
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not (len(arm) == len(depth) == y.size):
        raise ValueError("values, arm and depth must have equal length")
    if y.size == 0:
        raise ValueError("need at least one observation")
    arm_codes, a = _factorize(arm)
    depth_codes, b = _factorize(depth)
    if a < 2 or b < 2:
        raise ValueError("each factor needs at least two levels")
    counts = np.zeros((a, b), dtype=np.intp)
    np.add.at(counts, (arm_codes, depth_codes), 1)
    n_per_cell = int(counts.flat[0])
    if not np.all(counts == n_per_cell):
        raise ValueError("design must be balanced (equal observations per cell)")
    if n_per_cell < 2:
        raise ValueError("need at least two replicates per cell for an error term")
    return y, arm_codes, depth_codes, a, b, n_per_cell


def _group_means(
    y: NDArray[np.float64],
    arm_codes: NDArray[np.intp],
    depth_codes: NDArray[np.intp],
    a: int,
    b: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return per-observation cell means, arm-marginal means and depth-marginal means."""
    cell_mean = np.empty_like(y)
    arm_mean = np.empty_like(y)
    depth_mean = np.empty_like(y)
    for i in range(a):
        arm_mask = arm_codes == i
        arm_mean[arm_mask] = y[arm_mask].mean()
        for j in range(b):
            cell_mask = arm_mask & (depth_codes == j)
            cell_mean[cell_mask] = y[cell_mask].mean()
    for j in range(b):
        depth_mask = depth_codes == j
        depth_mean[depth_mask] = y[depth_mask].mean()
    return cell_mean, arm_mean, depth_mean


def align_for_effect(
    values: Sequence[float],
    arm: Sequence[Hashable],
    depth: Sequence[Hashable],
    effect: Effect,
) -> NDArray[np.float64]:
    r"""Align the response for one ``effect`` by stripping out all other effects.

    The aligned value is ``residual + estimated(effect)`` where
    ``residual = Y - cellmean`` and the estimated effect is the main-effect deviation
    (``ARM``/``DEPTH``) or the cell interaction term (``INTERACTION``).

    Parameters
    ----------
    values : Sequence[float]
        Response, one per observation.
    arm, depth : Sequence[Hashable]
        Factor labels for each observation (any hashable; levels found by first
        appearance).
    effect : Effect
        Which effect to align for.

    Returns
    -------
    numpy.ndarray
        The aligned (not yet ranked) responses, same order as ``values``.
    """
    y, arm_codes, depth_codes, a, b, _ = _prepare(values, arm, depth)
    grand = float(y.mean())
    cell_mean, arm_mean, depth_mean = _group_means(y, arm_codes, depth_codes, a, b)
    residual = y - cell_mean
    if effect is Effect.ARM:
        estimated = arm_mean - grand
    elif effect is Effect.DEPTH:
        estimated = depth_mean - grand
    else:
        estimated = cell_mean - arm_mean - depth_mean + grand
    aligned: NDArray[np.float64] = residual + estimated
    return aligned


def aligned_ranks(
    values: Sequence[float],
    arm: Sequence[Hashable],
    depth: Sequence[Hashable],
    effect: Effect,
) -> NDArray[np.float64]:
    """Return the average ranks of the responses aligned for ``effect`` (ties shared)."""
    aligned = align_for_effect(values, arm, depth, effect)
    ranks: NDArray[np.float64] = np.asarray(stats.rankdata(aligned), dtype=np.float64)
    return ranks


def _two_way_anova_f(
    ranks: NDArray[np.float64],
    arm_codes: NDArray[np.intp],
    depth_codes: NDArray[np.intp],
    a: int,
    b: int,
    effect: Effect,
) -> tuple[float, int, int]:
    """Balanced two-way ANOVA on ``ranks``; return ``(F, df_effect, df_error)``.

    For a balanced design the sums of squares are orthogonal, so the effect's F is the
    same under Type I/II/III decomposition.
    """
    n_total = ranks.size
    grand = ranks.mean()
    cell_mean, arm_mean, depth_mean = _group_means(ranks, arm_codes, depth_codes, a, b)
    ss_arm = float(np.sum((arm_mean - grand) ** 2))
    ss_depth = float(np.sum((depth_mean - grand) ** 2))
    ss_cells = float(np.sum((cell_mean - grand) ** 2))
    ss_interaction = ss_cells - ss_arm - ss_depth
    ss_total = float(np.sum((ranks - grand) ** 2))
    ss_error = ss_total - ss_cells
    df_arm, df_depth = a - 1, b - 1
    df_interaction = df_arm * df_depth
    df_error = n_total - a * b
    table = {
        Effect.ARM: (ss_arm, df_arm),
        Effect.DEPTH: (ss_depth, df_depth),
        Effect.INTERACTION: (ss_interaction, df_interaction),
    }
    ss_effect, df_effect = table[effect]
    ms_error = ss_error / df_error
    if ms_error == 0:
        f_value = float("inf") if ss_effect > 0 else 0.0
    else:
        f_value = (ss_effect / df_effect) / ms_error
    return f_value, df_effect, df_error


def _effect_result(
    values: Sequence[float],
    arm: Sequence[Hashable],
    depth: Sequence[Hashable],
    arm_codes: NDArray[np.intp],
    depth_codes: NDArray[np.intp],
    a: int,
    b: int,
    effect: Effect,
) -> EffectResult:
    """Align for ``effect``, rank, run the ANOVA on ranks, and package the F test."""
    ranks = aligned_ranks(values, arm, depth, effect)
    f_value, df_effect, df_error = _two_way_anova_f(ranks, arm_codes, depth_codes, a, b, effect)
    p_value = float(stats.f.sf(f_value, df_effect, df_error))
    return EffectResult(
        name=effect.value,
        f_statistic=f_value,
        df_effect=df_effect,
        df_error=df_error,
        p_value=p_value,
    )


def aligned_rank_transform(
    values: Sequence[float],
    arm: Sequence[Hashable],
    depth: Sequence[Hashable],
) -> ARTResult:
    """Run the Aligned Rank Transform on a balanced two-factor (arm x depth) design.

    Each effect is aligned, ranked, and tested with a balanced two-way ANOVA on the
    ranks; the F statistic for the aligned-for effect is returned.

    Parameters
    ----------
    values : Sequence[float]
        Response, one per observation.
    arm, depth : Sequence[Hashable]
        Factor labels for each observation; the design must be balanced (equal
        observations per cell) with at least two replicates per cell.

    Returns
    -------
    ARTResult
        F statistics (with degrees of freedom and p-values) for the ``arm`` and
        ``depth`` main effects and their interaction.
    """
    _, arm_codes, depth_codes, a, b, _ = _prepare(values, arm, depth)
    return ARTResult(
        arm=_effect_result(values, arm, depth, arm_codes, depth_codes, a, b, Effect.ARM),
        depth=_effect_result(values, arm, depth, arm_codes, depth_codes, a, b, Effect.DEPTH),
        interaction=_effect_result(
            values, arm, depth, arm_codes, depth_codes, a, b, Effect.INTERACTION
        ),
    )
