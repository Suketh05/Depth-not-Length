r"""Empirical crossover depth: where structured memory overtakes similarity.

The theory predicts a depth :math:`d^\star` at which structured memory's recovery
overtakes similarity's. This module *measures* it from the depth-stratified results
and puts a confidence interval on it, so the headline can read "theory predicts
:math:`d^\star \approx 2`, and we observe the crossover at 2.1 (95% CI 1.8-2.4)".

The crossover is the depth at which the structured-minus-similarity *advantage*
curve crosses zero (upward), located by linear interpolation between the bracketing
depths. Its uncertainty comes from a cluster bootstrap over tasks: resample tasks at
each depth, recompute the advantage curve, relocate the crossing, and take the
percentile interval over those replicates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["CrossoverEstimate", "bootstrap_crossover_ci", "interpolated_crossing"]


def interpolated_crossing(
    x: Sequence[float], y: Sequence[float], *, level: float = 0.0
) -> float | None:
    """Return the first ``x`` where ``y`` crosses ``level`` upward (linear interp).

    Returns the first x if ``y`` already exceeds ``level`` there, or ``None`` if it
    never does.
    """
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if xs.shape != ys.shape or xs.size == 0:
        raise ValueError("x and y must be non-empty and aligned")
    if ys[0] > level:
        return float(xs[0])
    for i in range(1, xs.size):
        if ys[i] > level >= ys[i - 1]:
            span = ys[i] - ys[i - 1]
            frac = 0.0 if span == 0 else (level - ys[i - 1]) / span
            return float(xs[i - 1] + frac * (xs[i] - xs[i - 1]))
    return None


@dataclass(frozen=True, slots=True)
class CrossoverEstimate:
    """The measured crossover depth with a bootstrap confidence interval."""

    d_star: float | None
    ci_low: float | None
    ci_high: float | None
    exists_fraction: float  # fraction of bootstrap replicates with a crossing


def bootstrap_crossover_ci(
    depths: Sequence[int],
    structured_by_depth: Sequence[Sequence[float]],
    similarity_by_depth: Sequence[Sequence[float]],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> CrossoverEstimate:
    """Estimate the crossover depth and its CI from per-task, per-depth recoveries.

    ``structured_by_depth[i]`` / ``similarity_by_depth[i]`` are the per-task recovery
    values (0/1 or rates) for the arm at ``depths[i]``. The point estimate uses the
    mean advantage curve; the CI resamples tasks within each depth.
    """
    if not (len(depths) == len(structured_by_depth) == len(similarity_by_depth)):
        raise ValueError("depths and per-depth arrays must align")
    if len(depths) < 2:
        raise ValueError("need at least two depths to locate a crossover")

    struct = [np.asarray(s, dtype=np.float64) for s in structured_by_depth]
    sim = [np.asarray(s, dtype=np.float64) for s in similarity_by_depth]
    xs = [float(d) for d in depths]

    point_adv = [float(s.mean() - m.mean()) for s, m in zip(struct, sim, strict=True)]
    point = interpolated_crossing(xs, point_adv)

    rng = np.random.default_rng(seed)
    crossings: list[float] = []
    for _ in range(n_resamples):
        adv = []
        for s, m in zip(struct, sim, strict=True):
            bs = s[rng.integers(0, s.size, s.size)] if s.size else s
            bm = m[rng.integers(0, m.size, m.size)] if m.size else m
            adv.append(float(bs.mean() - bm.mean()))
        crossing = interpolated_crossing(xs, adv)
        if crossing is not None:
            crossings.append(crossing)

    exists_fraction = len(crossings) / n_resamples
    if not crossings:
        return CrossoverEstimate(point, None, None, exists_fraction)
    alpha = 1.0 - confidence
    low, high = np.quantile(crossings, [alpha / 2, 1 - alpha / 2])
    return CrossoverEstimate(point, float(low), float(high), exists_fraction)
