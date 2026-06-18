r"""Effect sizes: how *large* the difference between arms is, not just whether it exists.

Significance says a difference is real; effect size says how big -- which is what a
"wins by a wide margin" claim actually rests on. This module provides the
non-parametric and proportion effect sizes appropriate for the benchmark's bounded,
often-binary outcomes:

* :func:`cliffs_delta` -- ordinal dominance in ``[-1, 1]`` with the standard
  magnitude labels; robust to the non-normal, capped distributions here.
* :func:`common_language_effect_size` -- ``P(X > Y)``, the probability a random draw
  from arm A beats one from arm B (the most communicable effect size).
* :func:`cohens_d` -- standardized mean difference (pooled SD).
* :func:`cohens_h` -- effect size for a difference of two proportions (arcsine).
* :func:`odds_ratio` -- Haldane-corrected odds ratio for two success counts.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "CliffsDelta",
    "cliffs_delta",
    "cohens_d",
    "cohens_h",
    "common_language_effect_size",
    "odds_ratio",
]


@dataclass(frozen=True, slots=True)
class CliffsDelta:
    """Cliff's delta with its qualitative magnitude label."""

    delta: float
    magnitude: str


def _pairwise_signs(a: Sequence[float], b: Sequence[float]) -> tuple[int, int, int]:
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    if av.size == 0 or bv.size == 0:
        raise ValueError("both samples must be non-empty")
    diff = av[:, None] - bv[None, :]
    greater = int(np.sum(diff > 0))
    less = int(np.sum(diff < 0))
    ties = int(diff.size - greater - less)
    return greater, less, ties


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> CliffsDelta:
    r"""Return Cliff's delta :math:`(\#a>b - \#a<b)/(n_a n_b)` with a magnitude label.

    Romano thresholds: |d|<0.147 negligible, <0.33 small, <0.474 medium, else large.
    """
    greater, less, ties = _pairwise_signs(a, b)
    total = greater + less + ties
    delta = (greater - less) / total
    mag = abs(delta)
    if mag < 0.147:
        label = "negligible"
    elif mag < 0.33:
        label = "small"
    elif mag < 0.474:
        label = "medium"
    else:
        label = "large"
    return CliffsDelta(delta=delta, magnitude=label)


def common_language_effect_size(a: Sequence[float], b: Sequence[float]) -> float:
    """Return ``P(X > Y)`` (ties counted as 0.5) -- the probability of superiority."""
    greater, _less, ties = _pairwise_signs(a, b)
    total = greater + _less + ties
    return (greater + 0.5 * ties) / total


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Return Cohen's d: standardized mean difference with pooled standard deviation."""
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    if av.size < 2 or bv.size < 2:
        raise ValueError("each sample needs at least two observations")
    na, nb = av.size, bv.size
    pooled_var = ((na - 1) * av.var(ddof=1) + (nb - 1) * bv.var(ddof=1)) / (na + nb - 2)
    if pooled_var == 0:
        return 0.0
    return float((av.mean() - bv.mean()) / math.sqrt(pooled_var))


def cohens_h(p1: float, p2: float) -> float:
    r"""Return Cohen's h for two proportions, :math:`2\arcsin\sqrt{p_1}-2\arcsin\sqrt{p_2}`."""
    for p in (p1, p2):
        if not 0.0 <= p <= 1.0:
            raise ValueError("proportions must be in [0, 1]")
    return float(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))


def odds_ratio(successes_a: int, trials_a: int, successes_b: int, trials_b: int) -> float:
    """Return the Haldane-corrected odds ratio of two success counts."""
    if not (0 <= successes_a <= trials_a and 0 <= successes_b <= trials_b):
        raise ValueError("require 0 <= successes <= trials")
    a = successes_a + 0.5
    b = (trials_a - successes_a) + 0.5
    c = successes_b + 0.5
    d = (trials_b - successes_b) + 0.5
    return float((a * d) / (b * c))
