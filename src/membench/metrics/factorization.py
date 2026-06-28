r"""Factorize observed compliance into retrieval times a use-factor.

The headline quantity of the benchmark is :math:`P_\text{comply}`, the rate at which
an arm's output honours the governing decisions. This module decomposes it into the
two mechanisms it actually depends on, via the **chain rule of probability**:

.. math::

    P_\text{comply}
        = P(\text{retrieved} \cap \text{used})
        = \underbrace{P(\text{retrieved})}_{P_\text{ret}}
          \cdot \underbrace{P(\text{used} \mid \text{retrieved})}_{\kappa}.

:math:`P_\text{ret}` is estimated by retrieval recall (did the arm surface the gold
governing decisions at all) and the **use-factor** :math:`\kappa = P_\text{comply} /
P_\text{ret}` is the conditional probability that a *retrieved* decision is actually
acted on. The two are complementary failure modes -- a low :math:`P_\text{ret}` is a
retrieval problem, a low :math:`\kappa` is a reasoning/grounding problem -- and the
product reconstructs the observed compliance exactly, which is the diagnostic point:
it tells you *where* a memory arm is losing compliance.

Counts are pooled (micro-averaged) across the rows of a group: the per-row recall is
the expected number of gold decisions retrieved and the per-row compliance the
expected number honoured, so summing gives the retrieved/honoured totals from which
:math:`\kappa` and its interval are computed. The interval on :math:`\kappa` is a
**Wilson score interval** for a binomial proportion, which is well behaved at the
extreme rates (near 0 or 1) and the small effective samples this benchmark produces,
unlike the Wald (normal-approximation) interval.

References
----------
.. [1] Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive
   NLP Tasks." *NeurIPS 33*. (The retrieve-then-use two-stage decomposition.)
.. [2] Wilson, E. B. (1927). "Probable Inference, the Law of Succession, and
   Statistical Inference." *Journal of the American Statistical Association*,
   22(158), 209-212. (The score interval used for :math:`\kappa`.)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum
from typing import TYPE_CHECKING

from scipy import stats

if TYPE_CHECKING:
    from membench.analysis.tables import ScoredRow

__all__ = [
    "Factorization",
    "GroupFactorization",
    "factorize_rates",
    "factorize_rows",
    "wilson_interval",
]


def wilson_interval(
    successes: float,
    trials: float,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    r"""Wilson score interval for a binomial proportion.

    Solves the score equation rather than inverting the Wald statistic, so the
    interval stays inside :math:`[0, 1]` and behaves well for small ``trials`` and
    extreme rates [2]_:

    .. math::

        \frac{\hat p + z^2/2n \pm z\sqrt{\hat p(1-\hat p)/n + z^2/4n^2}}{1 + z^2/n},

    with :math:`\hat p = \text{successes}/\text{trials}`, :math:`n = \text{trials}`
    and :math:`z` the standard-normal quantile for ``confidence``.

    Parameters
    ----------
    successes : float
        Number of successes (may be fractional -- pooled expected counts are allowed).
    trials : float
        Number of trials; must be positive.
    confidence : float, optional
        Two-sided confidence level in ``(0, 1)``, by default ``0.95``.

    Returns
    -------
    tuple of float
        The ``(low, high)`` bounds, both clamped to ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``trials <= 0``, ``successes < 0``, or ``confidence`` is not in ``(0, 1)``.
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes < 0:
        raise ValueError("successes must be non-negative")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    p_hat = min(max(successes / trials, 0.0), 1.0)
    z = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p_hat + z2 / (2.0 * trials)) / denominator
    margin = (
        z * (p_hat * (1.0 - p_hat) / trials + z2 / (4.0 * trials * trials)) ** 0.5 / denominator
    )
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return low, high


@dataclass(frozen=True, slots=True)
class Factorization:
    """The ``P_comply = P_ret * kappa`` decomposition for one group of rows.

    Attributes
    ----------
    n : int
        Number of rows pooled into the estimate.
    p_ret : float
        Estimated retrieval probability (mean recall of the gold decisions).
    kappa : float or None
        Use-factor ``P_comply / P_ret`` -- the conditional probability a retrieved
        decision is honoured. ``None`` when ``p_ret == 0`` (undefined, not a crash).
        Theoretically in ``[0, 1]``; a value above 1 signals compliance reached
        without retrieving the gold id (e.g. via the prompt) and is reported raw.
    p_comply : float
        Reconstructed compliance ``p_ret * kappa``; equals ``p_comply_observed`` by
        construction (and is ``0`` when ``p_ret == 0``).
    p_comply_observed : float
        Directly measured mean compliance rate, kept for the reconstruction check.
    kappa_ci : tuple of float, or None
        Wilson score interval for ``kappa``; ``None`` when ``kappa`` is undefined.
    confidence : float
        Confidence level of ``kappa_ci``.
    """

    n: int
    p_ret: float
    kappa: float | None
    p_comply: float
    p_comply_observed: float
    kappa_ci: tuple[float, float] | None
    confidence: float


@dataclass(frozen=True, slots=True)
class GroupFactorization:
    """A :class:`Factorization` tagged with the ``(arm, depth)`` group it describes."""

    arm: str
    depth: int
    factorization: Factorization


def factorize_rates(
    recall: Sequence[float],
    compliance: Sequence[float],
    *,
    confidence: float = 0.95,
) -> Factorization:
    r"""Factorize pooled compliance rates into ``P_ret * kappa``.

    Treats each ``recall[i]`` as the expected number of gold decisions retrieved and
    ``compliance[i]`` as the expected number honoured, pools them, and forms
    :math:`\kappa = \sum \text{compliance} / \sum \text{recall}`.

    Parameters
    ----------
    recall : sequence of float
        Per-row retrieval recall, each in ``[0, 1]``.
    compliance : sequence of float
        Per-row compliance rate, each in ``[0, 1]``; aligned with ``recall``.
    confidence : float, optional
        Confidence level for the Wilson interval on ``kappa``, by default ``0.95``.

    Returns
    -------
    Factorization
        The decomposition. ``kappa`` and ``kappa_ci`` are ``None`` when no recall mass
        was observed (``sum(recall) == 0``).

    Raises
    ------
    ValueError
        If the inputs are empty, of unequal length, or contain values outside
        ``[0, 1]``.
    """
    if len(recall) != len(compliance):
        raise ValueError("recall and compliance must have equal length")
    if not recall:
        raise ValueError("need at least one row")
    if any(not 0.0 <= r <= 1.0 for r in recall):
        raise ValueError("recall values must lie in [0, 1]")
    if any(not 0.0 <= c <= 1.0 for c in compliance):
        raise ValueError("compliance values must lie in [0, 1]")

    n = len(recall)
    total_ret = fsum(recall)
    total_comply = fsum(compliance)
    p_ret = total_ret / n
    p_comply_observed = total_comply / n

    if total_ret == 0.0:
        return Factorization(
            n=n,
            p_ret=p_ret,
            kappa=None,
            p_comply=0.0,
            p_comply_observed=p_comply_observed,
            kappa_ci=None,
            confidence=confidence,
        )

    kappa = total_comply / total_ret
    return Factorization(
        n=n,
        p_ret=p_ret,
        kappa=kappa,
        p_comply=p_ret * kappa,
        p_comply_observed=p_comply_observed,
        kappa_ci=wilson_interval(total_comply, total_ret, confidence=confidence),
        confidence=confidence,
    )


def factorize_rows(
    rows: Sequence[ScoredRow],
    *,
    confidence: float = 0.95,
) -> list[GroupFactorization]:
    """Factorize compliance per ``(arm, depth)`` group of scored rows.

    Parameters
    ----------
    rows : sequence of ScoredRow
        Scored attempts; their ``recall`` and ``compliance_rate`` fields are pooled
        within each ``(arm, depth)`` group.
    confidence : float, optional
        Confidence level for the Wilson interval on ``kappa``, by default ``0.95``.

    Returns
    -------
    list of GroupFactorization
        One entry per ``(arm, depth)`` group, sorted by ``(arm, depth)``.
    """
    groups: dict[tuple[str, int], tuple[list[float], list[float]]] = {}
    for row in rows:
        recall, compliance = groups.setdefault((row.arm, row.depth), ([], []))
        recall.append(row.recall)
        compliance.append(row.compliance_rate)
    return [
        GroupFactorization(
            arm=arm,
            depth=depth,
            factorization=factorize_rates(recall, compliance, confidence=confidence),
        )
        for (arm, depth), (recall, compliance) in sorted(groups.items())
    ]
