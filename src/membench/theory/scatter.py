r"""The scatter penalty: the cost of spreading one decision across many areas.

A governing decision is rarely stored in one place. When the facts that make up a
single decision are *scattered* across :math:`\sigma` distinct distractor areas of
the corpus, recovering the decision is no longer a single lookup: the agent must
locate and dereference a fragment in each of the :math:`\sigma` areas and assemble
them into the whole. If each per-area dereference succeeds independently with
probability :math:`p \in (0, 1]` (the per-area assembly fidelity), the chance of
assembling *all* :math:`\sigma` fragments is the product of the per-area
successes — a series-system reliability:

.. math::

    P_\text{asm}(\sigma) = p^{\,\sigma}.

This is the scatter penalty law (``eq:scatter`` / Theorem ``thm:scatter``,
"Depth, Not Length: A Benchmark and Theory of Memory, Retrieval, and Decision
Compliance in LLM Coding Agents", Varanasi et al., 2025, §"Scatter: The Cost of
Spreading a Decision"). It is the assembly term of the scatter-entropy bound
(``thm:scatentropy``): a decision scattered over :math:`\sigma` of :math:`N` areas
carries at least :math:`\sigma \log_2(N/\sigma)` bits of location information and
needs at least :math:`\sigma / \bar p` probes, so assembly succeeds with
probability at most :math:`\bar p^{\,\sigma}`. A single typed dereference
collapses :math:`\sigma` to :math:`1` and pays only :math:`p`; flat similarity
storage that splits a depth-``d`` chain across the corpus pays the full
:math:`\sigma = d`.

The penalty is *multiplicative* on top of any base recovery probability. Composed
with the similarity-decay chain recovery
:func:`membench.theory.recovery.similarity_recovery`, a chain of depth ``d`` that
is also scattered across :math:`\sigma` areas is recovered with probability

.. math::

    P_\text{scat}(d, \sigma) = P_\text{sim}(d)\, p^{\,\sigma}
                             = s_0^{\,d}\, \rho^{\,d(d+1)/2}\, p^{\,\sigma}.

Setting :math:`\sigma = 0` (the decision sits in one place, nothing to assemble)
leaves the base recovery unchanged, since :math:`p^0 = 1`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from membench.theory.decay import SimilarityDecayModel
from membench.theory.recovery import similarity_recovery

__all__ = [
    "ScatterDegradation",
    "ScatterModel",
    "degrade_recovery",
    "scatter_factor",
    "scattered_recovery",
]


def _check_p(p: float) -> None:
    if not 0.0 < p <= 1.0:
        raise ValueError(f"p must be in (0, 1], got {p}")


def _check_sigma(sigma: float) -> None:
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")


def _check_recovery(recovery: float) -> None:
    if not 0.0 <= recovery <= 1.0:
        raise ValueError(f"base recovery must be in [0, 1], got {recovery}")


def scatter_factor(p: float, sigma: float) -> float:
    r"""Multiplicative scatter penalty :math:`p^{\sigma}`.

    Parameters
    ----------
    p
        Per-area assembly fidelity, in :math:`(0, 1]`. The probability a single
        scattered fragment is located and dereferenced.
    sigma
        Scatter count: the number of distinct distractor areas the decision is
        spread across (non-negative; may be fractional for a fitted scatter).

    Returns
    -------
    float
        The factor :math:`p^{\sigma} \in (0, 1]`. Equals ``1.0`` when
        ``sigma == 0`` (nothing to assemble) and equals ``p`` when ``sigma == 1``.
    """
    _check_p(p)
    _check_sigma(sigma)
    return float(p**sigma)


def scattered_recovery(d: int, model: SimilarityDecayModel, p: float, sigma: float) -> float:
    r"""Similarity chain recovery degraded by the scatter penalty.

    Composes :func:`membench.theory.recovery.similarity_recovery` with the scatter
    factor: :math:`P_\text{scat}(d, \sigma) = P_\text{sim}(d)\, p^{\sigma}`.

    Parameters
    ----------
    d
        Chain depth (>= 1).
    model
        The similarity-decay law :math:`s_i = s_0 \rho^i`.
    p
        Per-area assembly fidelity, in :math:`(0, 1]`.
    sigma
        Scatter count (non-negative).

    Returns
    -------
    float
        Degraded recovery probability. Reduces to ``similarity_recovery(d, model)``
        when ``sigma == 0``.
    """
    return similarity_recovery(d, model) * scatter_factor(p, sigma)


@dataclass(frozen=True, slots=True)
class ScatterDegradation:
    r"""The effect of scattering a fixed base recovery across :math:`\sigma` areas.

    Attributes
    ----------
    sigma
        The scatter count applied.
    p
        The per-area assembly fidelity used.
    base_recovery
        The un-scattered recovery probability that was degraded.
    factor
        The scatter factor :math:`p^{\sigma}` applied.
    degraded_recovery
        ``base_recovery * factor`` — the recovery after the scatter penalty.
    """

    sigma: float
    p: float
    base_recovery: float
    factor: float
    degraded_recovery: float


def degrade_recovery(base_recovery: float, p: float, sigma: float) -> ScatterDegradation:
    r"""Apply the scatter penalty to an arbitrary base recovery probability.

    Parameters
    ----------
    base_recovery
        The un-scattered recovery probability, in :math:`[0, 1]` (for example a
        value from :func:`membench.theory.recovery.similarity_recovery` or an
        empirically measured recall).
    p
        Per-area assembly fidelity, in :math:`(0, 1]`.
    sigma
        Scatter count (non-negative).

    Returns
    -------
    ScatterDegradation
        The base recovery, the scatter factor, and the degraded recovery.
    """
    _check_recovery(base_recovery)
    factor = scatter_factor(p, sigma)
    return ScatterDegradation(
        sigma=float(sigma),
        p=float(p),
        base_recovery=float(base_recovery),
        factor=factor,
        degraded_recovery=float(base_recovery * factor),
    )


@dataclass(frozen=True, slots=True)
class ScatterModel:
    r"""Bundles the per-area assembly fidelity :math:`p` of the scatter penalty law.

    Mirrors :class:`membench.theory.recovery.RecoveryModel`: a small immutable value
    object carrying the one free parameter, with helpers that apply the law.

    Parameters
    ----------
    p
        Per-area assembly fidelity, in :math:`(0, 1]`. The fitted synthetic value
        in the paper's organization sweep is :math:`p \approx 0.92`.
    """

    p: float

    def __post_init__(self) -> None:
        _check_p(self.p)

    def factor(self, sigma: float) -> float:
        r"""Return the scatter factor :math:`p^{\sigma}` at scatter ``sigma``."""
        return scatter_factor(self.p, sigma)

    def degrade(self, base_recovery: float, sigma: float) -> ScatterDegradation:
        """Apply the penalty to ``base_recovery`` at scatter ``sigma``."""
        return degrade_recovery(base_recovery, self.p, sigma)

    def scattered_recovery(self, d: int, decay: SimilarityDecayModel, sigma: float) -> float:
        r"""Similarity chain recovery at depth ``d`` scattered across ``sigma`` areas."""
        return scattered_recovery(d, decay, self.p, sigma)

    def factor_curve(self, max_sigma: int) -> NDArray[np.float64]:
        r"""Return scatter factors :math:`p^{\sigma}` for ``sigma`` in ``0..max_sigma``."""
        if max_sigma < 0:
            raise ValueError(f"max_sigma must be non-negative, got {max_sigma}")
        sigmas = np.arange(0, max_sigma + 1, dtype=np.float64)
        return np.asarray(self.p**sigmas, dtype=np.float64)
