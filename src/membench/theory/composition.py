r"""Multi-hop decay composition: chaining per-hop survival along a reasoning path.

Recovering a fact at the head of a dependence chain
:math:`n_0 \to n_1 \to \dots \to n_d` requires *every* hop on the path to be
traversed successfully. Treating the hops as independent, the path is a **series
system**: it survives only if all of its components survive, so the whole-path
recovery probability is the product of the per-hop survival probabilities,

.. math::

    R(\text{path}) = \prod_{i=1}^{d} p_i,

where :math:`p_i \in [0, 1]` is the survival probability of hop :math:`i`. This is
the classical series-reliability law of reliability theory (Barlow, R. E. &
Proschan, F., *Statistical Theory of Reliability and Life Testing*, Holt,
Rinehart and Winston, 1975, ch. 2: the reliability of a series structure of
independent components is :math:`\prod_i p_i`).

The composition is **heterogeneous**: a path may mix *typed-edge* hops (following
an explicit structured link, which survives with a per-link reliability
:math:`q`, cf. :func:`membench.theory.recovery.structured_recovery`) with
*similarity-edge* hops (recovered by resemblance, which survives with the
geometric retrievability :math:`s_i = s_0 \rho^i` of
:class:`membench.theory.decay.SimilarityDecayModel`). The two whole-chain forms
of :mod:`membench.theory.recovery` are special cases of this product: a path of
``d`` similarity hops at distances :math:`1, \dots, d` recovers with
:math:`s_0^{d}\rho^{d(d+1)/2}`, and a path of ``d`` typed hops recovers with
:math:`q^{d}`.

Because the product underflows float64 for long paths, the survival is also
exposed in log-space (a sum of per-hop log-survivals), mirroring the numerical
discipline of :mod:`membench.theory.recovery`. The empty product is ``1.0``: a
path of no hops is recovered with certainty, which makes a unit-survival hop the
neutral element and the composition associative.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from membench.theory.decay import SimilarityDecayModel

__all__ = [
    "Hop",
    "HopType",
    "PathRecovery",
    "compose_log_survival",
    "compose_path",
    "compose_survival",
    "similarity_hop",
    "typed_hop",
]


class HopType(str, Enum):
    r"""The kind of edge traversed by a single hop.

    Attributes
    ----------
    TYPED
        An explicit structured link, traversed with per-link reliability ``q``.
    SIMILARITY
        A resemblance-based hop, traversed with geometric retrievability
        :math:`s_i = s_0 \rho^i`.
    """

    TYPED = "typed"
    SIMILARITY = "similarity"


def _check_survival(p: float) -> None:
    """Raise if ``p`` is not a valid survival probability in :math:`[0, 1]`."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"survival probability must be in [0, 1], got {p}")


@dataclass(frozen=True, slots=True)
class Hop:
    r"""A single hop on a reasoning path, with its survival probability.

    Parameters
    ----------
    survival
        Probability the hop is traversed successfully, in :math:`[0, 1]`.
    hop_type
        Whether the hop follows a typed (structured) edge or a similarity edge.
    """

    survival: float
    hop_type: HopType

    def __post_init__(self) -> None:
        _check_survival(self.survival)


@dataclass(frozen=True, slots=True)
class PathRecovery:
    r"""Whole-path recovery obtained by composing per-hop survival probabilities.

    Attributes
    ----------
    survival
        Product of the per-hop survival probabilities,
        :math:`\prod_{i} p_i` (``1.0`` for the empty path).
    log_survival
        Natural log of :attr:`survival`, computed as the sum of per-hop
        log-survivals so it stays finite where the product underflows
        (``-inf`` if any hop has zero survival).
    n_hops
        Number of hops composed.
    n_typed
        Number of typed-edge hops on the path.
    n_similarity
        Number of similarity-edge hops on the path.
    """

    survival: float
    log_survival: float
    n_hops: int
    n_typed: int
    n_similarity: int


def compose_survival(survivals: Sequence[float]) -> float:
    r"""Compose per-hop survival probabilities into whole-path survival.

    Returns the series-system reliability :math:`\prod_i p_i` (Barlow & Proschan,
    1975). The empty product is ``1.0``.

    Parameters
    ----------
    survivals
        Per-hop survival probabilities, each in :math:`[0, 1]`.

    Returns
    -------
    float
        The product of the survival probabilities.

    Raises
    ------
    ValueError
        If any survival probability lies outside :math:`[0, 1]`.
    """
    product = 1.0
    for p in survivals:
        _check_survival(p)
        product *= p
    return product


def compose_log_survival(survivals: Sequence[float]) -> float:
    r"""Compose per-hop survival probabilities in log-space.

    Returns :math:`\sum_i \log p_i`, the natural log of :func:`compose_survival`,
    computed as a sum so it stays finite where the product underflows. The empty
    sum is ``0.0``; a hop with zero survival yields ``-inf``.

    Parameters
    ----------
    survivals
        Per-hop survival probabilities, each in :math:`[0, 1]`.

    Returns
    -------
    float
        The sum of the per-hop log-survivals.

    Raises
    ------
    ValueError
        If any survival probability lies outside :math:`[0, 1]`.
    """
    total = 0.0
    for p in survivals:
        _check_survival(p)
        total += -math.inf if p == 0.0 else math.log(p)
    return total


def typed_hop(q: float) -> Hop:
    r"""Build a typed-edge hop with per-link reliability ``q``.

    Mirrors the per-link survival of
    :func:`membench.theory.recovery.structured_recovery` (a depth-``d`` typed
    path composes to :math:`q^d`).

    Parameters
    ----------
    q
        Per-link recovery probability, in :math:`[0, 1]`.
    """
    return Hop(survival=q, hop_type=HopType.TYPED)


def similarity_hop(model: SimilarityDecayModel, distance: int | float) -> Hop:
    r"""Build a similarity-edge hop from a decay model at a given causal distance.

    The hop survives with retrievability :math:`s_0 \rho^{\text{distance}}` from
    :meth:`membench.theory.decay.SimilarityDecayModel.retrievability`. Composing
    the hops at distances :math:`1, \dots, d` reproduces the similarity full-chain
    recovery :math:`s_0^{d}\rho^{d(d+1)/2}`.

    Parameters
    ----------
    model
        The geometric similarity-decay law.
    distance
        Causal distance of this hop (number of hops from the query node).

    Raises
    ------
    ValueError
        If the model's retrievability at ``distance`` exceeds 1 (an undecayed
        model can predict ``s > 1``, which is not a valid survival probability).
    """
    return Hop(survival=model.retrievability(distance), hop_type=HopType.SIMILARITY)


def compose_path(hops: Sequence[Hop]) -> PathRecovery:
    r"""Compose a heterogeneous path of hops into a :class:`PathRecovery`.

    Multiplies the per-hop survival probabilities (series-system reliability) and
    also reports the log-survival and the typed/similarity hop counts.

    Parameters
    ----------
    hops
        The ordered hops on the reasoning path; an empty path recovers with
        certainty (survival ``1.0``).

    Returns
    -------
    PathRecovery
        The composed whole-path recovery and its hop-type breakdown.
    """
    survivals = [hop.survival for hop in hops]
    n_typed = sum(1 for hop in hops if hop.hop_type is HopType.TYPED)
    n_similarity = sum(1 for hop in hops if hop.hop_type is HopType.SIMILARITY)
    return PathRecovery(
        survival=compose_survival(survivals),
        log_survival=compose_log_survival(survivals),
        n_hops=len(survivals),
        n_typed=n_typed,
        n_similarity=n_similarity,
    )
