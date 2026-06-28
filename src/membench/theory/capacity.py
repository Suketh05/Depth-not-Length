r"""The context-free floor: how well you can guess a decision with no memory.

This module makes the benchmark's *null model* honest. When a retrieval arm
surfaces nothing useful, the agent is reduced to guessing the governing decision
from its prior alone, and information theory pins down exactly how badly that has
to go. Two classical results give the floor.

**The Bayes floor.** For a governing decision :math:`X` drawn from an alphabet of
:math:`K` choices, the best possible no-context guess is the prior mode, so the
minimum achievable error is :math:`P_e^{\text{Bayes}} = 1 - \max_x p(x)`. For a
*uniform* :math:`K`-ary decision this is :math:`1 - 1/K`, i.e. the accuracy
ceiling with no information is the chance rate :math:`1/K`.

**The Fano floor.** Fano's inequality lower-bounds the error of *any* estimator
:math:`\hat X(Y)` of :math:`X` from a side observation :math:`Y` by its residual
uncertainty,

.. math::

    H(X \mid Y) \;\le\; H_b(P_e) + P_e \log_2(|\mathcal X| - 1),

where :math:`H_b(p) = -p\log_2 p - (1-p)\log_2(1-p)` is the binary entropy
function (Cover & Thomas, *Elements of Information Theory*, 2nd ed., Theorem
2.10.1). The *context-free* case is zero mutual information, :math:`I(X;Y)=0`, so
:math:`H(X\mid Y) = H(X) = \log_2 K` for the uniform prior. The right-hand side
:math:`f(P_e)=H_b(P_e)+P_e\log_2(K-1)` is maximised at :math:`P_e=(K-1)/K`, where
it equals exactly :math:`\log_2 K`; hence at zero information the only error
probability consistent with Fano is :math:`P_e = 1 - 1/K`. The Fano bound is
therefore *tight* in the context-free limit and coincides with the Bayes floor —
the two derivations agree on the same number, :math:`1 - 1/K`.

**Binary-symmetric-channel capacity.** The companion ceiling on how much a noisy
context channel *could* convey is the BSC capacity
:math:`C(p) = 1 - H_b(p)` bits per use, for crossover (bit-flip) probability
:math:`p` (Cover & Thomas, Section 7.1.4). A useless channel (:math:`p=1/2`) has
capacity :math:`0`; a noiseless one (:math:`p=0`) has capacity :math:`1` bit.

Together these bound the bench from below: similarity retrieval whose full-chain
recovery probability collapses with depth (see
:mod:`membench.theory.recovery`) degrades toward this context-free floor, while
structured retrieval that follows explicit links keeps :math:`H(X\mid Y)` low and
stays well above it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import optimize

__all__ = [
    "ContextFreeFloor",
    "binary_entropy",
    "bsc_capacity",
    "context_free_floor",
    "fano_error_lower_bound",
]


def binary_entropy(p: float) -> float:
    r"""Binary entropy :math:`H_b(p) = -p\log_2 p - (1-p)\log_2(1-p)` in bits.

    Parameters
    ----------
    p
        A probability in :math:`[0, 1]`.

    Returns
    -------
    float
        The entropy in bits, with the usual convention :math:`H_b(0)=H_b(1)=0`
        and a maximum of ``1.0`` bit at ``p = 0.5``.

    Raises
    ------
    ValueError
        If ``p`` is outside :math:`[0, 1]`.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    if p in (0.0, 1.0):
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def bsc_capacity(p: float) -> float:
    r"""Binary-symmetric-channel capacity :math:`C(p) = 1 - H_b(p)` in bits per use.

    Parameters
    ----------
    p
        Crossover (bit-flip) probability of the channel, in :math:`[0, 1]`.

    Returns
    -------
    float
        Capacity in bits per channel use, in :math:`[0, 1]`. It is symmetric about
        ``p = 0.5`` (where it is ``0``) and equals ``1`` bit at ``p = 0`` or
        ``p = 1`` (a perfectly (anti-)correlated channel).

    Raises
    ------
    ValueError
        If ``p`` is outside :math:`[0, 1]`.
    """
    return 1.0 - binary_entropy(p)


def fano_error_lower_bound(conditional_entropy: float, alphabet_size: int) -> float:
    r"""Tightest error probability allowed by Fano's inequality.

    Returns the smallest :math:`P_e` consistent with
    :math:`H(X\mid Y) \le H_b(P_e) + P_e\log_2(K-1)` for a :math:`K`-ary target,
    i.e. the inverse of the increasing branch of
    :math:`f(P_e)=H_b(P_e)+P_e\log_2(K-1)` on :math:`[0, (K-1)/K]`. No estimator
    of ``X`` from ``Y`` can have error below this value.

    Parameters
    ----------
    conditional_entropy
        Residual uncertainty :math:`H(X\mid Y) \ge 0` in bits. Values above the
        feasible maximum :math:`\log_2 K` are clamped to it.
    alphabet_size
        Number of decision choices :math:`K \ge 1`.

    Returns
    -------
    float
        The Fano lower bound on the error probability, in :math:`[0, (K-1)/K]`.

    Raises
    ------
    ValueError
        If ``alphabet_size`` is ``< 1`` or ``conditional_entropy`` is negative.

    Notes
    -----
    The inversion is transcendental, so the increasing branch is solved
    numerically with Brent's method, mirroring
    :func:`membench.theory.crossover.find_crossover_depth`. For ``K = 1`` (no
    uncertainty) the only feasible entropy is ``0`` and the bound is ``0``.
    """
    if alphabet_size < 1:
        raise ValueError(f"alphabet_size must be >= 1, got {alphabet_size}")
    if conditional_entropy < 0.0:
        raise ValueError(f"conditional_entropy must be non-negative, got {conditional_entropy}")

    K = alphabet_size
    h_max = math.log2(K)
    p_max = (K - 1) / K
    target = min(conditional_entropy, h_max)
    if target <= 0.0:
        return 0.0
    if target >= h_max:
        return p_max

    log2_km1 = math.log2(K - 1)

    def gap(p: float) -> float:
        return binary_entropy(p) + p * log2_km1 - target

    return float(optimize.brentq(gap, 0.0, p_max))


@dataclass(frozen=True, slots=True)
class ContextFreeFloor:
    r"""The no-context guessing floor for a uniform :math:`K`-ary decision.

    With zero mutual information the Bayes floor and the Fano floor coincide at
    :math:`1 - 1/K`; both fields are reported so callers can verify the agreement.

    Attributes
    ----------
    alphabet_size
        Number of equiprobable decision choices :math:`K \ge 1`.
    prior_entropy
        Prior uncertainty :math:`H(X) = \log_2 K` bits (also :math:`H(X\mid Y)`
        here, since the context-free channel carries no information).
    bayes_accuracy
        Best achievable no-context accuracy, the chance rate :math:`1/K`.
    bayes_error
        Bayes error floor :math:`1 - 1/K`.
    fano_error_lower_bound
        Fano's information-theoretic lower bound on the error at this entropy;
        equals :attr:`bayes_error` in the context-free limit.
    """

    alphabet_size: int
    prior_entropy: float
    bayes_accuracy: float
    bayes_error: float
    fano_error_lower_bound: float


def context_free_floor(alphabet_size: int) -> ContextFreeFloor:
    r"""Compute the context-free (zero-information) floor for a uniform decision.

    Parameters
    ----------
    alphabet_size
        Number of equiprobable decision choices :math:`K \ge 1`.

    Returns
    -------
    ContextFreeFloor
        The Bayes and Fano floors, which coincide at :math:`1 - 1/K`.

    Raises
    ------
    ValueError
        If ``alphabet_size`` is ``< 1``.
    """
    if alphabet_size < 1:
        raise ValueError(f"alphabet_size must be >= 1, got {alphabet_size}")
    K = alphabet_size
    prior_entropy = math.log2(K)
    bayes_accuracy = 1.0 / K
    return ContextFreeFloor(
        alphabet_size=K,
        prior_entropy=prior_entropy,
        bayes_accuracy=bayes_accuracy,
        bayes_error=1.0 - bayes_accuracy,
        fano_error_lower_bound=fano_error_lower_bound(prior_entropy, K),
    )
