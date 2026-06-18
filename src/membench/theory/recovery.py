r"""Full-chain recovery bounds for similarity vs. structured memory.

Acting correctly on a task at the head of a dependence chain
:math:`n_0 \to n_1 \to \dots \to n_d` requires recovering every upstream link.
If each node is recovered independently with probability equal to its
retrievability, the two memory regimes recover the *whole* chain with very
different probabilities.

**Similarity memory.** Multiplying the per-hop retrievabilities of the decay
model :math:`s_i = s_0 \rho^i` over the ``d`` hops gives

.. math::

    P_\text{sim}(d) = \prod_{i=1}^{d} s_0 \rho^{i}
                    = s_0^{\,d}\, \rho^{\,d(d+1)/2}.

The exponent :math:`d(d+1)/2` is the whole story (Proposition 1, the *depth
ceiling*): chain recovery falls off faster than any fixed exponential rate, so
for deep chains the agent is reduced to guessing the very links that hold the
task together.

**Structured memory.** Following an explicit link succeeds with probability
:math:`q` close to one, independent of whether two connected facts resemble each
other, so

.. math::

    P_\text{struct}(d) = q^{\,d},

which decays only gently. Their ratio
:math:`P_\text{struct}/P_\text{sim} = q^d / (s_0^d \rho^{d(d+1)/2}) \to \infty`
as :math:`d \to \infty`: the advantage of structure grows without bound with
depth.

All quantities are available in log-space as well, because the similarity product
underflows float64 for even moderately deep chains and the crossover solver and
plots must stay numerically honest there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from membench.theory.decay import SimilarityDecayModel

__all__ = [
    "RecoveryModel",
    "log_similarity_recovery",
    "similarity_recovery",
    "structured_recovery",
]


def _check_depth(d: int) -> None:
    if d < 1:
        raise ValueError(f"depth d must be >= 1, got {d}")


def similarity_recovery(d: int, model: SimilarityDecayModel) -> float:
    r"""Probability similarity memory recovers a full depth-``d`` chain.

    Evaluates :math:`P_\text{sim}(d) = s_0^{d} \rho^{d(d+1)/2}` directly.
    """
    _check_depth(d)
    return float(model.s0**d * model.rho ** (d * (d + 1) / 2))


def log_similarity_recovery(d: int, model: SimilarityDecayModel) -> float:
    r"""Natural log of :func:`similarity_recovery` (stable for large ``d``).

    :math:`\log P_\text{sim}(d) = d \log s_0 + \tfrac{d(d+1)}{2}\log \rho`.
    """
    _check_depth(d)
    return d * math.log(model.s0) + (d * (d + 1) / 2) * math.log(model.rho)


def structured_recovery(d: int, q: float) -> float:
    r"""Probability structured memory recovers a full depth-``d`` chain, :math:`q^d`.

    Parameters
    ----------
    d
        Chain depth (>= 1).
    q
        Per-link recovery probability, in :math:`(0, 1]`.
    """
    _check_depth(d)
    if not 0.0 < q <= 1.0:
        raise ValueError(f"q must be in (0, 1], got {q}")
    return float(q**d)


@dataclass(frozen=True, slots=True)
class RecoveryModel:
    r"""Bundles the similarity-decay law and the structured per-link reliability.

    Parameters
    ----------
    decay
        The similarity-decay model :math:`s_i = s_0 \rho^i`.
    q
        Structured-memory per-link recovery probability, in :math:`(0, 1]`.
    """

    decay: SimilarityDecayModel
    q: float

    def __post_init__(self) -> None:
        if not 0.0 < self.q <= 1.0:
            raise ValueError(f"q must be in (0, 1], got {self.q}")

    def p_sim(self, d: int) -> float:
        """Similarity full-chain recovery at depth ``d``."""
        return similarity_recovery(d, self.decay)

    def p_struct(self, d: int) -> float:
        """Structured full-chain recovery at depth ``d``."""
        return structured_recovery(d, self.q)

    def log_advantage(self, d: int) -> float:
        r"""Log advantage :math:`\log P_\text{struct} - \log P_\text{sim}` at depth ``d``.

        Computed in log-space so it stays finite where ``p_sim`` underflows; this
        is the quantity the crossover solver compares against the overhead ratio.
        """
        _check_depth(d)
        return d * math.log(self.q) - log_similarity_recovery(d, self.decay)

    def advantage_ratio(self, d: int) -> float:
        r"""Ratio :math:`P_\text{struct}(d) / P_\text{sim}(d)` (may overflow to inf)."""
        return float(np.exp(self.log_advantage(d)))

    def p_sim_curve(self, max_depth: int) -> NDArray[np.float64]:
        """Vector of similarity recovery for depths ``1..max_depth``."""
        return np.array([self.p_sim(d) for d in range(1, max_depth + 1)], dtype=np.float64)

    def p_struct_curve(self, max_depth: int) -> NDArray[np.float64]:
        """Vector of structured recovery for depths ``1..max_depth``."""
        return np.array([self.p_struct(d) for d in range(1, max_depth + 1)], dtype=np.float64)
