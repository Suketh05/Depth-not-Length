r"""Token economics: why the structured win is on cost as well as accuracy.

Accuracy is only half the story. Define the *useful signal* delivered to the
model as the amount of injected context actually worth reading,

.. math::

    U = (\text{tokens injected}) \times (\text{retrieval precision}).

To compensate for its low chance of catching each hop, similarity retrieval must
cast a wider net. Under the decay model :math:`s_i = s_0\rho^i`, the expected
number of candidates it must surface to capture the node at distance :math:`i` is
:math:`1/s_i` (the mean of a geometric trial with success probability
:math:`s_i`), so recovering a depth-:math:`d` chain costs it

.. math::

    T_\text{sim}(d) = c \sum_{i=1}^{d} \frac{1}{s_i}
                    = \frac{c}{s_0}\sum_{i=1}^{d}\rho^{-i},

which grows *geometrically* in depth (:math:`\rho^{-i}` explodes). Structured
memory injects exactly the chain it followed — :math:`d` nodes — so
:math:`T_\text{struct}(d) = c\,d`, linear in depth, at precision one.

The two interact with the recovery bounds to give the headline payoff metric,
expected **cost-to-correct** (tokens spent per *successful* recovery, assuming
independent retries):

.. math::

    \mathrm{C}_\text{sim}(d)    = \frac{T_\text{sim}(d)}{P_\text{sim}(d)}, \qquad
    \mathrm{C}_\text{struct}(d) = \frac{T_\text{struct}(d)}{P_\text{struct}(d)}.

The numerator and denominator both move against similarity as depth grows, so the
cost-to-correct ratio diverges: once past the crossover the structured store is
more often correct *and* reads fewer tokens — the joint win.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from membench.theory.recovery import RecoveryModel

__all__ = ["TokenEconomics", "useful_signal"]


def useful_signal(tokens_injected: float, precision: float) -> float:
    r"""Return the useful signal :math:`U = \text{tokens} \times \text{precision}`.

    Parameters
    ----------
    tokens_injected
        Number of context tokens injected into the prompt (non-negative).
    precision
        Fraction of the injected items that are on the gold chain, in ``[0, 1]``.
    """
    if tokens_injected < 0:
        raise ValueError(f"tokens_injected must be non-negative, got {tokens_injected}")
    if not 0.0 <= precision <= 1.0:
        raise ValueError(f"precision must be in [0, 1], got {precision}")
    return tokens_injected * precision


@dataclass(frozen=True, slots=True)
class TokenEconomics:
    r"""The depth-dependent cost model derived from the recovery parameters.

    Parameters
    ----------
    recovery
        The recovery model (decay law + structured per-link reliability ``q``).
    tokens_per_item
        Average token cost :math:`c` of one memory item / chain node (> 0).
    """

    recovery: RecoveryModel
    tokens_per_item: float

    def __post_init__(self) -> None:
        if self.tokens_per_item <= 0:
            raise ValueError(f"tokens_per_item must be positive, got {self.tokens_per_item}")

    # -- raw retrieval cost (tokens injected to recover the chain) --------------

    def similarity_tokens(self, d: int) -> float:
        r"""Return expected tokens similarity must inject for a depth-``d`` chain.

        :math:`T_\text{sim}(d) = c \sum_{i=1}^d 1/s_i`.
        """
        decay = self.recovery.decay
        total = sum(1.0 / decay.retrievability(i) for i in range(1, d + 1))
        return self.tokens_per_item * total

    def structured_tokens(self, d: int) -> float:
        r"""Return tokens the structured walk injects for a depth-``d`` chain, :math:`c\,d`."""
        if d < 1:
            raise ValueError(f"depth d must be >= 1, got {d}")
        return self.tokens_per_item * d

    # -- precision and useful signal --------------------------------------------

    def similarity_precision(self, d: int) -> float:
        r"""Return similarity-retrieval precision :math:`d / \sum_i 1/s_i`."""
        decay = self.recovery.decay
        denom = sum(1.0 / decay.retrievability(i) for i in range(1, d + 1))
        return d / denom

    def structured_precision(self) -> float:
        """Return the structured-walk precision (it injects exactly the chain), 1.0."""
        return 1.0

    def similarity_useful_signal(self, d: int) -> float:
        """Return the useful signal of similarity retrieval at depth ``d``."""
        return useful_signal(self.similarity_tokens(d), self.similarity_precision(d))

    def structured_useful_signal(self, d: int) -> float:
        """Return the useful signal of the structured walk at depth ``d``."""
        return useful_signal(self.structured_tokens(d), self.structured_precision())

    # -- expected cost-to-correct (the payoff metric) ---------------------------

    def similarity_cost_to_correct(self, d: int) -> float:
        r"""Return expected tokens per successful similarity recovery.

        :math:`C_\text{sim}(d) = T_\text{sim}(d)/P_\text{sim}(d)`.
        """
        return self.similarity_tokens(d) / self.recovery.p_sim(d)

    def structured_cost_to_correct(self, d: int) -> float:
        r"""Return expected tokens per successful structured recovery.

        :math:`C_\text{struct}(d) = T_\text{struct}(d)/P_\text{struct}(d)`.
        """
        return self.structured_tokens(d) / self.recovery.p_struct(d)

    def cost_to_correct_ratio(self, d: int) -> float:
        r"""Return the cost-to-correct ratio :math:`C_\text{sim}(d)/C_\text{struct}(d)`."""
        return self.similarity_cost_to_correct(d) / self.structured_cost_to_correct(d)

    def token_cost_ratio(self, d: int) -> float:
        r"""Return the raw injected-token ratio :math:`T_\text{sim}(d)/T_\text{struct}(d)`."""
        return self.similarity_tokens(d) / self.structured_tokens(d)

    def cost_to_correct_ratio_curve(self, max_depth: int) -> NDArray[np.float64]:
        """Return cost-to-correct ratios for depths ``1..max_depth``."""
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        return np.array(
            [self.cost_to_correct_ratio(d) for d in range(1, max_depth + 1)], dtype=np.float64
        )

    # -- Return on Tokens (the economic-buyer lens) -----------------------------
    # Brief frames the payoff as "Return on Invested Tokens": with a fixed token
    # budget (total spend is near-identical, < 1% difference), the win is *where*
    # the tokens go -- the share that produces merge-ready output rather than
    # rework. Return on Tokens is correct-output per injected token, the exact
    # reciprocal of cost-to-correct, surfaced under the name a CFO reads.

    def similarity_return_on_tokens(self, d: int) -> float:
        r"""Return similarity correct-output per token.

        :math:`\text{RoT}_\text{sim}(d) = P_\text{sim}(d)/T_\text{sim}(d)`.
        """
        return self.recovery.p_sim(d) / self.similarity_tokens(d)

    def structured_return_on_tokens(self, d: int) -> float:
        r"""Return structured correct-output per token.

        :math:`\text{RoT}_\text{struct}(d) = P_\text{struct}(d)/T_\text{struct}(d)`.
        """
        return self.recovery.p_struct(d) / self.structured_tokens(d)

    def return_on_tokens_ratio(self, d: int) -> float:
        """Return how many times more each structured token ships than a similarity token.

        Equals the cost-to-correct ratio by construction (Return on Tokens is its
        reciprocal per arm), expressed as "same budget, more shipped-right work".
        """
        return self.structured_return_on_tokens(d) / self.similarity_return_on_tokens(d)

    def cost_per_correct_dollars(
        self, d: int, dollars_per_token: float, *, structured: bool
    ) -> float:
        """Return expected dollars per merge-ready task for one arm at depth ``d``.

        Multiplies the arm's expected cost-to-correct (tokens per success) by the
        blended ``dollars_per_token`` price, giving the cost-per-merge-ready-task
        figure Brief reports to economic buyers.
        """
        if dollars_per_token < 0:
            raise ValueError(f"dollars_per_token must be non-negative, got {dollars_per_token}")
        tokens = (
            self.structured_cost_to_correct(d) if structured else self.similarity_cost_to_correct(d)
        )
        return tokens * dollars_per_token
