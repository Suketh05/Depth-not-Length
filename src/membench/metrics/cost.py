"""Cost-to-correct and Return on Tokens: the measured payoff metrics.

Per-call tokens are a reported control, never an optimisation target; this module
*measures* cost, it does not decide retries. Given the per-attempt costs for a task
(and whether each attempt was correct, decided upstream by compliance/correctness),
it sums tokens and dollars to the first success (cost-to-correct), and exposes the
economic-buyer framing: useful signal ``U = tokens x precision``, Return on Tokens
(correct output per token), and cost-per-correct in dollars -- the empirical
counterparts of the theoretical token-economics model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from membench.theory.information import useful_signal

__all__ = [
    "AttemptCost",
    "CostToCorrect",
    "cost_per_correct",
    "return_on_tokens",
    "score_cost_to_correct",
    "useful_signal",
]


@dataclass(frozen=True, slots=True)
class AttemptCost:
    """The token/dollar cost of one attempt and whether it was correct."""

    input_tokens: int
    output_tokens: int
    dollars: float
    correct: bool

    @property
    def total_tokens(self) -> int:
        """Input plus output tokens for this attempt."""
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class CostToCorrect:
    """Tokens and dollars spent up to (and including) the first correct attempt."""

    attempts: int
    resolved: bool
    iterations_to_correct: int | None
    total_input_tokens: int
    total_output_tokens: int
    total_dollars: float

    @property
    def total_tokens(self) -> int:
        """Total tokens summed over the attempts that were made."""
        return self.total_input_tokens + self.total_output_tokens


def score_cost_to_correct(attempts: Sequence[AttemptCost]) -> CostToCorrect:
    """Sum cost over attempts up to the first correct one (or all, if never correct)."""
    if not attempts:
        raise ValueError("need at least one attempt")
    iterations: int | None = None
    for index, attempt in enumerate(attempts, start=1):
        if attempt.correct:
            iterations = index
            break
    counted = attempts[:iterations] if iterations is not None else attempts
    return CostToCorrect(
        attempts=len(attempts),
        resolved=iterations is not None,
        iterations_to_correct=iterations,
        total_input_tokens=sum(a.input_tokens for a in counted),
        total_output_tokens=sum(a.output_tokens for a in counted),
        total_dollars=sum(a.dollars for a in counted),
    )


def return_on_tokens(correct_outputs: float, total_tokens: int) -> float:
    """Return correct output per 1k tokens spent (the Return-on-Tokens efficiency).

    ``correct_outputs`` is a count or expected count of merge-ready outputs; the
    result is scaled per 1,000 tokens so the numbers are readable.
    """
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    return 1000.0 * correct_outputs / total_tokens


def cost_per_correct(total_dollars: float, correct_outputs: float) -> float:
    """Return dollars per merge-ready output (infinite if nothing was correct)."""
    if total_dollars < 0:
        raise ValueError("total_dollars must be non-negative")
    if correct_outputs <= 0:
        return float("inf")
    return total_dollars / correct_outputs
