"""The LLM client contract and per-call response record."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

__all__ = ["LLMClient", "LLMResponse"]


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """A single model completion with its token and cost accounting.

    Attributes
    ----------
    text
        The generated text.
    input_tokens, output_tokens
        Token counts as reported by the backend (the model field of each result
        row; per-call tokens are a reported control, never an optimisation target).
    dollars
        Cost of this call under the pricing table (a primary payoff metric).
    model
        The model identifier that produced this response.
    """

    text: str
    input_tokens: int
    output_tokens: int
    dollars: float
    model: str


class LLMClient(ABC):
    """A language model the agent runner can call. Backends are interchangeable."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model identifier (used for pricing and result attribution)."""

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        """Generate a completion for the system+user prompt with token accounting."""
        raise NotImplementedError
