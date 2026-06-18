"""LLMClient abstraction. Anthropic backend only -- other backends (GPT,
open-weight) are the model-robustness arms and land in a later step.

Every call records input_tokens, output_tokens, and dollars: per-call tokens
are a reported, never-optimized control (CLAUDE.md), and cost-to-correct in
dollars is a primary payoff metric, so both must travel with every response
rather than being computed after the fact from logs that may not have them.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

# Dollars per million tokens (input, output). Source: shared Claude API
# pricing table, cached 2026-06-04. Update here if Anthropic reprices --
# this is the only place cost math reads from.
_PRICING_PER_MILLION = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5": (10.00, 50.00),
}


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    dollars: float
    model: str


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        raise NotImplementedError


class AnthropicLLMClient(LLMClient):
    """Real Anthropic Messages API backend."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        if model not in _PRICING_PER_MILLION:
            raise ValueError(f"no pricing entry for model {model!r}")
        import anthropic

        self._model = model
        self._client = anthropic.Anthropic()

    def complete(self, system: str, user: str, max_tokens: int = 4096) -> LLMResponse:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        input_price, output_price = _PRICING_PER_MILLION[self._model]
        dollars = (
            response.usage.input_tokens * input_price
            + response.usage.output_tokens * output_price
        ) / 1_000_000
        return LLMResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            dollars=dollars,
            model=self._model,
        )
