"""Anthropic Messages API backend (real LLM calls for live runs)."""

from __future__ import annotations

from typing import Any

from membench.agents.llm.base import LLMClient, LLMResponse
from membench.agents.llm.pricing import PRICING, price_dollars

__all__ = ["AnthropicLLMClient"]


class AnthropicLLMClient(LLMClient):
    """Calls the Anthropic Messages API and records token usage and dollar cost.

    Parameters
    ----------
    model
        A model id present in the pricing table (e.g. ``claude-sonnet-4-6``).
    """

    def __init__(
        self, model: str = "claude-sonnet-4-6", *, timeout: float = 60.0, max_retries: int = 3
    ) -> None:
        if model not in PRICING:
            raise ValueError(f"no pricing entry for model {model!r}")
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: Any | None = None

    @property
    def model_name(self) -> str:
        """The configured Claude model id."""
        return self._model

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise RuntimeError(
                    "AnthropicLLMClient needs the anthropic extra: uv sync --extra anthropic"
                ) from exc
            self._client = anthropic.Anthropic(timeout=self._timeout, max_retries=self._max_retries)
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        """Generate a completion via the Messages API with usage accounting."""
        response = self._ensure_client().messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        usage = response.usage
        return LLMResponse(
            text=text,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            dollars=price_dollars(self._model, usage.input_tokens, usage.output_tokens),
            model=self._model,
        )
