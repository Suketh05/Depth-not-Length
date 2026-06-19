"""OpenAI Chat Completions backend (the GPT robustness row, for live runs)."""

from __future__ import annotations

from typing import Any

from membench.agents.llm.base import LLMClient, LLMResponse
from membench.agents.llm.pricing import PRICING, price_dollars

__all__ = ["OpenAILLMClient"]


class OpenAILLMClient(LLMClient):
    """Calls the OpenAI Chat Completions API and records usage and dollar cost.

    Parameters
    ----------
    model
        A model id present in the pricing table (e.g. ``gpt-5.1``).
    """

    def __init__(self, model: str = "gpt-5.1") -> None:
        if model not in PRICING:
            raise ValueError(f"no pricing entry for model {model!r}")
        self._model = model
        self._client: Any | None = None

    @property
    def model_name(self) -> str:
        """The configured OpenAI model id."""
        return self._model

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError as exc:
                raise RuntimeError(
                    "OpenAILLMClient needs the openai extra: uv sync --extra openai"
                ) from exc
            self._client = openai.OpenAI()
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        """Generate a completion via Chat Completions with usage accounting."""
        response = self._ensure_client().chat.completions.create(
            model=self._model,
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResponse(
            text=text,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            dollars=price_dollars(self._model, usage.prompt_tokens, usage.completion_tokens),
            model=self._model,
        )
