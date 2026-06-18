"""A deterministic stub LLM for reproducible, offline, API-free benchmark runs.

The stub simulates an agent that uses the context it is given: it honours exactly
the decisions whose concrete identifiers (PascalCase / camelCase / function calls /
scoped packages) appear in the retrieved context. So an arm that recovers a
governing decision into the prompt scores compliant, and an arm that does not
scores a miss -- the accuracy result is driven entirely by retrieval quality. Its
token counts are the real prompt/response sizes, so the token-economics result
(precise arms cost less) also falls out deterministically. This is what lets the
full benchmark run green with no keys while still demonstrating the thesis; real
backends are swapped in for live confirmation.
"""

from __future__ import annotations

import re

from membench.agents.llm.base import LLMClient, LLMResponse
from membench.agents.llm.pricing import price_dollars

__all__ = ["StubLLMClient"]

# Concrete code-identifier patterns (shared in spirit with the compliance scorer).
_IDENTIFIER = re.compile(
    r"[A-Za-z][A-Za-z0-9_]*\(\)"  # function calls: withAuditLog()
    r"|@[\w-]+/[\w-]+"  # scoped packages: @t3-oss/env-nextjs
    r"|\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b"  # PascalCase: DateRangePicker
    r"|\b[a-z]+(?:[A-Z][a-z0-9]*)+\b"  # camelCase: withAuditLog
)
_CHARS_PER_TOKEN = 4


class StubLLMClient(LLMClient):
    """Deterministic context-using agent: echoes identifiers found in the prompt.

    Parameters
    ----------
    model
        Pricing/attribution label (defaults to the nominal ``"stub"`` price row).
    """

    def __init__(self, model: str = "stub") -> None:
        self._model = model

    @property
    def model_name(self) -> str:
        """The stub's pricing/attribution label."""
        return self._model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        """Honour every identifier present in the prompt; account tokens and cost."""
        identifiers = list(dict.fromkeys(_IDENTIFIER.findall(user)))
        if identifiers:
            text = "Implementation plan honouring: " + "; ".join(identifiers)
        else:
            text = "No governing decisions found in context; proceeding from the request alone."
        input_tokens = (len(system) + len(user)) // _CHARS_PER_TOKEN
        output_tokens = max(1, len(text) // _CHARS_PER_TOKEN)
        dollars = price_dollars(self._model, input_tokens, output_tokens)
        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            dollars=dollars,
            model=self._model,
        )
