"""Model pricing table (dollars per million input/output tokens).

Centralised so cost math reads from exactly one place. Claude-family prices are
from the published Anthropic pricing table (cached 2026-06; update here on a
reprice). GPT and open-weight rows are placeholders flagged for verification
against current vendor pricing before any dollar figure is published; they do not
affect the offline reproducible runs, which use the deterministic stub model.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PRICING", "ModelPricing", "price_dollars"]


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Dollar cost per million input and output tokens for a model."""

    input_per_mtok: float
    output_per_mtok: float
    verified: bool = True


PRICING: dict[str, ModelPricing] = {
    # Claude family (published Anthropic pricing).
    "claude-opus-4-8": ModelPricing(5.00, 25.00),
    "claude-sonnet-4-6": ModelPricing(3.00, 15.00),
    "claude-haiku-4-5": ModelPricing(1.00, 5.00),
    "claude-fable-5": ModelPricing(10.00, 50.00),
    # Robustness-row models: placeholders pending verified current pricing.
    "gpt-5.1": ModelPricing(5.00, 15.00, verified=False),
    "llama-4-maverick": ModelPricing(0.20, 0.60, verified=False),
    # Deterministic offline stub: a fixed nominal price so token-cost differences
    # between arms (driven by retrieved-context size) are visible and comparable.
    "stub": ModelPricing(1.00, 5.00),
}


def price_dollars(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return the dollar cost of a call under the pricing table.

    Raises
    ------
    KeyError
        If the model has no pricing entry (fail loud rather than silently zero-cost).
    """
    if model not in PRICING:
        raise KeyError(f"no pricing entry for model {model!r}; add one to PRICING")
    pricing = PRICING[model]
    return (
        input_tokens * pricing.input_per_mtok + output_tokens * pricing.output_per_mtok
    ) / 1_000_000
