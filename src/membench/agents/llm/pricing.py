"""Model pricing table (dollars per million input/output tokens).

Centralised so cost math reads from exactly one place. Claude-family prices are
from the published Anthropic pricing table (cached 2026-06; update here on a
reprice). GPT and open-weight rows are placeholders flagged for verification
against current vendor pricing before any dollar figure is published; they do not
affect the offline reproducible runs, which use the deterministic stub model.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "PAPER_SESSION_RATES",
    "PRICING",
    "BlendedSessionRate",
    "ModelPricing",
    "blended_session_dollars",
    "price_dollars",
]


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


@dataclass(frozen=True, slots=True)
class BlendedSessionRate:
    """A flat (input/output-blended) dollar rate per million session tokens.

    The paper's token-economics tables (``sec:tokecon``) publish *session
    tokens* and *USD* without an input/output split, so the only price they
    state is a blended per-session-token rate. These rates are implied by the
    published rows themselves — ``usd_per_mtok`` is the constant that
    reproduces every USD cell of the cited table from its token cell at the
    printed precision — not vendor quotes; ``provenance`` names the exact table.
    """

    usd_per_mtok: float
    provenance: str


PAPER_SESSION_RATES: dict[str, BlendedSessionRate] = {
    # tab:tok_agent_context_gpt55 ("Multi-turn collapse, one fixed backend
    # model (GPT-5.5)"): all 8 rows satisfy USD == session_tok x $12.00/Mtok at
    # the printed precision, e.g. 1835 tok x 12e-6 = $0.02202 -> printed $0.022
    # and 4737 x 12e-6 = $0.056844 -> printed $0.0568. The caption states the
    # model and "its per-token price are identical across rows".
    "gpt-5.5": BlendedSessionRate(12.00, "tab:tok_agent_context_gpt55"),
    # tab:tok_agent_economics ("Quality x cost across context layers", USD
    # (gpt-4o) column): all 16 rows satisfy USD == session_tok x $4.60/Mtok at
    # the printed precision, e.g. Brief 12400 x 4.6e-6 = $0.05704 -> printed
    # $0.057 and Kluris 51289 x 4.6e-6 = $0.2359294 -> printed $0.2359.
    "gpt-4o": BlendedSessionRate(4.60, "tab:tok_agent_economics"),
}
"""Blended session rates implied by the paper's own token-economics tables.

Deliberate omissions (integrity note): the sweep's other backend models named in
``sec:tokecon`` / ``tab:tok_context_brief_losses`` — GPT-5.3 Codex, Claude Opus
4.8, Mistral Large 3, Gemini 3.5 Flash, Composer 2.5, and Kimi K2.5 — appear
only with token counts, never with a USD column, so no paper price exists for
them and none is invented here. (Claude Opus 4.8 has a vendor-sourced
input/output row in :data:`PRICING`; that is Anthropic's published price, not a
paper-stated one, and the two tables are kept separate for exactly that reason.)
"""


def blended_session_dollars(model: str, session_tokens: int) -> float:
    """Dollar cost of a session under the paper-implied blended rate.

    ``session_tokens`` is the prompt+completion total across all turns (the
    "Session tok." column). Raises ``KeyError`` for models the paper does not
    price (fail loud rather than silently zero-cost, matching
    :func:`price_dollars`).
    """
    if session_tokens < 0:
        raise ValueError("session_tokens must be non-negative")
    if model not in PAPER_SESSION_RATES:
        raise KeyError(
            f"the paper states no blended session rate for model {model!r}; "
            f"known: {sorted(PAPER_SESSION_RATES)}"
        )
    return session_tokens * PAPER_SESSION_RATES[model].usd_per_mtok / 1_000_000


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
