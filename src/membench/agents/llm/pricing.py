"""Model pricing table (dollars per million input/output tokens).

Centralised so cost math reads from exactly one place. Claude-family prices are
from the published Anthropic pricing table (cached 2026-06; update here on a
reprice). GPT and open-weight rows are placeholders flagged for verification
against current vendor pricing before any dollar figure is published; they do not
affect the offline reproducible runs, which use the deterministic stub model.

Token-economics (paper ``sec:tokecon``) blended rates
------------------------------------------------------
The token-economics tables cost a whole *session* by its total token count under a
single per-token price for the fixed backend model (``sec:tokecon``: "the backend
model and its per-token price are identical across arms, [so] the entire cost
difference between two configurations is the difference in how many tokens they
burn"). The paper does **not** publish an explicit ``$/Mtok`` figure or an
input/output split for these backends; it publishes ``(session tokens, USD)`` pairs
in ``tab:tok_agent_context_gpt55`` (backend GPT-5.5) and ``tab:tok_agent_economics``
(billed at gpt-4o rates). :data:`SESSION_BLENDED_RATE_PER_MTOK` therefore records the
single blended per-token rate the paper's own USD columns *imply* -- derived from,
and verified to reproduce (to the paper's rounding), every published row -- rather
than an invented input/output split. See :data:`TOKECON_BACKEND_MODELS` for the
sweep's named backends whose individual per-token prices the paper does not publish.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "PRICING",
    "SESSION_BLENDED_RATE_PER_MTOK",
    "TOKECON_BACKEND_MODELS",
    "ModelPricing",
    "SessionBlendedRate",
    "price_dollars",
    "session_blended_cost",
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


@dataclass(frozen=True, slots=True)
class SessionBlendedRate:
    """A single blended dollars-per-million-token rate for a whole session.

    The paper costs a session as ``session_tokens * rate`` under one blended price
    for the fixed backend, not a separate input/output split. ``verified`` records
    that the rate reproduces the paper's published USD column (not that it is a
    vendor-quoted list price).

    Attributes
    ----------
    per_mtok
        Blended dollars per one million session tokens.
    source_table
        The paper table label whose ``(session tokens, USD)`` rows this rate
        reproduces.
    verified
        ``True`` when the rate has been checked to reproduce every published USD
        row in ``source_table`` to the paper's own rounding.
    """

    per_mtok: float
    source_table: str
    verified: bool = True


# Blended per-token session rates implied by the paper's token-economics tables.
#
# GPT-5.5 (tab:tok_agent_context_gpt55): the unique rate (to the cent) that rounds
# every published (session tokens -> USD) row correctly. Worked check on two rows:
#   3686 tok -> $0.0442  and  1835 tok -> $0.022  =>  rate ~ 0.0442/3686 ~ 1.199e-5,
#   i.e. $12.00/Mtok. Neighbours fail: at $11.99 the 2172-tok row gives $0.02604
#   (paper $0.0261); at $12.01 the 3686-tok row gives $0.04427 (paper $0.0442).
# gpt-4o (tab:tok_agent_economics): $4.60/Mtok reproduces all sixteen pooled rows,
#   e.g. 45235 tok -> $0.2081, 44767 tok -> $0.2059, 50298 tok -> $0.2314.
# Neither figure is published verbatim; both are derived from, and verified
# against, the published USD columns (see the test module for the row-by-row check).
SESSION_BLENDED_RATE_PER_MTOK: dict[str, SessionBlendedRate] = {
    "gpt-5.5": SessionBlendedRate(12.00, source_table="tab:tok_agent_context_gpt55"),
    "gpt-4o": SessionBlendedRate(4.60, source_table="tab:tok_agent_economics"),
}


# Backend models the token-economics sweep names (12 total; these are the ones
# stated by name in tab:tok_context_brief_losses / sec:tokecon prose). The paper
# publishes NO per-token or input/output price for any individual backend -- only
# the two blended session rates above -- so no ModelPricing row is invented for
# them. Listed for provenance and to make the omission explicit.
TOKECON_BACKEND_MODELS: frozenset[str] = frozenset(
    {
        "gpt-5.5",
        "gpt-5.3-codex",
        "claude-opus-4-8",
        "mistral-large-3",
        "gemini-3.5-flash",
        "composer-2.5",
        "kimi-k2.5",
    }
)


def session_blended_cost(model: str, session_tokens: int) -> float:
    """Return the blended-rate dollar cost of a session (``sec:tokecon`` cost model).

    Cost is ``session_tokens * rate`` under the single blended per-token price the
    paper applies to the fixed backend, so cost tracks token count (and hence turn
    count) exactly.

    Raises
    ------
    KeyError
        If ``model`` has no blended session rate (fail loud rather than mis-price a
        session under an unrelated per-call table).
    ValueError
        If ``session_tokens`` is negative.
    """
    if session_tokens < 0:
        raise ValueError("session_tokens must be non-negative")
    if model not in SESSION_BLENDED_RATE_PER_MTOK:
        raise KeyError(
            f"no blended session rate for model {model!r}; add one to SESSION_BLENDED_RATE_PER_MTOK"
        )
    return session_tokens * SESSION_BLENDED_RATE_PER_MTOK[model].per_mtok / 1_000_000
