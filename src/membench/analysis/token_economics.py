"""Token-economics tournament: session-cheaper matchups across the sweep (Results VII).

Implements the matchup machinery of the paper's Section ``sec:tokecon`` ("Results
VII: Token Economics and Multi-Turn Efficiency"). The economics sweep holds the
backend coding model fixed and swaps the *context layer* wrapped around it
(``Brief``, ``Mem0``, ``Zep``, ..., or no layer at all); because the model and its
per-token price are identical across arms, the entire cost difference between two
configurations is the difference in total session tokens.

A **matchup** is one (LLM x task x competing context layer) cell: the Brief
session on that (LLM, task) cell is paired against the competitor's session on
the same cell, and the winner is the configuration that spent *strictly fewer*
total session tokens (Table ``tab:tok_context_winrate``). Equal spend is a tie;
the paper's sweep produced **0 ties** ("There are no ties: on every cell one
configuration is strictly cheaper").

Paper anchors implemented or verified here
------------------------------------------
* ``tab:tok_context_winrate`` — 3600 matchups = 12 LLMs x 30 SWE agent tasks x
  10 context layers; Brief cheaper in 2880 (80.0%), loses 720 (20.0%), ties 0.
* ``tab:tok_context_by_competitor`` / ``fig:tokwinrate`` (F105) — the same 3600
  matchups resolved per layer (360 each), win rates spanning 68.6% (Oiya) to
  86.7% (Oracle Summary), 77.8% vs. ``none``.
* ``tab:tok_agent_economics`` — composite efficiency score (quality per token)
  and tokens per resolved point per context layer.
* ``tab:tok_agent_context_gpt55`` — per-turn token ledgers; a session's cost is
  the sum of its per-turn spends.
* ``tab:tok_context_brief_losses`` — competitor token-savings margin on the 720
  loss rows.

Everything in this module is a pure function over minimal typed session/ledger
records; nothing here calls a model or reads global state.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "BRIEF_SYSTEM",
    "MatchupOutcome",
    "SessionRecord",
]

BRIEF_SYSTEM = "brief"
"""Canonical system id of the Brief context layer (the tournament's fixed side)."""


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """One multi-turn agent session under a fixed backend model and context layer.

    Minimal typed record for the economics sweep (paper ``sec:tokecon``): a
    session is identified by the context layer wrapped around the backend model
    (``system``), the backend model itself (``llm``), and the SWE agent task
    (``task``); its cost is the total token spend summed over every turn
    (``session_tokens``, the "Session tok." column of
    ``tab:tok_agent_context_gpt55``).

    Parameters
    ----------
    system:
        Context-layer id, e.g. ``"brief"``, ``"mem0"``, ``"none"``.
    llm:
        Backend model id (12 in the paper's sweep).
    task:
        SWE agent task id (30 in the paper's sweep).
    session_tokens:
        Total tokens billed over all turns of the session. Strictly positive:
        every session pays at least its turn-1 prompt.
    resolved:
        Optional task outcome (the "Outcome" column); ``None`` when unknown.
    wall_clock_minutes:
        Optional wall-clock duration; ``None`` when unmeasured.
    """

    system: str
    llm: str
    task: str
    session_tokens: int
    resolved: bool | None = None
    wall_clock_minutes: float | None = None

    def __post_init__(self) -> None:
        """Validate the record's domain (positive integer token spend)."""
        if self.session_tokens <= 0:
            raise ValueError(
                f"session_tokens must be a positive integer, got {self.session_tokens!r}"
            )
        if self.wall_clock_minutes is not None and self.wall_clock_minutes < 0:
            raise ValueError(
                f"wall_clock_minutes must be non-negative, got {self.wall_clock_minutes!r}"
            )


class MatchupOutcome(Enum):
    """Outcome of one session-cheaper matchup, from Brief's side.

    The winner of a matchup is the configuration that spent *strictly fewer*
    session tokens (``tab:tok_context_winrate``); equal spend is a tie. Ties
    are counted in the matchup denominator but in neither wins nor losses; the
    paper's sweep has 0 ties.
    """

    BRIEF_WIN = "brief_win"
    BRIEF_LOSS = "brief_loss"
    TIE = "tie"
