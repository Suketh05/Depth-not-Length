"""Multi-turn sessions with a per-turn token ledger (paper Results VII, ``sec:tokecon``).

The paper's token-economics section evaluates *sessions*, not single completions:
a backend coding model is billed on every turn it runs, the context layer is
injected once on turn 1, and the entire cost difference between configurations is
how many tokens they burn before the task resolves — dominated by *turn count*
(paper ``sec:tokecon``; per-turn ledger table ``tab:tok_agent_context_gpt55``).
This module supplies the session primitives that section's tables are built from:

* :class:`TurnRecord` / :class:`Session` — one turn's prompt/completion token
  spend and the whole multi-turn trajectory, with the convergence turn (the turn
  at which the task resolved) recorded explicitly.
* :func:`run_session` — a deterministic multi-turn driver layered over the
  existing single-turn offline runner (:func:`membench.agents.runner.run_task`):
  turn 1 writes the corpus, retrieves under the fairness-locked budget, and
  prompts; later turns continue the conversation without re-injecting context.
* Ledger helpers — per-turn rows with cumulative token/dollar columns, and the
  paper's presentation collapse to Turn 1 / Turn 2 / Turn 3 / Turn 4+ columns
  (``tab:tok_agent_context_gpt55``).
* Matchup helpers — the session-token win-rate rule of
  ``tab:tok_context_winrate`` (winner = strictly fewer session tokens) and the
  competitor-saving margin of ``tab:tok_context_brief_losses``.

Everything here is offline and deterministic under the stub backend; no paper
number is produced by this module — the published values are asserted as goldens
in ``tests/test_agents_session.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "Session",
    "TurnRecord",
]


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """One turn of a multi-turn session: token spend, cost, and resolution flag.

    Mirrors one cell row of the paper's per-turn ledger
    (``tab:tok_agent_context_gpt55``): each turn is a separate billed call to the
    *same* backend model, so the session cost is the sum of these rows.

    Parameters
    ----------
    turn
        1-based turn index within the session.
    prompt_tokens, completion_tokens
        Token counts for this turn as reported by the backend (per-call tokens
        are a reported control, never an optimisation target — same convention
        as :class:`membench.agents.runner.AttemptRecord`).
    dollars
        Cost of this turn under the pricing table.
    response_text
        The model output for this turn (kept so a resolution predicate and the
        compliance scorer can be applied after the fact).
    resolved
        Whether the resolution predicate accepted this turn's output. A session
        stops at the first resolved turn, so this may be ``True`` only on the
        final turn of a session (enforced by :class:`Session`).
    """

    turn: int
    prompt_tokens: int
    completion_tokens: int
    dollars: float
    response_text: str
    resolved: bool = False

    def __post_init__(self) -> None:
        if self.turn < 1:
            raise ValueError(f"turn must be >= 1, got {self.turn}")
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if self.dollars < 0:
            raise ValueError("dollars must be non-negative")

    @property
    def turn_tokens(self) -> int:
        """Total tokens billed for this turn (prompt + completion)."""
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class Session:
    """A complete multi-turn trajectory of one (task, arm, model) configuration.

    This is the row unit of the paper's token-economics sweep
    (``sec:tokecon``): one session per (backend LLM x task x context layer)
    cell, whose *session tokens* decide each matchup of
    ``tab:tok_context_winrate`` and whose outcome column
    ("*k* turns, resolved/unresolved") is exactly
    (:attr:`n_turns`, :attr:`resolved`) in ``tab:tok_agent_context_gpt55``.

    Invariants (enforced):

    * turns are numbered contiguously ``1..n`` in order;
    * a session stops at its first resolved turn, so ``resolved=True`` may
      appear only on the final turn — hence the *convergence turn*, when it
      exists, is the last turn.

    An empty ``turns`` tuple is permitted (the vacuous session): all token and
    dollar sums are ``0`` and there is no convergence turn, mirroring the
    empty-sum convention used across the metrics layer.

    Parameters
    ----------
    task_id, dataset, arm, model
        Attribution keys, identical in meaning to
        :class:`membench.agents.runner.AttemptRecord`.
    turns
        The per-turn ledger, in turn order.
    max_turns
        The turn cap the driver ran under (a session that reaches it without
        resolving is the paper's "*k* turns, unresolved" outcome).
    retrieved_ids, governing_decisions
        Provenance of the turn-1 context injection (context is injected once on
        turn 1, per ``sec:tokecon``), kept so retrieval and compliance scoring
        remain possible on session rows.
    """

    task_id: str
    dataset: str
    arm: str
    model: str
    turns: tuple[TurnRecord, ...]
    max_turns: int
    retrieved_ids: tuple[str, ...] = ()
    governing_decisions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "turns", tuple(self.turns))
        object.__setattr__(self, "retrieved_ids", tuple(self.retrieved_ids))
        object.__setattr__(self, "governing_decisions", tuple(self.governing_decisions))
        if self.max_turns < 0:
            raise ValueError(f"max_turns must be non-negative, got {self.max_turns}")
        if len(self.turns) > self.max_turns:
            raise ValueError(
                f"session has {len(self.turns)} turns but max_turns={self.max_turns}"
            )
        for position, record in enumerate(self.turns, start=1):
            if record.turn != position:
                raise ValueError(
                    f"turns must be numbered contiguously from 1; "
                    f"expected turn {position}, got {record.turn}"
                )
            if record.resolved and position != len(self.turns):
                raise ValueError(
                    f"turn {position} is resolved but is not the final turn; "
                    "a session stops at its first resolved turn"
                )

    @property
    def n_turns(self) -> int:
        """Number of turns the session ran (the paper's outcome-column turn count)."""
        return len(self.turns)

    @property
    def resolved(self) -> bool:
        """Whether the session ended in resolution (final turn passed the predicate)."""
        return bool(self.turns) and self.turns[-1].resolved

    @property
    def convergence_turn(self) -> int | None:
        """The 1-based turn at which the task resolved, or ``None`` if it never did.

        By the stop-at-first-resolution invariant this is always the final turn
        when it exists — e.g. the paper's "Brief context ... 3 turns, resolved"
        row (``tab:tok_agent_context_gpt55``) has convergence turn 3.
        """
        return self.n_turns if self.resolved else None

    @property
    def prompt_tokens(self) -> int:
        """Total prompt tokens across all turns."""
        return sum(record.prompt_tokens for record in self.turns)

    @property
    def completion_tokens(self) -> int:
        """Total completion tokens across all turns."""
        return sum(record.completion_tokens for record in self.turns)

    @property
    def session_tokens(self) -> int:
        """Total session token spend — the matchup currency of ``tab:tok_context_winrate``."""
        return sum(record.turn_tokens for record in self.turns)

    @property
    def session_dollars(self) -> float:
        """Total session cost in dollars (sum of the per-turn costs)."""
        return sum(record.dollars for record in self.turns)
