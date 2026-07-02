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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from membench.agents.llm.base import LLMClient
from membench.agents.runner import _SYSTEM_PROMPT, RunConfig, run_task
from membench.retrieval.base import MemorySystem
from membench.types import Task

__all__ = [
    "DEFAULT_FOLLOW_UP_PROMPT",
    "PaperLedgerRow",
    "Session",
    "TurnRecord",
    "collapse_turns",
    "competitor_saving_pct",
    "ledger_rows",
    "matchup_winner",
    "paper_ledger_row",
    "run_session",
    "tokens_per_resolved_point",
    "win_rate",
]

DEFAULT_FOLLOW_UP_PROMPT = (
    "Verify the plan against the governing decisions and finish the task."
)
"""Fixed follow-up prompt for turns >= 2 (identical across arms: fairness lock).

Context is injected once on turn 1 (``sec:tokecon``); later turns continue the
conversation, so their prompt is the previous response plus this constant. Any
per-arm variation here would leak tokens into the very quantity under test.
"""


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


# ---------------------------------------------------------------------------
# Session ledger
# ---------------------------------------------------------------------------


def ledger_rows(session: Session) -> list[dict[str, Any]]:
    """Return the per-turn token ledger of a session with cumulative columns.

    One JSON-friendly row per turn: per-turn prompt/completion/total tokens and
    dollars, plus running (cumulative) token and dollar sums. The final row's
    cumulative values equal :attr:`Session.session_tokens` /
    :attr:`Session.session_dollars` by construction — this is the "turn-by-turn
    token ledger" the paper commits to publishing for every backend model
    (``sec:tokecon``, closing paragraph).
    """
    rows: list[dict[str, Any]] = []
    running_tokens = 0
    running_dollars = 0.0
    for record in session.turns:
        running_tokens += record.turn_tokens
        running_dollars += record.dollars
        rows.append(
            {
                "task_id": session.task_id,
                "arm": session.arm,
                "model": session.model,
                "turn": record.turn,
                "prompt_tokens": record.prompt_tokens,
                "completion_tokens": record.completion_tokens,
                "turn_tokens": record.turn_tokens,
                "dollars": record.dollars,
                "cumulative_tokens": running_tokens,
                "cumulative_dollars": running_dollars,
                "resolved": record.resolved,
            }
        )
    return rows


@dataclass(frozen=True, slots=True)
class PaperLedgerRow:
    """One row in the presentation shape of ``tab:tok_agent_context_gpt55``.

    The paper collapses a session's per-turn spend to four columns — Turn 1,
    Turn 2, Turn 3, and "Turn 4+" (the sum of all remaining turns) — plus the
    session total. Columns beyond the session's length are ``None`` (rendered
    "---" in the paper; e.g. the 3-turn "Brief context" row has no Turn 4+).

    The defining arithmetic identity, asserted against every published row in
    the tests: ``turn1 + turn2 + turn3 + turn4_plus == session_tokens``
    (treating ``None`` as 0).
    """

    turn1: int | None
    turn2: int | None
    turn3: int | None
    turn4_plus: int | None
    session_tokens: int

    def __post_init__(self) -> None:
        columns = (self.turn1, self.turn2, self.turn3, self.turn4_plus)
        total = sum(column for column in columns if column is not None)
        if total != self.session_tokens:
            raise ValueError(
                f"ledger row does not sum: columns total {total}, "
                f"session_tokens {self.session_tokens}"
            )


def collapse_turns(per_turn_tokens: Sequence[int]) -> PaperLedgerRow:
    """Collapse a per-turn token sequence to the paper's T1/T2/T3/T4+ columns.

    ``per_turn_tokens[i]`` is the total token spend of turn ``i+1``. Turns 4
    onward are summed into the ``turn4_plus`` column, exactly the presentation
    rule of ``tab:tok_agent_context_gpt55``; absent columns are ``None``.
    """
    if any(tokens < 0 for tokens in per_turn_tokens):
        raise ValueError("per-turn token counts must be non-negative")
    head = [
        per_turn_tokens[i] if i < len(per_turn_tokens) else None for i in range(3)
    ]
    tail = sum(per_turn_tokens[3:]) if len(per_turn_tokens) > 3 else None
    return PaperLedgerRow(
        turn1=head[0],
        turn2=head[1],
        turn3=head[2],
        turn4_plus=tail,
        session_tokens=sum(per_turn_tokens),
    )


def paper_ledger_row(session: Session) -> PaperLedgerRow:
    """Render a session in the column shape of ``tab:tok_agent_context_gpt55``."""
    return collapse_turns([record.turn_tokens for record in session.turns])


# ---------------------------------------------------------------------------
# Multi-turn driver over the single-turn offline runner
# ---------------------------------------------------------------------------


def run_session(
    task: Task,
    memory: MemorySystem,
    llm: LLMClient,
    config: RunConfig,
    *,
    arm_name: str,
    is_resolved: Callable[[str, Task], bool],
    max_turns: int = 8,
    follow_up_prompt: str = DEFAULT_FOLLOW_UP_PROMPT,
) -> Session:
    """Run one task as a multi-turn session and return its per-turn ledger.

    Implements the session mechanics of ``sec:tokecon``: the context layer is
    injected once, on turn 1, and the backend model is billed on every turn it
    runs until the task resolves or the turn cap is hit. Turn 1 is delegated to
    the existing single-turn offline runner (:func:`membench.agents.runner.run_task`,
    single-shot), so the corpus write, fairness-locked retrieval budget, and
    prompt construction are byte-identical to the single-turn benchmark; turns
    2..n continue the conversation on the previous response plus a fixed
    follow-up, with *no* re-retrieval.

    The session stops at the first turn whose output satisfies ``is_resolved``
    (that turn is the convergence turn), else after ``max_turns`` — the paper's
    "*k* turns, unresolved" outcome (``tab:tok_agent_context_gpt55``).

    Parameters
    ----------
    task, memory, llm, config, arm_name
        Exactly as for :func:`membench.agents.runner.run_task`. ``config``'s
        retry policy is ignored: within a session, continuation is a *turn*,
        never a hidden retry, so per-turn tokens stay a reported control.
    is_resolved
        Resolution predicate on (response text, task). Deciding resolution is
        the caller's contract (offline runs use a deterministic predicate; the
        paper's live sweep uses each benchmark's own resolution check).
    max_turns
        Turn cap (>= 1). Default 8 spans the paper's longest published session
        ("8 turns, unresolved", Mem0 row of ``tab:tok_agent_context_gpt55``).
    follow_up_prompt
        Constant continuation prompt for turns >= 2 (fairness-locked).
    """
    if max_turns < 1:
        raise ValueError(f"max_turns must be >= 1, got {max_turns}")

    # Turn 1: the single-turn offline runner, forced single-shot (attempts and
    # turns must not compound; a session's unit of repetition is the turn).
    first = run_task(task, memory, llm, config, arm_name=arm_name, is_correct=None)[0]
    turns = [
        TurnRecord(
            turn=1,
            prompt_tokens=first.input_tokens,
            completion_tokens=first.output_tokens,
            dollars=first.dollars,
            response_text=first.response_text,
            resolved=is_resolved(first.response_text, task),
        )
    ]

    while not turns[-1].resolved and len(turns) < max_turns:
        user = f"{turns[-1].response_text}\n\n{follow_up_prompt}"
        response = llm.complete(_SYSTEM_PROMPT, user, config.max_output_tokens)
        turns.append(
            TurnRecord(
                turn=len(turns) + 1,
                prompt_tokens=response.input_tokens,
                completion_tokens=response.output_tokens,
                dollars=response.dollars,
                response_text=response.text,
                resolved=is_resolved(response.text, task),
            )
        )

    return Session(
        task_id=task.task_id,
        dataset=task.dataset,
        arm=arm_name,
        model=first.model,
        turns=tuple(turns),
        max_turns=max_turns,
        retrieved_ids=first.retrieved_ids,
        governing_decisions=task.governing_decisions,
    )


# ---------------------------------------------------------------------------
# Matchup and economics arithmetic (tab:tok_context_winrate and friends)
# ---------------------------------------------------------------------------


def matchup_winner(brief_tokens: int, competitor_tokens: int) -> str | None:
    """Resolve one session-token matchup under the paper's rule.

    A matchup is one (LLM x task x competing context layer) cell and "the
    winner is the configuration that spent fewer total session tokens to the
    same backend model" (``tab:tok_context_winrate``). Returns ``"brief"`` or
    ``"competitor"``, or ``None`` on an exact tie. Ties are representable but
    the paper's sweep recorded zero of them ("There are no ties: on every cell
    one configuration is strictly cheaper").
    """
    if brief_tokens < 0 or competitor_tokens < 0:
        raise ValueError("session token counts must be non-negative")
    if brief_tokens < competitor_tokens:
        return "brief"
    if competitor_tokens < brief_tokens:
        return "competitor"
    return None


def win_rate(wins: int, matchups: int) -> float:
    """Fraction of matchups won: ``wins / matchups``.

    The paper's headline is ``2880 / 3600 = 0.8`` exactly — an 80.0% session-
    token win rate over 12 LLMs x 30 tasks x 10 context layers
    (``tab:tok_context_winrate``); per-competitor rates over 360 matchups each
    are in ``tab:tok_context_by_competitor``.
    """
    if matchups <= 0:
        raise ValueError(f"matchups must be positive, got {matchups}")
    if not 0 <= wins <= matchups:
        raise ValueError(f"wins must be in [0, {matchups}], got {wins}")
    return wins / matchups


def competitor_saving_pct(brief_tokens: int, competitor_tokens: int) -> float:
    """Competitor's token saving as a percentage of Brief's session spend.

    ``100 * (brief_tokens - competitor_tokens) / brief_tokens`` — the
    "Comp. saves %" column of the honest-losses table
    (``tab:tok_context_brief_losses``), a margin on token cost, not on outcome.
    Defined for the loss rows, where the competitor is cheaper; negative when
    Brief was cheaper.
    """
    if brief_tokens <= 0:
        raise ValueError(f"brief_tokens must be positive, got {brief_tokens}")
    if competitor_tokens < 0:
        raise ValueError("competitor_tokens must be non-negative")
    return 100.0 * (brief_tokens - competitor_tokens) / brief_tokens


def tokens_per_resolved_point(session_tokens: float, resolution_pct: float) -> float:
    """Mean session tokens per point of resolution rate.

    ``session_tokens / resolution_pct`` with resolution in percent — the
    "Tok./res. pt" column of ``tab:tok_agent_economics`` (e.g. Brief:
    12400 tokens at 48.0% resolution -> 258.3 tokens per resolved point).
    """
    if session_tokens < 0:
        raise ValueError("session_tokens must be non-negative")
    if resolution_pct <= 0:
        raise ValueError(f"resolution_pct must be positive, got {resolution_pct}")
    return session_tokens / resolution_pct
