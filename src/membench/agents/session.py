"""Multi-turn agent sessions with a per-turn token ledger (paper sec:tokecon).

Results VII of the paper ("Token Economics and Multi-Turn Efficiency",
``sec:tokecon``) measures what compliance *costs* by running a fixed backend model
over a multi-turn coding session under different turn-1 context layers and reading
off the per-turn token spend. This module supplies the missing session-level data
structure the single-turn :mod:`membench.agents.runner` cannot express:

``TurnLedger``
    One turn's prompt/completion token counts, its per-call dollar cost, and
    whether the task had resolved by that turn.
``Session``
    An ordered ledger of turns plus the derived quantities the paper's tables
    report: total session tokens, the turn at which the session converged, and
    the blended-rate dollar cost.
``run_session``
    A deterministic driver that composes the existing offline runner: turn 1 goes
    through :func:`membench.agents.runner.run_task` (context injected once, as in
    the paper), and follow-up turns call the same :class:`~membench.agents.llm.base.LLMClient`
    until a caller-supplied resolution predicate fires or a turn cap is hit.

Paper anchors implemented / verified here:

- ``tab:tok_agent_context_gpt55`` -- per-turn token ledger (T1/T2/T3/T4+), session
  total, USD, and outcome for one fixed backend (GPT-5.5). The identity that the
  ``Session tok.`` column equals the sum of the per-turn columns is asserted
  verbatim for all eight rows in the test module.
- ``tab:tok_agent_economics`` -- pooled resolution %, session tokens, USD (gpt-4o),
  tokens-per-resolved-point, and the composite efficiency score. The two derived
  columns are exposed as :func:`tokens_per_resolved_point` and
  :func:`efficiency_score` and cross-checked against all sixteen published rows.

The dollar figures use the *blended per-token session rate* the paper's own cost
model implies (``sec:tokecon``: "the backend model and its per-token price are
identical across arms, [so] the entire cost difference ... is the difference in how
many tokens they burn"); the rate lives in :mod:`membench.agents.llm.pricing`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from membench.agents.llm.base import LLMClient, LLMResponse
from membench.agents.llm.pricing import session_blended_cost
from membench.agents.runner import RunConfig, run_task
from membench.retrieval.base import MemorySystem
from membench.types import Task

__all__ = [
    "ResolutionPredicate",
    "Session",
    "TurnLedger",
    "confirm_follow_up",
    "efficiency_score",
    "run_session",
    "tokens_per_resolved_point",
]

# A resolution predicate reports whether the task has resolved given the current
# turn index and that turn's response text. Turn-aware so a session can be driven
# to a deterministic convergence turn in tests (and so a real judge could resolve
# on the transcript so far).
ResolutionPredicate = Callable[[int, str, Task], bool]

# The user prompt for turns >= 2. The paper injects context once on turn 1; later
# turns are short confirm/continue nudges, which is why they collapse to a couple
# hundred then a handful of tokens once the governing decision is already present.
FollowUpBuilder = Callable[[int, Task], str]

_SESSION_SYSTEM_PROMPT = (
    "You are a coding agent in a multi-turn session. Honour the team decisions in the "
    "context delivered on turn 1; on later turns, confirm and continue the implementation."
)


def confirm_follow_up(turn: int, task: Task) -> str:
    """Return the default turn->=2 prompt: a short, deterministic continue nudge.

    Deliberately does not re-inject the corpus (the paper delivers context once on
    turn 1), so later turns are cheap when the decision is already in hand.
    """
    return "Continue the implementation, honouring the decisions already provided."


@dataclass(frozen=True, slots=True)
class TurnLedger:
    """One turn of a session: its token spend, cost, and resolution state.

    Parameters
    ----------
    turn
        1-based turn index within the session.
    prompt_tokens, completion_tokens
        Input and output tokens billed for this turn's single model call. The
        paper's per-turn columns report the *total* (``prompt + completion``); the
        prompt/completion split is not published, so :meth:`from_total` records a
        published per-turn total without inventing a split (see the module note on
        omissions).
    resolved
        Whether the task had resolved *by the end of this turn*. The session's
        convergence turn is the first turn for which this is true.
    dollars
        The per-call dollar cost reported by the backend for this turn, when the
        session was produced by :func:`run_session`. It is ``0.0`` for ledgers
        built from a published per-turn total via :meth:`from_total`, where the
        cost is instead reconstructed from the blended session rate.
    """

    turn: int
    prompt_tokens: int
    completion_tokens: int
    resolved: bool
    dollars: float = 0.0

    def __post_init__(self) -> None:
        if self.turn < 1:
            raise ValueError(f"turn must be >= 1, got {self.turn}")
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if self.dollars < 0:
            raise ValueError("dollars must be non-negative")

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens billed this turn (the paper's per-turn cell)."""
        return self.prompt_tokens + self.completion_tokens

    @classmethod
    def from_total(
        cls, turn: int, total_tokens: int, *, resolved: bool, dollars: float = 0.0
    ) -> TurnLedger:
        """Build a turn from a published per-turn *total* (split unpublished).

        The paper's ``tab:tok_agent_context_gpt55`` reports one total-token figure
        per turn; the prompt/completion split behind it is not published. We record
        the whole total on ``prompt_tokens`` so that :attr:`total_tokens` matches
        the published cell exactly. Every session-level quantity the paper reports
        (session tokens, cost) depends only on the total, and is therefore invariant
        to how the split is imputed -- a property the test module asserts directly.
        """
        return cls(
            turn=turn,
            prompt_tokens=total_tokens,
            completion_tokens=0,
            resolved=resolved,
            dollars=dollars,
        )

    @classmethod
    def from_response(cls, turn: int, response: LLMResponse, *, resolved: bool) -> TurnLedger:
        """Build a turn from a live/offline :class:`LLMResponse` (the runner path)."""
        return cls(
            turn=turn,
            prompt_tokens=response.input_tokens,
            completion_tokens=response.output_tokens,
            resolved=resolved,
            dollars=response.dollars,
        )


@dataclass(frozen=True, slots=True)
class Session:
    """An ordered per-turn ledger for one multi-turn coding session (sec:tokecon).

    The turn tuple is the raw evidence; every reported quantity in
    ``tab:tok_agent_context_gpt55`` and ``tab:tok_agent_economics`` is derived from
    it. A ``Session`` is deliberately metrics-free about the model and the cost
    rate: it carries the backend ``model`` id so the blended-rate cost can be
    reconstructed (see :meth:`blended_cost_dollars`), but it stores only counts and
    resolution flags, never a scoring decision.

    Parameters
    ----------
    model
        Backend model id billed on every turn (e.g. ``"gpt-5.5"``). Fixed across
        the session, matching the paper's "one fixed backend model" design.
    arm_label
        The turn-1 context layer wrapped around the backend (e.g. ``"brief_graph"``,
        ``"mem0"``, ``"none"``) -- the axis the sweep varies.
    turns
        The per-turn ledger rows, in order. May be empty (a session that made no
        model call: session tokens 0, never resolved).
    """

    model: str
    arm_label: str
    turns: tuple[TurnLedger, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "turns", tuple(self.turns))
        for expected, turn in enumerate(self.turns, start=1):
            if turn.turn != expected:
                raise ValueError(
                    f"turns must be 1-based and contiguous; expected turn {expected}, "
                    f"got {turn.turn}"
                )
        # A resolution is absorbing: nothing runs after the task is resolved.
        resolved_seen = False
        for turn in self.turns:
            if resolved_seen:
                raise ValueError("no turn may follow a resolved turn (resolution is absorbing)")
            resolved_seen = turn.resolved

    @property
    def num_turns(self) -> int:
        """Number of turns (model calls) the session ran (the outcome's turn count)."""
        return len(self.turns)

    @property
    def session_tokens(self) -> int:
        """Total tokens billed over the whole session (paper ``Session tok.`` column).

        Equal to the sum of the per-turn totals; the identity that this equals the
        sum of the published per-turn columns is asserted for every row of
        ``tab:tok_agent_context_gpt55``.
        """
        return sum(turn.total_tokens for turn in self.turns)

    @property
    def total_prompt_tokens(self) -> int:
        """Prompt tokens summed over all turns."""
        return sum(turn.prompt_tokens for turn in self.turns)

    @property
    def total_completion_tokens(self) -> int:
        """Completion tokens summed over all turns."""
        return sum(turn.completion_tokens for turn in self.turns)

    @property
    def reported_dollars(self) -> float:
        """Sum of the per-call dollar costs reported on each turn (runner path)."""
        return sum(turn.dollars for turn in self.turns)

    @property
    def resolved(self) -> bool:
        """Whether the session resolved the task (paper outcome ``resolved``)."""
        return any(turn.resolved for turn in self.turns)

    @property
    def convergence_turn(self) -> int | None:
        """First turn index at which the task resolved, or ``None`` if unresolved.

        This is the paper's headline collapse quantity: with ``Brief`` context the
        session converges at turn 2--3; with no context or a consolidation layer it
        runs long or never resolves.
        """
        for turn in self.turns:
            if turn.resolved:
                return turn.turn
        return None

    def cumulative_tokens(self) -> tuple[int, ...]:
        """Return the running total of session tokens after each turn (monotone)."""
        running = 0
        totals: list[int] = []
        for turn in self.turns:
            running += turn.total_tokens
            totals.append(running)
        return tuple(totals)

    def blended_cost_dollars(self, model: str | None = None) -> float:
        """Return the session's blended-rate dollar cost (paper ``USD`` column).

        Uses :func:`membench.agents.llm.pricing.session_blended_cost` at the
        session's total token count. ``model`` defaults to this session's
        :attr:`model`; pass it explicitly to price the same token ledger under a
        different backend (e.g. the ``tab:tok_agent_economics`` gpt-4o rate).

        Raises
        ------
        KeyError
            If the model has no blended session rate.
        """
        return session_blended_cost(model or self.model, self.session_tokens)

    def cumulative_dollars(self, model: str | None = None) -> tuple[float, ...]:
        """Return the blended-rate dollar cost accrued after each turn (monotone)."""
        priced = model or self.model
        return tuple(session_blended_cost(priced, tokens) for tokens in self.cumulative_tokens())


def run_session(
    task: Task,
    memory: MemorySystem,
    llm: LLMClient,
    config: RunConfig,
    *,
    arm_name: str,
    resolves_at: ResolutionPredicate,
    max_turns: int = 8,
    follow_up: FollowUpBuilder = confirm_follow_up,
) -> Session:
    """Drive a multi-turn session over the offline runner and return its ledger.

    Turn 1 goes through :func:`membench.agents.runner.run_task` (single-shot), so
    the corpus is written and retrieved exactly as in the single-turn harness and
    the context is injected once, matching the paper's design. Follow-up turns call
    the same :class:`LLMClient` with :func:`follow_up` prompts, accumulating a
    :class:`TurnLedger` per turn, until ``resolves_at`` fires or ``max_turns`` is
    reached (the paper's "runs long / unresolved" case). The driver is fully
    deterministic given a deterministic backend: identical inputs yield an equal
    :class:`Session`.

    Parameters
    ----------
    task, memory, llm, config
        The same four inputs the single-turn runner takes; ``config`` supplies the
        shared retrieval budget and ``max_output_tokens``. Its retry policy is not
        used here -- each turn is a single model call.
    arm_name
        The turn-1 context layer, recorded as :attr:`Session.arm_label`.
    resolves_at
        Predicate ``(turn, response_text, task) -> bool``; the first turn for which
        it is true is the convergence turn and the session stops.
    max_turns
        Hard cap on turns (>= 1). If never resolved by then the session is returned
        unresolved (``convergence_turn is None``).
    follow_up
        Builds the user prompt for turns >= 2 (defaults to :func:`confirm_follow_up`).

    Raises
    ------
    ValueError
        If ``max_turns`` < 1.
    """
    if max_turns < 1:
        raise ValueError(f"max_turns must be >= 1, got {max_turns}")

    turn_config = RunConfig(
        budget_tokens=config.budget_tokens,
        retry_policy="single_shot",
        max_output_tokens=config.max_output_tokens,
    )
    first = run_task(task, memory, llm, turn_config, arm_name=arm_name)[0]
    turns: list[TurnLedger] = [
        TurnLedger(
            turn=1,
            prompt_tokens=first.input_tokens,
            completion_tokens=first.output_tokens,
            resolved=resolves_at(1, first.response_text, task),
            dollars=first.dollars,
        )
    ]

    while not turns[-1].resolved and len(turns) < max_turns:
        turn = len(turns) + 1
        response = llm.complete(
            _SESSION_SYSTEM_PROMPT, follow_up(turn, task), config.max_output_tokens
        )
        turns.append(
            TurnLedger.from_response(
                turn, response, resolved=resolves_at(turn, response.text, task)
            )
        )

    return Session(model=first.model, arm_label=arm_name, turns=tuple(turns))


def tokens_per_resolved_point(session_tokens: float, resolution_pct: float) -> float:
    """Return session tokens per resolved point (``tab:tok_agent_economics`` column).

    Defined as ``session_tokens / resolution_pct`` with ``resolution_pct`` a
    percentage in ``(0, 100]`` (e.g. Brief: ``12400 / 48.0 = 258.3``). Lower is
    better: fewer tokens spent per unit of resolved work.

    Raises
    ------
    ValueError
        If ``resolution_pct`` is not in ``(0, 100]``.
    """
    if not 0 < resolution_pct <= 100:
        raise ValueError(f"resolution_pct must be in (0, 100], got {resolution_pct}")
    return session_tokens / resolution_pct


def efficiency_score(resolution_pct: float, session_tokens: float) -> float:
    """Return the composite efficiency score (``tab:tok_agent_economics`` column).

    Defined as ``1e5 * resolution_pct / session_tokens`` -- resolved points per
    hundred-thousand tokens (e.g. Brief: ``1e5 * 48.0 / 12400 = 387.1``). Higher is
    better; it is the reciprocal of :func:`tokens_per_resolved_point` up to the
    ``1e5`` scale.

    Raises
    ------
    ValueError
        If ``session_tokens <= 0`` or ``resolution_pct`` is negative.
    """
    if session_tokens <= 0:
        raise ValueError("session_tokens must be positive")
    if resolution_pct < 0:
        raise ValueError("resolution_pct must be non-negative")
    return 1e5 * resolution_pct / session_tokens
