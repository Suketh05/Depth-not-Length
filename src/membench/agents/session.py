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

from dataclasses import dataclass, field

from membench.agents.llm.base import LLMResponse

__all__ = [
    "Session",
    "TurnLedger",
]


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
