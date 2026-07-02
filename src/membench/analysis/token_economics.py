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
    "CompetitorRecord",
    "Matchup",
    "MatchupOutcome",
    "SessionRecord",
    "TournamentSummary",
    "decide_matchup",
    "pair_matchups",
    "run_tournament",
    "session_tokens_from_turns",
    "sweep_matchup_count",
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


def decide_matchup(brief_tokens: int, competitor_tokens: int) -> MatchupOutcome:
    """Apply the strict session-cheaper rule to one matchup.

    Brief wins iff ``brief_tokens < competitor_tokens``; loses iff
    ``brief_tokens > competitor_tokens``; equal spend is a
    :attr:`MatchupOutcome.TIE`. This is the winner rule of
    ``tab:tok_context_winrate`` ("the winner is the configuration that spent
    fewer total session tokens to the same backend model").

    Parameters
    ----------
    brief_tokens:
        Brief session's total token spend (positive).
    competitor_tokens:
        Competitor session's total token spend on the same (LLM, task) cell.

    Returns
    -------
    MatchupOutcome
        Which side was strictly cheaper, or a tie on equal spend.
    """
    if brief_tokens <= 0 or competitor_tokens <= 0:
        raise ValueError(
            "session token counts must be positive, got "
            f"brief={brief_tokens!r}, competitor={competitor_tokens!r}"
        )
    if brief_tokens < competitor_tokens:
        return MatchupOutcome.BRIEF_WIN
    if brief_tokens > competitor_tokens:
        return MatchupOutcome.BRIEF_LOSS
    return MatchupOutcome.TIE


@dataclass(frozen=True, slots=True)
class Matchup:
    """One resolved (LLM x task x competitor) cell of the tournament.

    A matchup pairs the Brief session on an (LLM, task) cell against one
    competitor's session on the same cell (paper ``tab:tok_context_winrate``).
    The outcome is derived, never stored, so a matchup can never disagree with
    its own token counts.
    """

    llm: str
    task: str
    competitor: str
    brief_tokens: int
    competitor_tokens: int

    @property
    def outcome(self) -> MatchupOutcome:
        """Strict session-cheaper outcome of this cell (see :func:`decide_matchup`)."""
        return decide_matchup(self.brief_tokens, self.competitor_tokens)


def session_tokens_from_turns(turn_tokens: Sequence[int]) -> int:
    """Total session spend as the sum of a per-turn token ledger.

    The paper's per-trace ledger (``tab:tok_agent_context_gpt55``) reports
    tokens per turn and their total in the "Session tok." column; this is that
    total. E.g. the Brief-context row: ``1598 + 203 + 34 = 1835``, and the
    unaided worst case ``1403 + 698 + 601 + 1461 = 4163`` (the final bucket is
    the table's "Turn 4+" aggregate).

    Parameters
    ----------
    turn_tokens:
        Per-turn (or per-turn-bucket) token spends; each strictly positive and
        the ledger non-empty (every session has a turn 1).

    Returns
    -------
    int
        Sum of the ledger.
    """
    if len(turn_tokens) == 0:
        raise ValueError("a session ledger must contain at least one turn")
    for i, tokens in enumerate(turn_tokens):
        if tokens <= 0:
            raise ValueError(f"turn {i + 1} tokens must be positive, got {tokens!r}")
    return sum(turn_tokens)


def sweep_matchup_count(n_llms: int, n_tasks: int, n_competitors: int) -> int:
    """Number of matchups in a full crossed sweep.

    One matchup per (LLM x task x competing context layer) cell, so the total
    is the plain product. Paper instance (``tab:tok_context_winrate``):
    ``12 LLMs x 30 SWE agent tasks x 10 context layers = 3600``.

    Parameters
    ----------
    n_llms, n_tasks, n_competitors:
        Positive sweep dimensions.

    Returns
    -------
    int
        ``n_llms * n_tasks * n_competitors``.
    """
    if n_llms <= 0 or n_tasks <= 0 or n_competitors <= 0:
        raise ValueError(
            "sweep dimensions must be positive, got "
            f"n_llms={n_llms!r}, n_tasks={n_tasks!r}, n_competitors={n_competitors!r}"
        )
    return n_llms * n_tasks * n_competitors


def pair_matchups(
    brief_sessions: Iterable[SessionRecord],
    competitor_sessions: Iterable[SessionRecord],
) -> list[Matchup]:
    """Pair every competitor session against the Brief session on its cell.

    Builds the tournament of ``tab:tok_context_winrate``: for each competitor
    session on an (LLM, task) cell there must be exactly one Brief session on
    the same cell, and the pair forms one matchup. The pairing is total and
    injective by construction — a duplicate Brief cell, a duplicate
    (competitor, LLM, task) session, a competitor session with no Brief
    counterpart, or a session on the wrong side all raise ``ValueError``
    rather than silently dropping or double-counting a cell.

    Parameters
    ----------
    brief_sessions:
        Sessions whose ``system`` is :data:`BRIEF_SYSTEM`, one per (LLM, task)
        cell that appears in ``competitor_sessions``.
    competitor_sessions:
        Sessions of every non-Brief context layer (including ``"none"``).

    Returns
    -------
    list[Matchup]
        One matchup per competitor session, in input order.
    """
    brief_by_cell: dict[tuple[str, str], SessionRecord] = {}
    for record in brief_sessions:
        if record.system != BRIEF_SYSTEM:
            raise ValueError(
                f"brief_sessions must have system={BRIEF_SYSTEM!r}, got {record.system!r}"
            )
        cell = (record.llm, record.task)
        if cell in brief_by_cell:
            raise ValueError(f"duplicate Brief session for cell (llm, task)={cell!r}")
        brief_by_cell[cell] = record

    matchups: list[Matchup] = []
    seen: set[tuple[str, str, str]] = set()
    for record in competitor_sessions:
        if record.system == BRIEF_SYSTEM:
            raise ValueError("competitor_sessions must not contain Brief sessions")
        key = (record.system, record.llm, record.task)
        if key in seen:
            raise ValueError(f"duplicate competitor session for (system, llm, task)={key!r}")
        seen.add(key)
        cell = (record.llm, record.task)
        brief = brief_by_cell.get(cell)
        if brief is None:
            raise ValueError(f"no Brief session for cell (llm, task)={cell!r}")
        matchups.append(
            Matchup(
                llm=record.llm,
                task=record.task,
                competitor=record.system,
                brief_tokens=brief.session_tokens,
                competitor_tokens=record.session_tokens,
            )
        )
    return matchups


@dataclass(frozen=True, slots=True)
class CompetitorRecord:
    """Win/loss/tie tally of the Brief-vs-one-competitor slice of the tournament.

    One row of ``tab:tok_context_by_competitor`` (360 matchups per layer in the
    paper's sweep).
    """

    competitor: str
    matchups: int
    wins: int
    losses: int
    ties: int

    def __post_init__(self) -> None:
        """Validate that the tally partitions the matchups."""
        if min(self.matchups, self.wins, self.losses, self.ties) < 0:
            raise ValueError("tally counts must be non-negative")
        if self.wins + self.losses + self.ties != self.matchups:
            raise ValueError(
                f"wins + losses + ties must equal matchups: "
                f"{self.wins} + {self.losses} + {self.ties} != {self.matchups}"
            )

    @property
    def win_rate(self) -> float | None:
        """Brief's win fraction ``wins / matchups`` (``None`` on an empty slice)."""
        if self.matchups == 0:
            return None
        return self.wins / self.matchups

    @property
    def win_rate_pct(self) -> float | None:
        """Brief's win rate in percent, as printed in ``tab:tok_context_by_competitor``."""
        rate = self.win_rate
        return None if rate is None else 100.0 * rate


@dataclass(frozen=True, slots=True)
class TournamentSummary:
    """Overall + per-competitor tallies of a session-cheaper tournament.

    The overall block is ``tab:tok_context_winrate``; ``per_competitor`` is
    ``tab:tok_context_by_competitor``. Ties count in the matchup denominator
    but in neither wins nor losses.
    """

    matchups: int
    wins: int
    losses: int
    ties: int
    per_competitor: Mapping[str, CompetitorRecord]

    @property
    def win_rate(self) -> float | None:
        """Brief's overall win fraction (paper: ``2880 / 3600 = 0.800``)."""
        if self.matchups == 0:
            return None
        return self.wins / self.matchups

    @property
    def win_rate_pct(self) -> float | None:
        """Overall win rate in percent (paper headline: 80.0%)."""
        rate = self.win_rate
        return None if rate is None else 100.0 * rate

    @classmethod
    def from_matchups(cls, matchups: Iterable[Matchup]) -> TournamentSummary:
        """Tally an iterable of resolved matchups into the two paper tables.

        Parameters
        ----------
        matchups:
            Resolved (LLM x task x competitor) cells, e.g. from
            :func:`pair_matchups`.

        Returns
        -------
        TournamentSummary
            Overall and per-competitor win/loss/tie counts; competitors are
            keyed by their system id, in first-seen order.
        """
        total = wins = losses = ties = 0
        tallies: dict[str, list[int]] = {}
        for matchup in matchups:
            tally = tallies.setdefault(matchup.competitor, [0, 0, 0, 0])
            tally[0] += 1
            total += 1
            outcome = matchup.outcome
            if outcome is MatchupOutcome.BRIEF_WIN:
                tally[1] += 1
                wins += 1
            elif outcome is MatchupOutcome.BRIEF_LOSS:
                tally[2] += 1
                losses += 1
            else:
                tally[3] += 1
                ties += 1
        per_competitor = {
            competitor: CompetitorRecord(
                competitor=competitor,
                matchups=tally[0],
                wins=tally[1],
                losses=tally[2],
                ties=tally[3],
            )
            for competitor, tally in tallies.items()
        }
        return cls(
            matchups=total,
            wins=wins,
            losses=losses,
            ties=ties,
            per_competitor=per_competitor,
        )


def run_tournament(
    brief_sessions: Iterable[SessionRecord],
    competitor_sessions: Iterable[SessionRecord],
) -> TournamentSummary:
    """Pair and tally the full session-cheaper tournament in one call.

    Composition of :func:`pair_matchups` and
    :meth:`TournamentSummary.from_matchups`; on the paper's sweep this yields
    the 3600-matchup summary of ``tab:tok_context_winrate`` and the
    per-competitor breakdown of ``tab:tok_context_by_competitor``.

    Parameters
    ----------
    brief_sessions:
        One Brief session per (LLM, task) cell.
    competitor_sessions:
        Every competitor session in the sweep.

    Returns
    -------
    TournamentSummary
        Overall and per-competitor tallies.
    """
    return TournamentSummary.from_matchups(pair_matchups(brief_sessions, competitor_sessions))
