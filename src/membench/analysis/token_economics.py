r"""Token-economics tournament: session-cheaper win rate and efficiency scoring.

Implements the multi-turn token-economics analysis of Results VII
(Section ``sec:tokecon``). The sweep runs every backend model against every
SWE-agent task under every context layer, records the *session* token ledger
each configuration spends to reach an outcome, and asks -- for each
``(backend model, task)`` cell -- whether the typed decision-graph store
(``Brief``) or a competing context layer spent fewer session tokens.

Two families of published number are reconstructed here from plain session
records, so a reader can cross-check the paper against runnable code:

**Win rate (``tab:tok_context_winrate`` / ``fig:tokwinrate`` = F105).** A
*matchup* is one ``(LLM, task, competing layer)`` cell; the winner is the
configuration that spent strictly fewer total session tokens against the same
backend model. Over the full sweep -- ``12`` LLMs :math:`\times` ``30`` tasks
:math:`\times` ``10`` context layers ``= 3600`` matchups -- ``Brief`` is cheaper
in ``2880`` (``80.0\%``); it loses ``720`` (``20.0\%``); there are **0 ties**
(on every cell one configuration is strictly cheaper). Broken out per competitor
(``tab:tok_context_by_competitor``; ``360`` matchups each) the win rate ranges
from ``68.6\%`` against ``Oiya`` to ``86.7\%`` against ``Oracle Summary``.

**Efficiency score (``tab:tok_agent_economics`` / ``tab:tok_pareto``).** Quality
per token: the resolution rate a layer achieves divided by the session tokens it
spends, together with its companion *tokens per resolved point* (session tokens
per percentage-point of resolution). ``Brief`` resolves ``48.0\%`` of tasks at
``12{,}400`` session tokens -- ``258.3`` tokens per resolved point and a
composite efficiency of ``387.1`` -- against ``23.6``--``34.3\%`` at
``44``--``51``k tokens (efficiency ``46``--``77``) for the field.

The module is deliberately dependency-free over the rest of ``membench``: it
operates on a minimal :class:`SessionRecord` value type (backend model, task,
context layer, session tokens, resolved) so the tournament arithmetic can be
verified in isolation, without a session runner. All functions are pure.

Paper anchors
-------------
* ``sec:tokecon`` -- Results VII: Token Economics and Multi-Turn Efficiency.
* ``tab:tok_context_winrate`` -- 3600-matchup win rate (2880 wins, 80.0%, 0 ties).
* ``tab:tok_context_by_competitor`` -- per-competitor win rate (68.6%--86.7%).
* ``tab:tok_agent_economics`` -- resolution, session tokens, tokens per resolved
  point, and the composite efficiency score.
* ``tab:tok_pareto`` -- compliance per thousand tokens (swebench swe3).
* ``fig:tokwinrate`` (F105), ``fig:tokpareto`` (F104).
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "DEFAULT_BRIEF_LAYER",
    "EFFICIENCY_SCALE",
    "Matchup",
    "MatchupOutcome",
    "SessionRecord",
    "SystemEconomics",
    "WinRate",
    "build_matchups",
    "economics_by_system",
    "efficiency_score",
    "matchup_count",
    "tally",
    "tally_by_competitor",
    "tokens_per_resolved_point",
]

#: Scale constant for the composite efficiency score of ``tab:tok_agent_economics``.
#: With the score defined as ``EFFICIENCY_SCALE * resolution_rate / session_tokens``
#: and a "resolved point" being one percentage point of resolution, the score and
#: its companion :func:`tokens_per_resolved_point` satisfy the exact identity
#: ``efficiency_score * tokens_per_resolved_point == EFFICIENCY_SCALE / 100`` (a
#: constant ``1e5``). ``1e7`` reproduces the paper's published column, e.g. Brief's
#: ``1e7 * 0.480 / 12400 = 387.10`` (published ``387.1``).
EFFICIENCY_SCALE = 1e7

#: The context-layer name reserved for the typed decision-graph store under test.
#: The tournament pairs this layer against every other layer in the sweep.
DEFAULT_BRIEF_LAYER = "brief"


class MatchupOutcome(Enum):
    """Result of one session-token matchup, from the typed store's point of view.

    The winner of a matchup is the configuration that spent *strictly* fewer
    total session tokens against the same backend model on the same task
    (Section ``sec:tokecon``). Equal spend is a genuine tie; the paper reports
    zero ties because on every measured cell one configuration was strictly
    cheaper (``tab:tok_context_winrate``).
    """

    BRIEF_CHEAPER = "brief_cheaper"
    COMPETITOR_CHEAPER = "competitor_cheaper"
    TIE = "tie"


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """One multi-turn session's total token ledger and outcome.

    A minimal, self-contained record of what a single configuration spent on a
    single task; the tournament is computed entirely from a collection of these
    (Section ``sec:tokecon``). The token count is the *session* total summed over
    all turns (the per-turn ledger of ``tab:tok_agent_context_gpt55`` collapsed to
    its sum), not a per-turn or per-call figure.

    Parameters
    ----------
    backend_model
        The answer model driving the session (e.g. one of the 12 sweep LLMs).
        Matchups are only formed within a shared backend model.
    task_id
        The SWE-agent task identifier (one of the 30 tasks in the sweep).
    context_layer
        The memory/context system supplying context to the backend model
        (``Brief`` or a competitor; ``"none"`` is the backend run with no layer).
    session_tokens
        Total tokens spent across every turn of the session. Non-negative.
    resolved
        Whether the session resolved the task (fed the resolution rate and the
        efficiency score of ``tab:tok_agent_economics``).
    """

    backend_model: str
    task_id: str
    context_layer: str
    session_tokens: int
    resolved: bool

    def __post_init__(self) -> None:
        """Validate that the session token count is non-negative."""
        if self.session_tokens < 0:
            msg = f"session_tokens must be non-negative, got {self.session_tokens}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Matchup:
    """A single ``Brief``-vs-competitor session-token comparison on one cell.

    One matchup is a ``(backend model, task, competing layer)`` cell
    (``tab:tok_context_winrate``): the typed store and one competitor each ran a
    session against the same backend model on the same task, and the cheaper
    session (fewer tokens) wins.

    Parameters
    ----------
    backend_model
        Shared backend model for both sessions.
    task_id
        Shared task.
    competitor
        The competing context layer's name.
    brief_tokens
        Session tokens spent by the typed store.
    competitor_tokens
        Session tokens spent by the competitor.
    """

    backend_model: str
    task_id: str
    competitor: str
    brief_tokens: int
    competitor_tokens: int

    @property
    def outcome(self) -> MatchupOutcome:
        """Classify the matchup by which configuration spent fewer tokens."""
        if self.brief_tokens < self.competitor_tokens:
            return MatchupOutcome.BRIEF_CHEAPER
        if self.brief_tokens > self.competitor_tokens:
            return MatchupOutcome.COMPETITOR_CHEAPER
        return MatchupOutcome.TIE

    @property
    def brief_wins(self) -> bool:
        """``True`` iff the typed store spent strictly fewer session tokens."""
        return self.outcome is MatchupOutcome.BRIEF_CHEAPER


def build_matchups(
    sessions: Iterable[SessionRecord],
    *,
    brief_layer: str = DEFAULT_BRIEF_LAYER,
) -> tuple[Matchup, ...]:
    """Pair the typed store against every competitor on each shared cell.

    Groups the session records by ``(backend model, task)`` cell and, within each
    cell, forms one matchup between the ``brief_layer`` session and every other
    context layer present (Section ``sec:tokecon``, ``tab:tok_context_winrate``).
    A full sweep of ``M`` backend models, ``T`` tasks and ``C`` competing layers
    (each with exactly one session per cell) yields ``M * T * C`` matchups -- the
    ``12 * 30 * 10 = 3600`` of the paper.

    Matchups are returned in a deterministic order: cells sorted by
    ``(backend_model, task_id)``, and within a cell competitors sorted by name, so
    the output is reproducible regardless of input ordering.

    Parameters
    ----------
    sessions
        Session records. Each ``(backend model, task, context layer)`` triple must
        appear at most once.
    brief_layer
        The context-layer name treated as the typed store under test; every other
        layer in a cell becomes a competitor.

    Returns
    -------
    tuple of Matchup
        One matchup per ``(cell, competitor)`` pair, deterministically ordered.

    Raises
    ------
    ValueError
        If a ``(backend model, task, context layer)`` triple is duplicated, or if
        a cell contains competitor sessions but no ``brief_layer`` session (a
        missing typed-store baseline is a data error, not a silent skip).
    """
    brief_by_cell: dict[tuple[str, str], SessionRecord] = {}
    competitors_by_cell: dict[tuple[str, str], dict[str, SessionRecord]] = defaultdict(dict)

    for session in sessions:
        cell = (session.backend_model, session.task_id)
        if session.context_layer == brief_layer:
            if cell in brief_by_cell:
                msg = f"duplicate {brief_layer!r} session for cell {cell}"
                raise ValueError(msg)
            brief_by_cell[cell] = session
        else:
            layer_map = competitors_by_cell[cell]
            if session.context_layer in layer_map:
                msg = f"duplicate {session.context_layer!r} session for cell {cell}"
                raise ValueError(msg)
            layer_map[session.context_layer] = session

    matchups: list[Matchup] = []
    for cell in sorted(competitors_by_cell):
        brief_session = brief_by_cell.get(cell)
        if brief_session is None:
            msg = f"cell {cell} has competitor sessions but no {brief_layer!r} session"
            raise ValueError(msg)
        for competitor in sorted(competitors_by_cell[cell]):
            competitor_session = competitors_by_cell[cell][competitor]
            matchups.append(
                Matchup(
                    backend_model=cell[0],
                    task_id=cell[1],
                    competitor=competitor,
                    brief_tokens=brief_session.session_tokens,
                    competitor_tokens=competitor_session.session_tokens,
                )
            )
    return tuple(matchups)


@dataclass(frozen=True, slots=True)
class WinRate:
    """Aggregated outcome of a set of session-token matchups.

    The win rate is the fraction of matchups in which the typed store spent fewer
    session tokens (``tab:tok_context_winrate``: ``2880 / 3600 = 0.800``). Ties are
    counted separately and remain in the denominator, so ``rate`` reads as a share
    of *all* matchups; because the paper reports 0 ties, ``rate`` also equals
    ``wins / (wins + losses)`` there.

    Parameters
    ----------
    wins
        Matchups the typed store won (strictly fewer session tokens).
    losses
        Matchups the competitor won.
    ties
        Matchups with equal session-token spend.
    """

    wins: int
    losses: int
    ties: int

    @property
    def total(self) -> int:
        """Total matchups, ``wins + losses + ties``."""
        return self.wins + self.losses + self.ties

    @property
    def rate(self) -> float:
        """Share of all matchups the typed store won, in ``[0, 1]``.

        Raises
        ------
        ValueError
            If there are no matchups (an undefined rate).
        """
        if self.total == 0:
            msg = "win rate is undefined for zero matchups"
            raise ValueError(msg)
        return self.wins / self.total

    @property
    def loss_rate(self) -> float:
        """Share of all matchups the competitor won, in ``[0, 1]``."""
        if self.total == 0:
            msg = "loss rate is undefined for zero matchups"
            raise ValueError(msg)
        return self.losses / self.total


def tally(matchups: Iterable[Matchup]) -> WinRate:
    """Count wins, losses and ties over a set of matchups.

    Implements the aggregate of ``tab:tok_context_winrate``: over the sweep's
    matchups, how often the typed store is the cheaper session.

    Parameters
    ----------
    matchups
        The matchups to aggregate.

    Returns
    -------
    WinRate
        The win/loss/tie counts.
    """
    wins = losses = ties = 0
    for matchup in matchups:
        outcome = matchup.outcome
        if outcome is MatchupOutcome.BRIEF_CHEAPER:
            wins += 1
        elif outcome is MatchupOutcome.COMPETITOR_CHEAPER:
            losses += 1
        else:
            ties += 1
    return WinRate(wins=wins, losses=losses, ties=ties)


def tally_by_competitor(matchups: Iterable[Matchup]) -> dict[str, WinRate]:
    """Break the win rate out per competing context layer.

    Implements ``tab:tok_context_by_competitor`` / ``fig:tokwinrate`` (F105): the
    same matchups of :func:`tally`, resolved separately for each competitor
    (``360`` matchups each in the full sweep), yielding the ``68.6%``--``86.7%``
    band.

    Parameters
    ----------
    matchups
        The matchups to aggregate.

    Returns
    -------
    dict of str to WinRate
        One :class:`WinRate` per competitor, keys sorted by competitor name.
    """
    wins: dict[str, int] = defaultdict(int)
    losses: dict[str, int] = defaultdict(int)
    ties: dict[str, int] = defaultdict(int)
    for matchup in matchups:
        outcome = matchup.outcome
        if outcome is MatchupOutcome.BRIEF_CHEAPER:
            wins[matchup.competitor] += 1
        elif outcome is MatchupOutcome.COMPETITOR_CHEAPER:
            losses[matchup.competitor] += 1
        else:
            ties[matchup.competitor] += 1
    competitors = sorted(wins.keys() | losses.keys() | ties.keys())
    return {
        competitor: WinRate(
            wins=wins[competitor],
            losses=losses[competitor],
            ties=ties[competitor],
        )
        for competitor in competitors
    }


def tokens_per_resolved_point(resolution_rate: float, mean_session_tokens: float) -> float:
    """Session tokens spent per percentage-point of resolution.

    Column ``Tok./res. pt`` of ``tab:tok_agent_economics``: the mean session
    tokens divided by the resolution rate expressed as a percentage. A "resolved
    point" is one percentage point of resolution, so for ``Brief`` (``48.0%``
    resolution at ``12{,}400`` tokens) the cost is ``12400 / 48.0 = 258.3`` tokens
    per resolved point.

    Parameters
    ----------
    resolution_rate
        Fraction of sessions that resolved, in ``(0, 1]``.
    mean_session_tokens
        Mean session tokens spent by the layer. Non-negative.

    Returns
    -------
    float
        Tokens per percentage-point of resolution.

    Raises
    ------
    ValueError
        If ``resolution_rate`` is not in ``(0, 1]`` (a layer that never resolves
        has an undefined cost per resolved point), or ``mean_session_tokens`` is
        negative.
    """
    if not 0.0 < resolution_rate <= 1.0:
        msg = f"resolution_rate must be in (0, 1], got {resolution_rate}"
        raise ValueError(msg)
    if mean_session_tokens < 0.0:
        msg = f"mean_session_tokens must be non-negative, got {mean_session_tokens}"
        raise ValueError(msg)
    resolution_percent = resolution_rate * 100.0
    return mean_session_tokens / resolution_percent


def efficiency_score(
    resolution_rate: float,
    mean_session_tokens: float,
    *,
    scale: float = EFFICIENCY_SCALE,
) -> float:
    """Composite quality-per-token efficiency score.

    Column ``Efficiency score`` of ``tab:tok_agent_economics``: resolution quality
    delivered per session token, scaled for readability. Defined as
    ``scale * resolution_rate / mean_session_tokens``. With the default
    :data:`EFFICIENCY_SCALE` this reproduces the paper's column, e.g. ``Brief``
    (``48.0%`` at ``12{,}400`` tokens) scores ``1e7 * 0.48 / 12400 = 387.10``
    (published ``387.1``), and it is the reciprocal companion of
    :func:`tokens_per_resolved_point`: their product is the constant
    ``scale / 100``.

    Parameters
    ----------
    resolution_rate
        Fraction of sessions that resolved, in ``[0, 1]``.
    mean_session_tokens
        Mean session tokens spent by the layer. Strictly positive.
    scale
        Readability multiplier; the default reproduces the published column.

    Returns
    -------
    float
        The efficiency score (``0.0`` when nothing resolves).

    Raises
    ------
    ValueError
        If ``resolution_rate`` is outside ``[0, 1]`` or ``mean_session_tokens`` is
        not strictly positive.
    """
    if not 0.0 <= resolution_rate <= 1.0:
        msg = f"resolution_rate must be in [0, 1], got {resolution_rate}"
        raise ValueError(msg)
    if mean_session_tokens <= 0.0:
        msg = f"mean_session_tokens must be strictly positive, got {mean_session_tokens}"
        raise ValueError(msg)
    return scale * resolution_rate / mean_session_tokens


@dataclass(frozen=True, slots=True)
class SystemEconomics:
    """Quality-vs-cost summary for one context layer (a row of tab:tok_agent_economics).

    Parameters
    ----------
    system
        The context-layer name.
    n_sessions
        Number of sessions aggregated.
    resolution_rate
        Fraction of the layer's sessions that resolved, in ``[0, 1]``.
    mean_session_tokens
        Mean session tokens spent across the layer's sessions.
    tokens_per_resolved_point
        Session tokens per percentage-point of resolution
        (``None`` when the layer never resolves, so the ratio is undefined).
    efficiency_score
        Composite quality-per-token score (``0.0`` when nothing resolves).
    """

    system: str
    n_sessions: int
    resolution_rate: float
    mean_session_tokens: float
    tokens_per_resolved_point: float | None
    efficiency_score: float


def economics_by_system(
    sessions: Iterable[SessionRecord],
    *,
    scale: float = EFFICIENCY_SCALE,
) -> dict[str, SystemEconomics]:
    """Aggregate per-layer resolution, cost and efficiency (``tab:tok_agent_economics``).

    Groups the session records by context layer and, for each, computes the
    resolution rate, mean session tokens, tokens per resolved point, and the
    composite efficiency score -- the quality-vs-cost row the paper reports for
    every layer in the sweep.

    Parameters
    ----------
    sessions
        Session records to aggregate.
    scale
        Passed through to :func:`efficiency_score`.

    Returns
    -------
    dict of str to SystemEconomics
        One summary per layer, keys sorted by layer name.
    """
    tokens_by_layer: dict[str, list[int]] = defaultdict(list)
    resolved_by_layer: dict[str, int] = defaultdict(int)
    for session in sessions:
        tokens_by_layer[session.context_layer].append(session.session_tokens)
        if session.resolved:
            resolved_by_layer[session.context_layer] += 1

    summaries: dict[str, SystemEconomics] = {}
    for system in sorted(tokens_by_layer):
        token_counts = tokens_by_layer[system]
        n_sessions = len(token_counts)
        resolution_rate = resolved_by_layer[system] / n_sessions
        mean_tokens = statistics.fmean(token_counts)
        cost_per_point = (
            tokens_per_resolved_point(resolution_rate, mean_tokens)
            if resolution_rate > 0.0
            else None
        )
        summaries[system] = SystemEconomics(
            system=system,
            n_sessions=n_sessions,
            resolution_rate=resolution_rate,
            mean_session_tokens=mean_tokens,
            tokens_per_resolved_point=cost_per_point,
            efficiency_score=efficiency_score(resolution_rate, mean_tokens, scale=scale),
        )
    return summaries


def matchup_count(n_backend_models: int, n_tasks: int, n_competitors: int) -> int:
    """Return the number of matchups for a fully-crossed sweep.

    The sweep is fully crossed, so the matchup count is the product of its three
    dimensions (``tab:tok_context_winrate``: ``12 * 30 * 10 = 3600``).

    Parameters
    ----------
    n_backend_models
        Number of backend LLMs (12 in the paper's sweep).
    n_tasks
        Number of SWE-agent tasks (30).
    n_competitors
        Number of competing context layers compared against the typed store (10).

    Returns
    -------
    int
        ``n_backend_models * n_tasks * n_competitors``.

    Raises
    ------
    ValueError
        If any dimension is negative.
    """
    if n_backend_models < 0 or n_tasks < 0 or n_competitors < 0:
        msg = "sweep dimensions must be non-negative"
        raise ValueError(msg)
    return n_backend_models * n_tasks * n_competitors
