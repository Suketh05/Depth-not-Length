"""Tests for the token-economics tournament (paper sec:tokecon).

Golden values are transcribed verbatim from the paper and its committed data:

* ``tab:tok_context_winrate`` -- 3600 matchups, 2880 wins (80.0%), 720 losses, 0 ties.
* ``tab:tok_context_by_competitor`` -- per-competitor wins/win-rate (68.6%--86.7%).
* ``tab:tok_agent_economics`` -- resolution %, session tokens, tokens per resolved
  point, and the composite efficiency score for all 16 layers.

The per-competitor win counts sum to exactly 2880, and each equals
``round(published_pct/100 * 360)`` -- the identity a reader recomputes from
``results/data/paper_token_economics.csv``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from membench.analysis.token_economics import (
    DEFAULT_BRIEF_LAYER,
    Matchup,
    MatchupOutcome,
    SessionRecord,
    WinRate,
    build_matchups,
    tally,
    tally_by_competitor,
)

# --- Paper goldens: tab:tok_context_winrate ---------------------------------
# "Brief is cheaper in 80.0% of matchups (2880 of 3600)"; 0 ties.
N_BACKEND_MODELS = 12
N_TASKS = 30
N_COMPETITORS = 10
TOTAL_MATCHUPS = 3600
BRIEF_WINS = 2880
BRIEF_LOSSES = 720
MATCHUPS_PER_COMPETITOR = 360  # 12 LLMs x 30 tasks

# --- Paper goldens: tab:tok_context_by_competitor ---------------------------
# Per-competitor Brief win counts (out of 360) and published win %; these are the
# verbatim table rows. The win counts sum to BRIEF_WINS = 2880.
COMPETITOR_WINS: dict[str, int] = {
    "mem0": 291,
    "zep": 279,
    "contextq": 282,
    "oracle_summary": 312,
    "supermemory": 288,
    "unabyss": 303,
    "driver": 296,
    "oiya": 247,
    "kluris": 302,
    "none": 280,
}
COMPETITOR_WIN_PCT: dict[str, float] = {
    "mem0": 80.8,
    "zep": 77.5,
    "contextq": 78.3,
    "oracle_summary": 86.7,
    "supermemory": 80.0,
    "unabyss": 84.2,
    "driver": 82.2,
    "oiya": 68.6,
    "kluris": 83.9,
    "none": 77.8,
}

# --- Paper goldens: tab:tok_agent_economics ---------------------------------
# (system, resolution %, session tokens, tokens/resolved-pt, efficiency score).
# Transcribed verbatim; efficiency = 1e5 * resolution% / tokens (= EFFICIENCY_SCALE
# * rate / tokens), tokens/res-pt = tokens / resolution%.
ECONOMICS_ROWS: tuple[tuple[str, float, int, float, float], ...] = (
    ("Brief", 48.0, 12400, 258.3, 387.1),
    ("Mem0", 24.2, 45235, 1869.2, 53.5),
    ("Zep", 32.6, 45942, 1409.3, 70.96),
    ("Supermemory", 30.3, 46342, 1529.4, 65.38),
    ("MemGPT", 30.3, 45541, 1503.0, 66.53),
    ("GraphRAG", 28.1, 48382, 1721.8, 58.08),
    ("A-Mem", 32.6, 45250, 1388.0, 72.04),
    ("RAPTOR", 26.6, 48167, 1810.8, 55.22),
    ("ContextQ", 34.0, 45049, 1325.0, 75.47),
    ("Unabyss", 29.0, 47768, 1647.2, 60.71),
    ("ctx|", 31.1, 46290, 1488.4, 67.19),
    ("Driver", 31.7, 46512, 1467.3, 68.15),
    ("Oiya", 27.3, 50298, 1842.4, 54.28),
    ("Kluris", 23.6, 51289, 2173.3, 46.01),
    ("Oracle Summary", 34.3, 44767, 1305.2, 76.62),
    ("OpenViking", 29.2, 47972, 1642.9, 60.87),
)

_PAPER_CSV = Path(__file__).resolve().parents[1] / "results" / "data" / "paper_token_economics.csv"


def _session(
    model: str, task: str, layer: str, tokens: int, *, resolved: bool = True
) -> SessionRecord:
    return SessionRecord(
        backend_model=model,
        task_id=task,
        context_layer=layer,
        session_tokens=tokens,
        resolved=resolved,
    )


def _keys(matchups: tuple[Matchup, ...]) -> list[tuple[str, str, str]]:
    return [(m.backend_model, m.task_id, m.competitor) for m in matchups]


class TestSessionRecord:
    def test_totals_and_fields(self) -> None:
        rec = _session("gpt-5.5", "swe-001", "brief", 1835, resolved=True)
        assert rec.backend_model == "gpt-5.5"
        assert rec.task_id == "swe-001"
        assert rec.context_layer == "brief"
        assert rec.session_tokens == 1835
        assert rec.resolved is True

    def test_negative_tokens_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _session("m", "t", "brief", -1)

    def test_frozen(self) -> None:
        rec = _session("m", "t", "brief", 100)
        with pytest.raises(AttributeError):
            rec.session_tokens = 200  # type: ignore[misc]


class TestMatchupOutcome:
    def test_brief_cheaper(self) -> None:
        matchup = Matchup("m", "t", "mem0", brief_tokens=1835, competitor_tokens=4737)
        assert matchup.outcome is MatchupOutcome.BRIEF_CHEAPER
        assert matchup.brief_wins is True

    def test_competitor_cheaper(self) -> None:
        # tab:tok_context_brief_losses: an honest loss (competitor spends fewer).
        matchup = Matchup("m", "t", "mem0", brief_tokens=2428, competitor_tokens=1995)
        assert matchup.outcome is MatchupOutcome.COMPETITOR_CHEAPER
        assert matchup.brief_wins is False

    def test_tie_is_strict_equality(self) -> None:
        matchup = Matchup("m", "t", "zep", brief_tokens=2000, competitor_tokens=2000)
        assert matchup.outcome is MatchupOutcome.TIE
        assert matchup.brief_wins is False

    def test_default_brief_layer_constant(self) -> None:
        assert DEFAULT_BRIEF_LAYER == "brief"


class TestBuildMatchups:
    def test_one_cell_pairs_brief_against_each_competitor(self) -> None:
        sessions = [
            _session("gpt-5.5", "swe-001", "brief", 1835),
            _session("gpt-5.5", "swe-001", "mem0", 4737),
            _session("gpt-5.5", "swe-001", "zep", 2724),
        ]
        matchups = build_matchups(sessions)
        assert len(matchups) == 2
        # Deterministic: competitors sorted by name (mem0 before zep).
        assert [m.competitor for m in matchups] == ["mem0", "zep"]
        assert all(m.brief_tokens == 1835 for m in matchups)
        assert [m.competitor_tokens for m in matchups] == [4737, 2724]
        assert all(m.brief_wins for m in matchups)

    def test_order_independent_of_input_order(self) -> None:
        forward = [
            _session("a", "t1", "brief", 100),
            _session("a", "t1", "zep", 200),
            _session("b", "t1", "brief", 100),
            _session("b", "t1", "mem0", 50),
        ]
        reversed_input = list(reversed(forward))
        assert _keys(build_matchups(forward)) == _keys(build_matchups(reversed_input))
        # Sorted by (model, task) then competitor.
        assert _keys(build_matchups(forward)) == [
            ("a", "t1", "zep"),
            ("b", "t1", "mem0"),
        ]

    def test_custom_brief_layer_name(self) -> None:
        sessions = [
            _session("m", "t", "typed_graph", 100),
            _session("m", "t", "mem0", 200),
        ]
        matchups = build_matchups(sessions, brief_layer="typed_graph")
        assert len(matchups) == 1
        assert matchups[0].brief_tokens == 100
        assert matchups[0].competitor == "mem0"

    def test_empty_input_yields_no_matchups(self) -> None:
        assert build_matchups([]) == ()

    def test_duplicate_brief_session_rejected(self) -> None:
        sessions = [
            _session("m", "t", "brief", 100),
            _session("m", "t", "brief", 101),
            _session("m", "t", "mem0", 200),
        ]
        with pytest.raises(ValueError, match="duplicate 'brief'"):
            build_matchups(sessions)

    def test_duplicate_competitor_session_rejected(self) -> None:
        sessions = [
            _session("m", "t", "brief", 100),
            _session("m", "t", "mem0", 200),
            _session("m", "t", "mem0", 201),
        ]
        with pytest.raises(ValueError, match="duplicate 'mem0'"):
            build_matchups(sessions)

    def test_missing_brief_baseline_rejected(self) -> None:
        # A cell with a competitor but no brief session is a data error, not a skip.
        sessions = [_session("m", "t", "mem0", 200)]
        with pytest.raises(ValueError, match="no 'brief' session"):
            build_matchups(sessions)


class TestTally:
    def test_counts_and_rate(self) -> None:
        matchups = [
            Matchup("m", "t", "a", brief_tokens=1, competitor_tokens=2),  # win
            Matchup("m", "t", "b", brief_tokens=2, competitor_tokens=1),  # loss
            Matchup("m", "t", "c", brief_tokens=3, competitor_tokens=3),  # tie
            Matchup("m", "t", "d", brief_tokens=1, competitor_tokens=5),  # win
        ]
        result = tally(matchups)
        assert result == WinRate(wins=2, losses=1, ties=1)
        assert result.total == 4
        assert result.rate == 0.5  # ties stay in the denominator
        assert result.loss_rate == 0.25

    def test_empty_rate_is_undefined(self) -> None:
        result = tally([])
        assert result.total == 0
        with pytest.raises(ValueError, match="undefined"):
            _ = result.rate
        with pytest.raises(ValueError, match="undefined"):
            _ = result.loss_rate

    def test_by_competitor_partitions(self) -> None:
        matchups = [
            Matchup("m", "t1", "mem0", brief_tokens=1, competitor_tokens=2),  # win
            Matchup("m", "t2", "mem0", brief_tokens=2, competitor_tokens=1),  # loss
            Matchup("m", "t1", "zep", brief_tokens=1, competitor_tokens=2),  # win
        ]
        by_competitor = tally_by_competitor(matchups)
        assert list(by_competitor) == ["mem0", "zep"]  # sorted keys
        assert by_competitor["mem0"] == WinRate(wins=1, losses=1, ties=0)
        assert by_competitor["zep"] == WinRate(wins=1, losses=0, ties=0)
        # Per-competitor totals reconstitute the overall tally.
        overall = tally(matchups)
        assert sum(wr.wins for wr in by_competitor.values()) == overall.wins
        assert sum(wr.total for wr in by_competitor.values()) == overall.total
