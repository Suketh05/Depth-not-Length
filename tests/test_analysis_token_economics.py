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

import csv
from fractions import Fraction
from pathlib import Path

import pytest

from membench.analysis.token_economics import (
    DEFAULT_BRIEF_LAYER,
    EFFICIENCY_SCALE,
    Matchup,
    MatchupOutcome,
    SessionRecord,
    SystemEconomics,
    WinRate,
    build_matchups,
    economics_by_system,
    efficiency_score,
    matchup_count,
    tally,
    tally_by_competitor,
    tokens_per_resolved_point,
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


# Fixed token amounts: Brief is constant, competitors are set strictly above (a
# Brief win) or strictly below (a Brief loss) so every cell is decisive (0 ties).
_BRIEF_TOKENS = 1000
_COMPETITOR_WIN_TOKENS = 2000  # competitor pricier -> Brief cheaper
_COMPETITOR_LOSS_TOKENS = 500  # competitor cheaper -> Brief loses


def _full_sweep_sessions() -> list[SessionRecord]:
    """Reconstruct the paper's 3600-matchup sweep from session records.

    12 backend models x 30 tasks = 360 cells. For each competitor, its published
    Brief-win count (COMPETITOR_WINS) determines how many cells Brief wins: the
    first ``wins`` cells get a pricier competitor, the rest a cheaper one. Brief
    spends a constant amount everywhere, so a cell is a Brief win iff the
    competitor priced above it.
    """
    models = [f"llm-{i:02d}" for i in range(N_BACKEND_MODELS)]
    tasks = [f"swe-{j:03d}" for j in range(N_TASKS)]
    cells = [(model, task) for model in models for task in tasks]
    assert len(cells) == MATCHUPS_PER_COMPETITOR

    sessions = [_session(model, task, "brief", _BRIEF_TOKENS) for model, task in cells]
    for competitor, wins in COMPETITOR_WINS.items():
        for index, (model, task) in enumerate(cells):
            tokens = _COMPETITOR_WIN_TOKENS if index < wins else _COMPETITOR_LOSS_TOKENS
            sessions.append(_session(model, task, competitor, tokens))
    return sessions


class TestFullSweepTournament:
    """Reconstruct tab:tok_context_winrate and tab:tok_context_by_competitor."""

    def test_sweep_size_is_3600(self) -> None:
        assert matchup_count(N_BACKEND_MODELS, N_TASKS, N_COMPETITORS) == TOTAL_MATCHUPS
        assert N_BACKEND_MODELS * N_TASKS * N_COMPETITORS == 3600
        assert build_matchups(_full_sweep_sessions()) != ()
        assert len(build_matchups(_full_sweep_sessions())) == TOTAL_MATCHUPS

    def test_overall_win_rate_is_2880_of_3600(self) -> None:
        result = tally(build_matchups(_full_sweep_sessions()))
        assert result.wins == BRIEF_WINS  # 2880
        assert result.losses == BRIEF_LOSSES  # 720
        assert result.ties == 0
        assert result.total == TOTAL_MATCHUPS
        # 2880 / 3600 = 0.800 exactly (= 4/5).
        assert Fraction(result.wins, result.total) == Fraction(4, 5)
        assert result.rate == 0.8
        assert round(result.rate * 100, 1) == 80.0

    def test_zero_ties_every_cell_decisive(self) -> None:
        matchups = build_matchups(_full_sweep_sessions())
        assert all(m.outcome is not MatchupOutcome.TIE for m in matchups)

    def test_per_competitor_matches_published_table(self) -> None:
        by_competitor = tally_by_competitor(build_matchups(_full_sweep_sessions()))
        assert set(by_competitor) == set(COMPETITOR_WINS)
        for competitor, win_rate in by_competitor.items():
            assert win_rate.total == MATCHUPS_PER_COMPETITOR  # 360
            assert win_rate.ties == 0
            assert win_rate.wins == COMPETITOR_WINS[competitor]
            assert round(win_rate.rate * 100, 1) == COMPETITOR_WIN_PCT[competitor]
        # Per-competitor wins sum to the overall 2880.
        assert sum(wr.wins for wr in by_competitor.values()) == BRIEF_WINS

    def test_win_rate_band_68_6_to_86_7(self) -> None:
        by_competitor = tally_by_competitor(build_matchups(_full_sweep_sessions()))
        pct = {c: round(wr.rate * 100, 1) for c, wr in by_competitor.items()}
        assert min(pct.values()) == 68.6  # Oiya
        assert max(pct.values()) == 86.7  # Oracle Summary
        assert pct["oiya"] == 68.6
        assert pct["oracle_summary"] == 86.7


def _load_paper_csv() -> dict[str, float]:
    """Read the committed metric->value rows of paper_token_economics.csv."""
    with _PAPER_CSV.open(newline="") as handle:
        return {row["metric"]: float(row["value"]) for row in csv.DictReader(handle)}


class TestPaperCsvIdentity:
    """The cross-check a reader runs directly against the committed CSV.

    ``results/data/paper_token_economics.csv`` carries the published aggregates of
    tab:tok_context_winrate / tab:tok_context_by_competitor. These identities show
    the numbers are internally consistent and that this module's arithmetic agrees.
    """

    def test_headline_win_rate_identity(self) -> None:
        data = _load_paper_csv()
        assert data["matchups_total"] == TOTAL_MATCHUPS
        assert data["matchups_won"] == BRIEF_WINS
        assert data["matchups_lost"] == BRIEF_LOSSES
        assert data["matchups_tied"] == 0
        # won + lost + tied == total.
        assert (
            data["matchups_won"] + data["matchups_lost"] + data["matchups_tied"]
            == (data["matchups_total"])
        )
        # 2880 / 3600 = 0.800 == published percentage.
        assert data["matchups_won"] / data["matchups_total"] == 0.8
        assert data["session_token_win_rate_pct"] == 80.0

    def test_per_competitor_reconstructs_2880(self) -> None:
        data = _load_paper_csv()
        # Map CSV keys (win_rate_vs_<layer>_pct) back to competitor names.
        published_pct = {
            key[len("win_rate_vs_") : -len("_pct")]: value
            for key, value in data.items()
            if key.startswith("win_rate_vs_") and key.endswith("_pct")
        }
        assert set(published_pct) == set(COMPETITOR_WIN_PCT)
        assert published_pct == COMPETITOR_WIN_PCT
        # Reconstruct wins = round(pct/100 * 360); the ten rows sum to 2880.
        reconstructed = {
            competitor: round(pct / 100.0 * MATCHUPS_PER_COMPETITOR)
            for competitor, pct in published_pct.items()
        }
        assert reconstructed == COMPETITOR_WINS
        assert sum(reconstructed.values()) == BRIEF_WINS
        assert sum(reconstructed.values()) / TOTAL_MATCHUPS == 0.8

    def test_win_rate_band(self) -> None:
        data = _load_paper_csv()
        pct_values = [
            value
            for key, value in data.items()
            if key.startswith("win_rate_vs_") and key.endswith("_pct")
        ]
        assert min(pct_values) == 68.6
        assert max(pct_values) == 86.7


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


class TestEfficiencyGoldens:
    """Reproduce every row of tab:tok_agent_economics from published inputs."""

    @pytest.mark.parametrize(
        ("system", "resolution_pct", "session_tokens", "cost_per_point", "efficiency"),
        ECONOMICS_ROWS,
        ids=[row[0] for row in ECONOMICS_ROWS],
    )
    def test_row_reproduced(
        self,
        system: str,
        resolution_pct: float,
        session_tokens: int,
        cost_per_point: float,
        efficiency: float,
    ) -> None:
        rate = resolution_pct / 100.0
        # Published values are 1--2 dp roundings of these ratios (abs 0.05).
        assert tokens_per_resolved_point(rate, session_tokens) == pytest.approx(
            cost_per_point, abs=0.05
        )
        assert efficiency_score(rate, session_tokens) == pytest.approx(efficiency, abs=0.05)

    def test_brief_efficiency_full_precision(self) -> None:
        # 1e7 * 0.48 / 12400 = 387.0967... -> published 387.1.
        assert efficiency_score(0.48, 12400) == 1e7 * 0.48 / 12400
        assert round(efficiency_score(0.48, 12400), 1) == 387.1
        assert tokens_per_resolved_point(0.48, 12400) == 12400 / 48.0
        assert round(tokens_per_resolved_point(0.48, 12400), 1) == 258.3

    @pytest.mark.parametrize(
        ("resolution_pct", "session_tokens"),
        [(row[1], row[2]) for row in ECONOMICS_ROWS],
        ids=[row[0] for row in ECONOMICS_ROWS],
    )
    def test_reciprocal_identity(self, resolution_pct: float, session_tokens: int) -> None:
        # efficiency_score * tokens_per_resolved_point == EFFICIENCY_SCALE / 100.
        rate = resolution_pct / 100.0
        product = efficiency_score(rate, session_tokens) * tokens_per_resolved_point(
            rate, session_tokens
        )
        assert product == pytest.approx(EFFICIENCY_SCALE / 100.0, rel=1e-12)
        assert EFFICIENCY_SCALE / 100.0 == 1e5

    def test_scale_default_and_override(self) -> None:
        assert EFFICIENCY_SCALE == 1e7
        # Doubling the scale doubles the score.
        base = efficiency_score(0.5, 1000)
        assert efficiency_score(0.5, 1000, scale=2e7) == pytest.approx(2 * base)

    def test_chance_floor_and_saturation(self) -> None:
        # A layer that never resolves scores 0 quality per token.
        assert efficiency_score(0.0, 1000) == 0.0
        # Full resolution at minimal spend maximises the score.
        assert efficiency_score(1.0, 1) > efficiency_score(0.5, 1)

    def test_domain_guards(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            efficiency_score(1.5, 1000)
        with pytest.raises(ValueError, match="strictly positive"):
            efficiency_score(0.5, 0)
        # tokens_per_resolved_point is undefined for a never-resolving layer.
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            tokens_per_resolved_point(0.0, 1000)
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            tokens_per_resolved_point(1.5, 1000)


class TestMatchupCount:
    def test_full_sweep(self) -> None:
        assert matchup_count(12, 30, 10) == 3600

    def test_zero_dimension(self) -> None:
        assert matchup_count(0, 30, 10) == 0

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            matchup_count(-1, 30, 10)


class TestEconomicsBySystem:
    """Aggregate session records into tab:tok_agent_economics rows."""

    def test_reconstructs_brief_row(self) -> None:
        # 25 sessions, 12 resolved -> 48.0% resolution; every session spends 12400
        # tokens -> the published Brief row (258.3 tok/pt, efficiency 387.1).
        sessions = [
            _session("gpt-4o", f"swe-{i:03d}", "brief", 12400, resolved=i < 12) for i in range(25)
        ]
        summaries = economics_by_system(sessions)
        assert list(summaries) == ["brief"]
        brief = summaries["brief"]
        assert isinstance(brief, SystemEconomics)
        assert brief.n_sessions == 25
        assert brief.resolution_rate == 0.48
        assert brief.mean_session_tokens == 12400.0
        assert brief.tokens_per_resolved_point == pytest.approx(258.3, abs=0.05)
        assert brief.efficiency_score == pytest.approx(387.1, abs=0.05)
        # The aggregated row agrees with the standalone scalar functions.
        assert brief.efficiency_score == efficiency_score(0.48, 12400.0)
        assert brief.tokens_per_resolved_point == tokens_per_resolved_point(0.48, 12400.0)

    def test_never_resolved_layer_has_no_cost_per_point(self) -> None:
        sessions = [_session("m", f"t{i}", "raptor", 48000, resolved=False) for i in range(4)]
        summaries = economics_by_system(sessions)
        raptor = summaries["raptor"]
        assert raptor.resolution_rate == 0.0
        assert raptor.tokens_per_resolved_point is None  # undefined, not a crash
        assert raptor.efficiency_score == 0.0

    def test_groups_multiple_layers(self) -> None:
        sessions = [
            _session("m", "t1", "brief", 1000, resolved=True),
            _session("m", "t2", "brief", 3000, resolved=False),
            _session("m", "t1", "mem0", 5000, resolved=True),
        ]
        summaries = economics_by_system(sessions)
        assert list(summaries) == ["brief", "mem0"]  # sorted keys
        assert summaries["brief"].mean_session_tokens == 2000.0
        assert summaries["brief"].resolution_rate == 0.5
        assert summaries["mem0"].n_sessions == 1
        assert summaries["mem0"].resolution_rate == 1.0
