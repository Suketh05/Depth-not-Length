"""Tests for the token-economics tournament (paper sec:tokecon).

Goldens come from two independent sources, never from running the module:

* Hand-built matchup sets whose win rates are exact fractions computed in the
  test itself (3/4, 1/2, ...).
* The paper's published numbers, quoted verbatim with their table anchors:
  tab:tok_context_winrate (2880/3600 = 80.0%, 0 ties),
  tab:tok_context_by_competitor (per-layer rates 68.6%..86.7% and win counts),
  tab:tok_agent_economics (efficiency score / tokens-per-resolved-point rows),
  tab:tok_agent_context_gpt55 (per-turn ledgers), and
  tab:tok_context_brief_losses (loss margins). The committed CSV
  results/data/paper_token_economics.csv is read directly, so these tests are
  the cross-check a reader runs against the paper.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import pytest

from membench.analysis.token_economics import (
    BRIEF_SYSTEM,
    CompetitorRecord,
    Matchup,
    MatchupOutcome,
    SessionRecord,
    TournamentSummary,
    decide_matchup,
    pair_matchups,
    run_tournament,
    session_tokens_from_turns,
    sweep_matchup_count,
    win_count_from_published_pct,
)

PAPER_CSV = Path(__file__).resolve().parents[1] / "results" / "data" / "paper_token_economics.csv"


def _brief(llm: str, task: str, tokens: int) -> SessionRecord:
    return SessionRecord(system=BRIEF_SYSTEM, llm=llm, task=task, session_tokens=tokens)


def _comp(system: str, llm: str, task: str, tokens: int) -> SessionRecord:
    return SessionRecord(system=system, llm=llm, task=task, session_tokens=tokens)


class TestSessionRecord:
    def test_accepts_valid_record(self) -> None:
        rec = SessionRecord(
            system="mem0",
            llm="gpt-5.5",
            task="swe-004",
            session_tokens=4737,
            resolved=True,
            wall_clock_minutes=27.8,
        )
        assert rec.session_tokens == 4737

    def test_rejects_non_positive_tokens(self) -> None:
        # Every session pays at least its turn-1 prompt.
        with pytest.raises(ValueError, match="positive"):
            SessionRecord(system="mem0", llm="m", task="t", session_tokens=0)
        with pytest.raises(ValueError, match="positive"):
            SessionRecord(system="mem0", llm="m", task="t", session_tokens=-5)

    def test_rejects_negative_wall_clock(self) -> None:
        with pytest.raises(ValueError, match="wall_clock"):
            SessionRecord(system="zep", llm="m", task="t", session_tokens=1, wall_clock_minutes=-1)


class TestDecideMatchup:
    """The strict session-cheaper rule of tab:tok_context_winrate."""

    def test_strictly_fewer_tokens_wins(self) -> None:
        assert decide_matchup(1835, 4737) is MatchupOutcome.BRIEF_WIN
        assert decide_matchup(4737, 1835) is MatchupOutcome.BRIEF_LOSS

    def test_equal_spend_is_a_tie(self) -> None:
        # Tie rule: neither side is *strictly* cheaper. The paper's sweep
        # produced 0 such cells ("There are no ties: on every cell one
        # configuration is strictly cheaper", sec:tokecon).
        assert decide_matchup(2000, 2000) is MatchupOutcome.TIE

    def test_one_token_margin_is_decisive(self) -> None:
        assert decide_matchup(1999, 2000) is MatchupOutcome.BRIEF_WIN
        assert decide_matchup(2001, 2000) is MatchupOutcome.BRIEF_LOSS

    def test_rejects_non_positive_counts(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            decide_matchup(0, 100)
        with pytest.raises(ValueError, match="positive"):
            decide_matchup(100, -1)

    def test_matchup_outcome_property_matches_rule(self) -> None:
        m = Matchup(llm="m", task="t", competitor="zep", brief_tokens=10, competitor_tokens=20)
        assert m.outcome is MatchupOutcome.BRIEF_WIN


class TestSessionTokensFromTurns:
    """Ledger sums quoted verbatim from tab:tok_agent_context_gpt55."""

    @pytest.mark.parametrize(
        ("turns", "session_total"),
        [
            # alone (unresolved): 1403 + 698 + 601 + 1461 = 4163
            ((1403, 698, 601, 1461), 4163),
            # alone (best case): 1420 + 715 + 612 + 939 = 3686
            ((1420, 715, 612, 939), 3686),
            # Brief context: 1598 + 203 + 34 = 1835 (session collapses by turn 3)
            ((1598, 203, 34), 1835),
            # Brief context (slow task): 1598 + 412 + 118 + 44 = 2172
            ((1598, 412, 118, 44), 2172),
            # Mem0 context: 1712 + 684 + 598 + 1743 = 4737
            ((1712, 684, 598, 1743), 4737),
            # Mem0 context (unresolved): 1708 + 691 + 605 + 2166 = 5170
            ((1708, 691, 605, 2166), 5170),
            # ContextQ context: 1648 + 612 + 541 + 926 = 3727
            ((1648, 612, 541, 926), 3727),
            # Zep context: 1576 + 448 + 382 + 318 = 2724
            ((1576, 448, 382, 318), 2724),
        ],
    )
    def test_paper_ledger_rows(self, turns: tuple[int, ...], session_total: int) -> None:
        assert session_tokens_from_turns(turns) == session_total

    def test_rejects_empty_ledger(self) -> None:
        with pytest.raises(ValueError, match="at least one turn"):
            session_tokens_from_turns(())

    def test_rejects_non_positive_turn(self) -> None:
        with pytest.raises(ValueError, match="turn 2"):
            session_tokens_from_turns((1598, 0, 34))


class TestSweepMatchupCount:
    def test_paper_sweep_is_3600(self) -> None:
        # tab:tok_context_winrate: 12 LLMs x 30 SWE agent tasks x 10 layers.
        assert sweep_matchup_count(12, 30, 10) == 3600

    def test_rejects_non_positive_dimension(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            sweep_matchup_count(12, 0, 10)


class TestPairMatchups:
    def test_pairs_every_competitor_session_with_its_cell(self) -> None:
        briefs = [_brief("m1", "t1", 100), _brief("m1", "t2", 200)]
        comps = [
            _comp("mem0", "m1", "t1", 150),
            _comp("zep", "m1", "t1", 90),
            _comp("mem0", "m1", "t2", 250),
        ]
        matchups = pair_matchups(briefs, comps)
        assert [(m.competitor, m.llm, m.task) for m in matchups] == [
            ("mem0", "m1", "t1"),
            ("zep", "m1", "t1"),
            ("mem0", "m1", "t2"),
        ]
        assert [(m.brief_tokens, m.competitor_tokens) for m in matchups] == [
            (100, 150),
            (100, 90),
            (200, 250),
        ]

    def test_empty_competitors_yield_empty_tournament(self) -> None:
        assert pair_matchups([_brief("m1", "t1", 100)], []) == []

    def test_missing_brief_cell_raises(self) -> None:
        with pytest.raises(ValueError, match="no Brief session"):
            pair_matchups([_brief("m1", "t1", 100)], [_comp("mem0", "m1", "t9", 150)])

    def test_duplicate_brief_cell_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate Brief"):
            pair_matchups([_brief("m1", "t1", 100), _brief("m1", "t1", 110)], [])

    def test_duplicate_competitor_session_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate competitor"):
            pair_matchups(
                [_brief("m1", "t1", 100)],
                [_comp("mem0", "m1", "t1", 150), _comp("mem0", "m1", "t1", 160)],
            )

    def test_brief_session_on_wrong_side_raises(self) -> None:
        with pytest.raises(ValueError, match="must not contain Brief"):
            pair_matchups([_brief("m1", "t1", 100)], [_brief("m1", "t1", 150)])
        with pytest.raises(ValueError, match="must have system"):
            pair_matchups([_comp("mem0", "m1", "t1", 100)], [])


class TestTournamentTallies:
    def test_exact_fraction_win_rate_three_quarters(self) -> None:
        # Hand-built: 3 wins, 1 loss => win rate exactly 3/4 = 0.75.
        briefs = [_brief("m1", f"t{i}", 100) for i in range(4)]
        comps = [
            _comp("mem0", "m1", "t0", 150),  # win
            _comp("mem0", "m1", "t1", 150),  # win
            _comp("mem0", "m1", "t2", 150),  # win
            _comp("mem0", "m1", "t3", 50),  # loss
        ]
        summary = run_tournament(briefs, comps)
        assert (summary.matchups, summary.wins, summary.losses, summary.ties) == (4, 3, 1, 0)
        assert summary.win_rate == 0.75  # exactly representable: 3/4
        assert summary.win_rate_pct == 75.0

    def test_per_competitor_split_with_exact_fractions(self) -> None:
        # mem0: 1 win / 2 => 1/2; zep: 2 wins / 2 => 1.0; overall 3/4.
        briefs = [_brief("m1", "t1", 100), _brief("m1", "t2", 100)]
        comps = [
            _comp("mem0", "m1", "t1", 300),  # win
            _comp("mem0", "m1", "t2", 80),  # loss
            _comp("zep", "m1", "t1", 101),  # win
            _comp("zep", "m1", "t2", 400),  # win
        ]
        summary = run_tournament(briefs, comps)
        assert summary.win_rate == 0.75
        mem0 = summary.per_competitor["mem0"]
        zep = summary.per_competitor["zep"]
        assert (mem0.matchups, mem0.wins, mem0.losses, mem0.ties) == (2, 1, 1, 0)
        assert mem0.win_rate == 0.5  # exactly 1/2
        assert (zep.matchups, zep.wins, zep.losses, zep.ties) == (2, 2, 0, 0)
        assert zep.win_rate == 1.0
        # Per-competitor tallies partition the overall tally.
        assert mem0.wins + zep.wins == summary.wins
        assert mem0.matchups + zep.matchups == summary.matchups

    def test_ties_counted_in_denominator_only(self) -> None:
        briefs = [_brief("m1", "t1", 100), _brief("m1", "t2", 100)]
        comps = [
            _comp("mem0", "m1", "t1", 100),  # tie (equal spend)
            _comp("mem0", "m1", "t2", 200),  # win
        ]
        summary = run_tournament(briefs, comps)
        assert (summary.matchups, summary.wins, summary.losses, summary.ties) == (2, 1, 0, 1)
        assert summary.win_rate == 0.5  # 1 win / 2 matchups; the tie dilutes

    def test_empty_tournament_has_undefined_rate(self) -> None:
        summary = run_tournament([], [])
        assert (summary.matchups, summary.wins, summary.losses, summary.ties) == (0, 0, 0, 0)
        assert summary.win_rate is None
        assert summary.win_rate_pct is None
        assert dict(summary.per_competitor) == {}

    def test_competitor_record_rejects_inconsistent_tally(self) -> None:
        with pytest.raises(ValueError, match="must equal matchups"):
            CompetitorRecord(competitor="mem0", matchups=360, wins=291, losses=68, ties=0)

    def test_seeded_synthetic_sweep_matches_independent_tally(self) -> None:
        # Deterministic 4 LLMs x 5 tasks x 3 competitors = 60-matchup sweep;
        # expected counts computed by an independent plain-loop tally over the
        # same generated tokens, never by the module under test.
        rng = random.Random(20260628)
        llms = [f"llm{i}" for i in range(4)]
        tasks = [f"task{j}" for j in range(5)]
        competitors = ["mem0", "zep", "oiya"]
        brief_tokens = {(m, t): rng.randint(1000, 3000) for m in llms for t in tasks}
        comp_tokens = {
            (c, m, t): rng.randint(1000, 3000) for c in competitors for m in llms for t in tasks
        }

        expected_wins = dict.fromkeys(competitors, 0)
        expected_losses = dict.fromkeys(competitors, 0)
        expected_ties = dict.fromkeys(competitors, 0)
        for (c, m, t), ct in comp_tokens.items():
            bt = brief_tokens[(m, t)]
            if bt < ct:
                expected_wins[c] += 1
            elif bt > ct:
                expected_losses[c] += 1
            else:
                expected_ties[c] += 1

        briefs = [_brief(m, t, tok) for (m, t), tok in brief_tokens.items()]
        comps = [_comp(c, m, t, tok) for (c, m, t), tok in comp_tokens.items()]
        summary = run_tournament(briefs, comps)

        assert summary.matchups == sweep_matchup_count(4, 5, 3) == 60
        assert summary.wins == sum(expected_wins.values())
        assert summary.losses == sum(expected_losses.values())
        assert summary.ties == sum(expected_ties.values())
        for c in competitors:
            rec = summary.per_competitor[c]
            assert rec.matchups == 20
            assert (rec.wins, rec.losses, rec.ties) == (
                expected_wins[c],
                expected_losses[c],
                expected_ties[c],
            )
            assert rec.win_rate == expected_wins[c] / 20

    def test_summary_reconstructs_paper_headline_from_published_counts(self) -> None:
        # Verbatim per-competitor (wins, losses) of tab:tok_context_by_competitor;
        # 360 matchups per layer, 0 ties.
        published = {
            "mem0": (291, 69),
            "zep": (279, 81),
            "contextq": (282, 78),
            "oracle_summary": (312, 48),
            "supermemory": (288, 72),
            "unabyss": (303, 57),
            "driver": (296, 64),
            "oiya": (247, 113),
            "kluris": (302, 58),
            "none": (280, 80),
        }
        summary = TournamentSummary(
            matchups=3600,
            wins=sum(w for w, _ in published.values()),
            losses=sum(losses for _, losses in published.values()),
            ties=0,
            per_competitor={
                name: CompetitorRecord(competitor=name, matchups=360, wins=w, losses=losses, ties=0)
                for name, (w, losses) in published.items()
            },
        )
        # tab:tok_context_winrate: 2880 wins, 720 losses, 80.0%.
        assert summary.wins == 2880
        assert summary.losses == 720
        assert summary.win_rate == 2880 / 3600 == 0.8
        # tab:tok_context_by_competitor: printed one-decimal percentages.
        pct = {name: summary.per_competitor[name].win_rate_pct for name in published}
        rounded = {name: round(p, 1) for name, p in pct.items() if p is not None}
        assert rounded == {
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


def _load_paper_csv() -> dict[str, float]:
    with PAPER_CSV.open(newline="") as fh:
        return {row["metric"]: float(row["value"]) for row in csv.DictReader(fh)}


class TestPaperCsvIdentities:
    """Recompute tab:tok_context_winrate / tab:tok_context_by_competitor from the committed CSV."""

    def test_headline_win_rate_is_2880_over_3600(self) -> None:
        vals = _load_paper_csv()
        assert vals["matchups_total"] == 3600
        assert vals["matchups_won"] == 2880
        assert vals["matchups_lost"] == 720
        assert vals["matchups_tied"] == 0  # paper: "There are no ties"
        # The identity a reader checks: 2880 / 3600 = 0.800 exactly.
        assert vals["matchups_won"] / vals["matchups_total"] == 0.800
        assert 100.0 * vals["matchups_won"] / vals["matchups_total"] == pytest.approx(
            vals["session_token_win_rate_pct"], abs=1e-12
        )
        # Wins + losses + ties partition the matchups.
        assert (
            vals["matchups_won"] + vals["matchups_lost"] + vals["matchups_tied"]
            == vals["matchups_total"]
        )
        # Sweep shape: 12 LLMs x 30 tasks x 10 context layers.
        assert sweep_matchup_count(12, 30, 10) == vals["matchups_total"]

    def test_per_competitor_band_is_68_6_to_86_7(self) -> None:
        vals = _load_paper_csv()
        per_comp = {
            k.removeprefix("win_rate_vs_").removesuffix("_pct"): v
            for k, v in vals.items()
            if k.startswith("win_rate_vs_")
        }
        # Ten layers, 360 matchups each => 10 * 360 = 3600.
        assert len(per_comp) == 10
        assert vals["matchups_total"] == 10 * 360
        # tab:tok_context_by_competitor band: lowest 68.6 (Oiya), highest 86.7
        # (Oracle Summary); fig:tokwinrate quotes the same 68.6%-86.7% span.
        assert min(per_comp.values()) == 68.6
        assert max(per_comp.values()) == 86.7
        assert min(per_comp, key=per_comp.__getitem__) == "oiya"
        assert max(per_comp, key=per_comp.__getitem__) == "oracle_summary"
        # All rates sit above the 50% chance line ("all bars sit well above").
        assert all(v > 50.0 for v in per_comp.values())

    def test_published_percentages_invert_to_counts_summing_to_2880(self) -> None:
        # Each one-decimal percentage over 360 matchups pins a unique integer
        # win count (grid spacing 100/360 ~ 0.278 > 0.1); the ten inverted
        # counts must reproduce the headline 2880 exactly, and must match the
        # win-count column of tab:tok_context_by_competitor verbatim.
        vals = _load_paper_csv()
        per_comp = {
            k.removeprefix("win_rate_vs_").removesuffix("_pct"): v
            for k, v in vals.items()
            if k.startswith("win_rate_vs_")
        }
        counts = {
            name: win_count_from_published_pct(p, matchups=360) for name, p in per_comp.items()
        }
        assert counts == {
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
        assert sum(counts.values()) == 2880 == vals["matchups_won"]

    def test_inversion_helper_guards(self) -> None:
        # 80.8% over 360 matchups is exactly 291 wins and nothing else.
        assert win_count_from_published_pct(80.8, matchups=360) == 291
        # A percentage no integer count rounds to is rejected...
        with pytest.raises(ValueError, match="no win count"):
            win_count_from_published_pct(80.9, matchups=360)
        # ...and a print precision coarser than the grid is ambiguous:
        # at 0 decimals both 287 and 288 of 360 round to 80%.
        with pytest.raises(ValueError, match="ambiguous"):
            win_count_from_published_pct(80.0, matchups=360, decimals=0)
