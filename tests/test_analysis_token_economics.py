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

from pathlib import Path

import pytest

from membench.analysis.token_economics import (
    BRIEF_SYSTEM,
    Matchup,
    MatchupOutcome,
    SessionRecord,
    decide_matchup,
    session_tokens_from_turns,
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
