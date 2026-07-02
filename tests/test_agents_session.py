"""Tests for multi-turn sessions and the per-turn token ledger (paper sec:tokecon).

Golden values are transcribed VERBATIM from the paper's token-economics tables
(``tab:tok_agent_context_gpt55`` and ``tab:tok_agent_economics``) and from the
canonical golden pack; they are never produced by running the module under test.
Where a paper cell is a rounded dollar figure, the test asserts that the module's
value rounds to the published figure (tolerance = half a unit in the last published
place), which is the identity a reader adding up the paper's own columns can check.
"""

from __future__ import annotations

import pytest

from membench.agents.llm.pricing import session_blended_cost
from membench.agents.session import Session, TurnLedger


def _usd_tol(decimals: int) -> float:
    """Half a unit in the last published decimal place (plus float dust)."""
    return 0.5 * 10 ** (-decimals) + 1e-9


# tab:tok_agent_context_gpt55 -- one fixed backend (GPT-5.5). Per-turn columns
# (T1, T2, T3, T4+ with '---' dropped), the published session total, the published
# USD and its number of decimals, whether the row resolved, and the convergence
# turn ONLY for rows where the per-turn columns map 1:1 to turns (i.e. no T4+
# aggregation). Transcribed verbatim from the paper table.
# label, per_turn, session_tok, usd, usd_decimals, resolved, convergence
_GPT55_ROWS: tuple[tuple[str, tuple[int, ...], int, float, int, bool, int | None], ...] = (
    ("alone (unresolved)", (1403, 698, 601, 1461), 4163, 0.05, 2, False, None),
    ("alone (best)", (1420, 715, 612, 939), 3686, 0.0442, 4, True, None),
    ("Brief context", (1598, 203, 34), 1835, 0.022, 3, True, 3),
    ("Brief context (slow)", (1598, 412, 118, 44), 2172, 0.0261, 4, True, 4),
    ("Mem0 context (resolved)", (1712, 684, 598, 1743), 4737, 0.0568, 4, True, None),
    ("Mem0 context (unresolved)", (1708, 691, 605, 2166), 5170, 0.062, 3, False, None),
    ("ContextQ context", (1648, 612, 541, 926), 3727, 0.0447, 4, False, None),
    ("Zep context", (1576, 448, 382, 318), 2724, 0.0327, 4, True, 4),
)


class TestPaperTurnLedgerGpt55:
    """Cross-check every row of tab:tok_agent_context_gpt55."""

    @pytest.mark.parametrize("row", _GPT55_ROWS, ids=[r[0] for r in _GPT55_ROWS])
    def test_session_tokens_equal_sum_of_per_turn_columns(
        self, row: tuple[str, tuple[int, ...], int, float, int, bool, int | None]
    ) -> None:
        # A reader adding up the paper's per-turn columns must land on the
        # published "Session tok." cell -- an internal-consistency identity.
        _label, per_turn, session_tok, *_ = row
        assert sum(per_turn) == session_tok

    @pytest.mark.parametrize("row", _GPT55_ROWS, ids=[r[0] for r in _GPT55_ROWS])
    def test_blended_rate_reproduces_published_usd(
        self, row: tuple[str, tuple[int, ...], int, float, int, bool, int | None]
    ) -> None:
        _label, _per_turn, session_tok, usd, decimals, *_ = row
        computed = session_blended_cost("gpt-5.5", session_tok)
        assert abs(computed - usd) <= _usd_tol(decimals)

    def test_convergence_turns_for_clean_resolved_rows(self) -> None:
        # For rows whose per-turn columns map 1:1 to turns, a Session built from
        # them recovers the outcome column's turn count and resolution.
        for label, per_turn, _tok, _usd, _d, resolved, convergence in _GPT55_ROWS:
            if convergence is None:
                continue
            turns = tuple(
                TurnLedger.from_total(i, total, resolved=(i == len(per_turn) and resolved))
                for i, total in enumerate(per_turn, start=1)
            )
            session = Session("gpt-5.5", label, turns)
            assert session.convergence_turn == convergence, label
            assert session.num_turns == len(per_turn), label


class TestTurnLedger:
    def test_total_is_prompt_plus_completion(self) -> None:
        assert TurnLedger(1, 1598, 237, resolved=False).total_tokens == 1835

    def test_from_total_puts_whole_value_on_total(self) -> None:
        # The paper publishes per-turn TOTALS only; from_total must reproduce the
        # cell exactly regardless of the (unpublished) prompt/completion split.
        turn = TurnLedger.from_total(2, 203, resolved=False)
        assert turn.total_tokens == 203
        assert turn.completion_tokens == 0

    def test_from_response_carries_tokens_and_dollars(self) -> None:
        from membench.agents.llm.base import LLMResponse

        resp = LLMResponse(text="ok", input_tokens=100, output_tokens=20, dollars=0.01, model="m")
        turn = TurnLedger.from_response(3, resp, resolved=True)
        assert (turn.prompt_tokens, turn.completion_tokens) == (100, 20)
        assert turn.dollars == pytest.approx(0.01) and turn.resolved

    def test_rejects_zero_turn(self) -> None:
        with pytest.raises(ValueError, match="turn must be >= 1"):
            TurnLedger(0, 10, 5, resolved=False)

    def test_rejects_negative_tokens(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            TurnLedger(1, -1, 5, resolved=False)

    def test_rejects_negative_dollars(self) -> None:
        with pytest.raises(ValueError, match="dollars"):
            TurnLedger(1, 10, 5, resolved=False, dollars=-0.01)


def _turns(*totals_and_resolved: tuple[int, bool]) -> tuple[TurnLedger, ...]:
    return tuple(
        TurnLedger.from_total(i, total, resolved=res)
        for i, (total, res) in enumerate(totals_and_resolved, start=1)
    )


class TestSessionLedger:
    def test_session_tokens_sum_and_convergence(self) -> None:
        # paper tab:tok_agent_context_gpt55 "Brief context": 1598/203/34, resolves turn 3.
        session = Session("gpt-5.5", "brief_graph", _turns((1598, False), (203, False), (34, True)))
        assert session.session_tokens == 1835
        assert session.num_turns == 3
        assert session.convergence_turn == 3
        assert session.resolved is True

    def test_unresolved_session_has_no_convergence(self) -> None:
        session = Session("gpt-5.5", "none", _turns((1403, False), (698, False), (601, False)))
        assert session.convergence_turn is None
        assert session.resolved is False

    def test_cumulative_tokens_are_monotone_running_sum(self) -> None:
        session = Session("gpt-5.5", "brief_graph", _turns((1598, False), (203, False), (34, True)))
        assert session.cumulative_tokens() == (1598, 1801, 1835)

    def test_prompt_and_completion_totals(self) -> None:
        turns = (TurnLedger(1, 1000, 200, resolved=False), TurnLedger(2, 30, 4, resolved=True))
        session = Session("gpt-5.5", "brief_graph", turns)
        assert session.total_prompt_tokens == 1030
        assert session.total_completion_tokens == 204
        assert session.session_tokens == 1234

    def test_empty_session_is_well_defined(self) -> None:
        session = Session("gpt-5.5", "none", ())
        assert session.num_turns == 0
        assert session.session_tokens == 0
        assert session.convergence_turn is None
        assert session.resolved is False
        assert session.cumulative_tokens() == ()

    def test_rejects_non_contiguous_turns(self) -> None:
        with pytest.raises(ValueError, match="contiguous"):
            Session(
                "gpt-5.5",
                "none",
                (TurnLedger(1, 10, 1, resolved=False), TurnLedger(3, 10, 1, resolved=False)),
            )

    def test_rejects_turn_after_resolution(self) -> None:
        # Resolution is absorbing: no billed turn may follow a resolved one.
        with pytest.raises(ValueError, match="absorbing"):
            Session(
                "gpt-5.5",
                "none",
                (TurnLedger(1, 10, 1, resolved=True), TurnLedger(2, 10, 1, resolved=False)),
            )
