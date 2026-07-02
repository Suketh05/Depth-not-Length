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

from membench.agents.session import Session, TurnLedger


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
