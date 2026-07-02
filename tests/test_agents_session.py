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

from membench.agents.llm.pricing import (
    SESSION_BLENDED_RATE_PER_MTOK,
    TOKECON_BACKEND_MODELS,
    session_blended_cost,
)
from membench.agents.session import (
    Session,
    TurnLedger,
    efficiency_score,
    tokens_per_resolved_point,
)
from membench.types import Task


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


# tab:tok_agent_economics -- pooled quality x cost, gpt-4o billing. Transcribed
# verbatim: system, resolution %, session tokens, USD, USD decimals, tokens per
# resolved point, composite efficiency score.
_ECON_ROWS: tuple[tuple[str, float, int, float, int, float, float], ...] = (
    ("Brief", 48.0, 12400, 0.057, 3, 258.3, 387.1),
    ("Mem0", 24.2, 45235, 0.2081, 4, 1869.2, 53.5),
    ("Zep", 32.6, 45942, 0.2113, 4, 1409.3, 70.96),
    ("Supermemory", 30.3, 46342, 0.2132, 4, 1529.4, 65.38),
    ("MemGPT", 30.3, 45541, 0.2095, 4, 1503.0, 66.53),
    ("GraphRAG", 28.1, 48382, 0.2226, 4, 1721.8, 58.08),
    ("A-Mem", 32.6, 45250, 0.2082, 4, 1388.0, 72.04),
    ("RAPTOR", 26.6, 48167, 0.2216, 4, 1810.8, 55.22),
    ("ContextQ", 34.0, 45049, 0.2072, 4, 1325.0, 75.47),
    ("Unabyss", 29.0, 47768, 0.2197, 4, 1647.2, 60.71),
    ("ctx|", 31.1, 46290, 0.2129, 4, 1488.4, 67.19),
    ("Driver", 31.7, 46512, 0.214, 3, 1467.3, 68.15),
    ("Oiya", 27.3, 50298, 0.2314, 4, 1842.4, 54.28),
    ("Kluris", 23.6, 51289, 0.2359, 4, 2173.3, 46.01),
    ("Oracle Summary", 34.3, 44767, 0.2059, 4, 1305.2, 76.62),
    ("OpenViking", 29.2, 47972, 0.2207, 4, 1642.9, 60.87),
)


class TestBlendedRateGpt4o:
    """The gpt-4o blended rate reproduces every USD in tab:tok_agent_economics."""

    @pytest.mark.parametrize("row", _ECON_ROWS, ids=[r[0] for r in _ECON_ROWS])
    def test_blended_rate_reproduces_published_usd(
        self, row: tuple[str, float, int, float, int, float, float]
    ) -> None:
        _system, _res, session_tok, usd, decimals, *_ = row
        computed = session_blended_cost("gpt-4o", session_tok)
        assert abs(computed - usd) <= _usd_tol(decimals)


class TestBlendedRateUniqueness:
    """The published rates are the unique cent-level rates that fit every row."""

    def test_gpt55_rate_is_12_dollars(self) -> None:
        assert SESSION_BLENDED_RATE_PER_MTOK["gpt-5.5"].per_mtok == 12.00

    def test_gpt4o_rate_is_4_60_dollars(self) -> None:
        assert SESSION_BLENDED_RATE_PER_MTOK["gpt-4o"].per_mtok == 4.60

    def test_neighbouring_gpt55_rates_break_a_row(self) -> None:
        # $11.99 mis-rounds the 2172-tok row; $12.01 mis-rounds the 3686-tok row.
        assert round(2172 * 11.99e-6, 4) != 0.0261
        assert round(3686 * 12.01e-6, 4) != 0.0442
        # ... while $12.00 gets both right.
        assert round(2172 * 12.00e-6, 4) == 0.0261
        assert round(3686 * 12.00e-6, 4) == 0.0442

    def test_rates_are_flagged_verified_against_their_table(self) -> None:
        assert (
            SESSION_BLENDED_RATE_PER_MTOK["gpt-5.5"].source_table == "tab:tok_agent_context_gpt55"
        )
        assert SESSION_BLENDED_RATE_PER_MTOK["gpt-4o"].source_table == "tab:tok_agent_economics"
        assert all(rate.verified for rate in SESSION_BLENDED_RATE_PER_MTOK.values())


class TestPricingOmission:
    """The paper names backend models but publishes no per-token price for them."""

    def test_named_backends_have_no_invented_blended_rate(self) -> None:
        # Only the two blended rates the paper's USD columns imply are priced; the
        # individually-named backends (e.g. gpt-5.3-codex) are intentionally not.
        unpriced = TOKECON_BACKEND_MODELS - set(SESSION_BLENDED_RATE_PER_MTOK)
        assert "gpt-5.3-codex" in unpriced
        assert "claude-opus-4-8" in unpriced
        for model in unpriced:
            with pytest.raises(KeyError, match="no blended session rate"):
                session_blended_cost(model, 1000)


class TestEconomicsIdentities:
    """tok-per-resolved-point and efficiency-score columns of tab:tok_agent_economics."""

    @pytest.mark.parametrize("row", _ECON_ROWS, ids=[r[0] for r in _ECON_ROWS])
    def test_tokens_per_resolved_point(
        self, row: tuple[str, float, int, float, int, float, float]
    ) -> None:
        _system, res, session_tok, _usd, _d, tprp, _eff = row
        # session_tokens / resolution% (Brief: 12400/48.0 = 258.3). Paper prints 1 dp.
        assert tokens_per_resolved_point(session_tok, res) == pytest.approx(tprp, abs=0.05)

    @pytest.mark.parametrize("row", _ECON_ROWS, ids=[r[0] for r in _ECON_ROWS])
    def test_efficiency_score(self, row: tuple[str, float, int, float, int, float, float]) -> None:
        _system, res, session_tok, _usd, _d, _tprp, eff = row
        # 1e5 * resolution% / session_tokens (Brief: 1e5*48.0/12400 = 387.1).
        assert efficiency_score(res, session_tok) == pytest.approx(eff, abs=0.05)

    def test_efficiency_is_reciprocal_of_tok_per_point_up_to_scale(self) -> None:
        # eff = 1e5 * res / tok ; tprp = tok / res ; so eff * tprp == 1e5 exactly.
        for _system, res, session_tok, *_ in _ECON_ROWS:
            eff = efficiency_score(res, session_tok)
            tprp = tokens_per_resolved_point(session_tok, res)
            assert eff * tprp == pytest.approx(1e5, rel=1e-12)

    def test_domain_guards(self) -> None:
        with pytest.raises(ValueError, match="resolution_pct"):
            tokens_per_resolved_point(1000, 0.0)
        with pytest.raises(ValueError, match="resolution_pct"):
            tokens_per_resolved_point(1000, 101.0)
        with pytest.raises(ValueError, match="session_tokens"):
            efficiency_score(30.0, 0)


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

    def test_session_api_reproduces_brief_row_usd(self) -> None:
        # The whole "Brief context" row through the Session API: 1598/203/34 tokens
        # at the GPT-5.5 blended rate -> the published $0.022.
        turns = (
            TurnLedger.from_total(1, 1598, resolved=False),
            TurnLedger.from_total(2, 203, resolved=False),
            TurnLedger.from_total(3, 34, resolved=True),
        )
        session = Session("gpt-5.5", "brief_graph", turns)
        assert abs(session.blended_cost_dollars() - 0.022) <= _usd_tol(3)
        # Cumulative dollars are monotone and end at the session cost.
        cumulative = session.cumulative_dollars()
        assert list(cumulative) == sorted(cumulative)
        assert cumulative[-1] == pytest.approx(session.blended_cost_dollars())

    def test_cross_pricing_the_same_ledger_under_gpt4o(self) -> None:
        # The same token ledger can be repriced under a different backend rate; a
        # 12400-token session at the gpt-4o rate is the paper's Brief economics USD.
        session = Session(
            "gpt-5.5", "brief_graph", (TurnLedger.from_total(1, 12400, resolved=True),)
        )
        assert abs(session.blended_cost_dollars("gpt-4o") - 0.057) <= _usd_tol(3)

    def test_as_row_is_json_friendly(self) -> None:
        turns = (
            TurnLedger.from_total(1, 1598, resolved=False),
            TurnLedger.from_total(2, 203, resolved=False),
            TurnLedger.from_total(3, 34, resolved=True),
        )
        row = Session("gpt-5.5", "brief_graph", turns).as_row()
        assert row["per_turn_tokens"] == [1598, 203, 34]
        assert row["session_tokens"] == 1835
        assert row["convergence_turn"] == 3
        assert row["resolved"] is True
        assert isinstance(row["per_turn_tokens"], list)


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


class TestCostEdgeCases:
    """Zero-token / empty and out-of-domain cost behaviour."""

    def test_empty_session_costs_nothing(self) -> None:
        assert Session("gpt-5.5", "none", ()).blended_cost_dollars() == 0.0

    def test_zero_token_turn_costs_nothing(self) -> None:
        # A turn that made a call but billed zero tokens (a degenerate chance-floor
        # case) contributes no tokens and no dollars.
        session = Session("gpt-5.5", "none", (TurnLedger(1, 0, 0, resolved=False),))
        assert session.session_tokens == 0
        assert session.blended_cost_dollars() == 0.0

    def test_unpriced_session_model_raises(self) -> None:
        # A session on a backend with no blended rate must fail loud, not zero-cost.
        session = Session(
            "claude-sonnet-4-6", "brief_graph", (TurnLedger(1, 100, 20, resolved=True),)
        )
        with pytest.raises(KeyError, match="no blended session rate"):
            session.blended_cost_dollars()

    def test_session_blended_cost_rejects_negative_tokens(self) -> None:
        with pytest.raises(ValueError, match="session_tokens"):
            session_blended_cost("gpt-5.5", -1)


class TestSplitInvariance:
    """Session tokens and cost depend only on per-turn totals, not the split."""

    def test_session_tokens_and_cost_invariant_to_prompt_completion_split(self) -> None:
        # Two ledgers with the same per-turn TOTALS but different prompt/completion
        # splits (the paper does not publish the split) must agree on every reported
        # quantity -- session tokens and blended cost.
        totals = (1598, 203, 34)
        all_prompt = Session(
            "gpt-5.5",
            "brief_graph",
            tuple(TurnLedger(i, t, 0, resolved=(i == 3)) for i, t in enumerate(totals, start=1)),
        )
        mixed = Session(
            "gpt-5.5",
            "brief_graph",
            (
                TurnLedger(1, 1200, 398, resolved=False),
                TurnLedger(2, 150, 53, resolved=False),
                TurnLedger(3, 20, 14, resolved=True),
            ),
        )
        assert all_prompt.session_tokens == mixed.session_tokens == 1835
        assert all_prompt.blended_cost_dollars() == mixed.blended_cost_dollars()
        assert all_prompt.convergence_turn == mixed.convergence_turn == 3


def _make_system(arm: str) -> tuple[object, object]:
    """Return a fresh (llm, memory) pair; a memory arm is single-use per session."""
    import membench.retrieval.builtin  # noqa: F401  (registers the builtin arms)
    from membench.agents.combos import Combo, build_system

    return build_system(Combo(arm, "claude", arm), offline=True)


def _stripped_task() -> Task:
    from membench.types import MemoryItem, Scorer, SpecVariant, Task

    corpus = (
        MemoryItem(
            "D-1",
            "primary buttons must use the DateRangePicker component",
            {"node_type": "constraint"},
        ),
        MemoryItem(
            "D-1-rationale",
            "the export dialog call to action should be accessible",
            {"node_type": "justification", "constrains": "D-1"},
        ),
        MemoryItem("D-2", "unrelated note about logging"),
    )
    return Task(
        task_id="sess-1",
        dataset="dcbench",
        query="make the export dialog call to action accessible",
        repo_ref="github.com/x/y@a",
        memory_corpus=corpus,
        governing_decisions=("D-1",),
        depth=3,
        spec_variant=SpecVariant.STRIPPED,
        scorer=Scorer.COMPLIANCE,
    )


def _honours_governing(_turn: int, text: str, _task: Task) -> bool:
    """A compliance-style predicate: the governing identifier is echoed back."""
    return "DateRangePicker" in text


class TestRunSession:
    def test_context_arm_resolves_on_turn_one(self) -> None:
        from membench.agents.runner import RunConfig
        from membench.agents.session import run_session

        llm, memory = _make_system("brief_graph")
        session = run_session(
            _stripped_task(),
            memory,
            llm,
            RunConfig(budget_tokens=10_000),
            arm_name="brief_graph",
            resolves_at=_honours_governing,
            max_turns=8,
        )
        # The typed store recovers D-1 through the CONSTRAINS link into turn-1
        # context, so the stub honours it and the session collapses immediately.
        assert session.convergence_turn == 1
        assert session.num_turns == 1
        assert session.arm_label == "brief_graph"
        assert session.reported_dollars > 0

    def test_no_context_runs_to_the_turn_cap_unresolved(self) -> None:
        from membench.agents.runner import RunConfig
        from membench.agents.session import run_session

        llm, memory = _make_system("none")
        session = run_session(
            _stripped_task(),
            memory,
            llm,
            RunConfig(budget_tokens=10_000),
            arm_name="none",
            resolves_at=_honours_governing,
            max_turns=6,
        )
        # No context layer -> the governing decision never enters the prompt -> the
        # session runs the full budget of turns and never resolves.
        assert session.convergence_turn is None
        assert session.resolved is False
        assert session.num_turns == 6

    def test_convergence_at_a_scripted_turn(self) -> None:
        from membench.agents.runner import RunConfig
        from membench.agents.session import run_session

        llm, memory = _make_system("none")
        session = run_session(
            _stripped_task(),
            memory,
            llm,
            RunConfig(budget_tokens=10_000),
            arm_name="none",
            resolves_at=lambda turn, _text, _task: turn >= 3,
            max_turns=8,
        )
        assert session.convergence_turn == 3
        assert session.num_turns == 3

    def test_is_deterministic(self) -> None:
        from membench.agents.runner import RunConfig
        from membench.agents.session import run_session

        def run() -> object:
            llm, memory = _make_system("brief_graph")
            return run_session(
                _stripped_task(),
                memory,
                llm,
                RunConfig(budget_tokens=10_000),
                arm_name="brief_graph",
                resolves_at=lambda turn, _t, _k: turn >= 3,
                max_turns=8,
            )

        assert run() == run()

    def test_custom_follow_up_is_used(self) -> None:
        from membench.agents.runner import RunConfig
        from membench.agents.session import run_session

        seen: list[int] = []

        def follow_up(turn: int, _task: Task) -> str:
            seen.append(turn)
            return "please continue"

        llm, memory = _make_system("none")
        run_session(
            _stripped_task(),
            memory,
            llm,
            RunConfig(budget_tokens=10_000),
            arm_name="none",
            resolves_at=lambda turn, _t, _k: turn >= 3,
            max_turns=8,
            follow_up=follow_up,
        )
        assert seen == [2, 3]  # follow-up builds turns 2 and 3 only

    def test_max_turns_one_is_single_turn(self) -> None:
        from membench.agents.runner import RunConfig
        from membench.agents.session import run_session

        llm, memory = _make_system("none")
        session = run_session(
            _stripped_task(),
            memory,
            llm,
            RunConfig(budget_tokens=10_000),
            arm_name="none",
            resolves_at=lambda *_: False,
            max_turns=1,
        )
        assert session.num_turns == 1

    def test_rejects_non_positive_max_turns(self) -> None:
        from membench.agents.runner import RunConfig
        from membench.agents.session import run_session

        llm, memory = _make_system("none")
        with pytest.raises(ValueError, match="max_turns"):
            run_session(
                _stripped_task(),
                memory,
                llm,
                RunConfig(budget_tokens=10_000),
                arm_name="none",
                resolves_at=lambda *_: True,
                max_turns=0,
            )
