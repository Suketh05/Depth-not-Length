"""Tests for multi-turn sessions and the per-turn token ledger (paper sec:tokecon).

Golden values are transcribed VERBATIM from the paper's token-economics tables
(tab:tok_agent_context_gpt55, tab:tok_context_winrate,
tab:tok_context_by_competitor, tab:tok_agent_economics, tab:tok_pareto,
tab:tok_context_brief_losses) — never produced by running this repo's code.
Where the paper prints a rounded decimal, the assertion tolerance is half an
ulp of the printed precision (0.51 * 10^-dp), so a reader can reproduce every
check with a hand calculator.
"""

from __future__ import annotations

import pytest

from membench.agents.llm.pricing import (
    PAPER_SESSION_RATES,
    PRICING,
    UNPRICED_SWEEP_MODELS,
    blended_session_dollars,
)
from membench.agents.session import PaperLedgerRow, tokens_per_resolved_point

# ---------------------------------------------------------------------------
# tab:tok_agent_context_gpt55 — "Multi-turn collapse, one fixed backend model
# (GPT-5.5)". Each row: configuration, Turn 1, Turn 2, Turn 3, Turn 4+,
# session tokens, printed USD, printed USD decimal places, outcome turn count,
# resolved?  All numbers verbatim from the paper table.
# ---------------------------------------------------------------------------
GPT55_LEDGER_ROWS: list[tuple[str, int, int, int, int | None, int, float, int, int, bool]] = [
    ("alone", 1403, 698, 601, 1461, 4163, 0.05, 2, 6, False),
    ("alone (best case)", 1420, 715, 612, 939, 3686, 0.0442, 4, 5, True),
    ("Brief context", 1598, 203, 34, None, 1835, 0.022, 3, 3, True),
    ("Brief context (slow task)", 1598, 412, 118, 44, 2172, 0.0261, 4, 4, True),
    ("Mem0 context", 1712, 684, 598, 1743, 4737, 0.0568, 4, 7, True),
    ("Mem0 context (worst)", 1708, 691, 605, 2166, 5170, 0.062, 3, 8, False),
    ("ContextQ context", 1648, 612, 541, 926, 3727, 0.0447, 4, 5, False),
    ("Zep context", 1576, 448, 382, 318, 2724, 0.0327, 4, 4, True),
]


class TestGpt55PaperLedgerGoldens:
    """Every published GPT-5.5 ledger row satisfies the ledger arithmetic."""

    @pytest.mark.parametrize(
        ("config", "t1", "t2", "t3", "t4p", "session"),
        [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in GPT55_LEDGER_ROWS],
    )
    def test_columns_sum_to_session_tokens(
        self, config: str, t1: int, t2: int, t3: int, t4p: int | None, session: int
    ) -> None:
        # Constructing PaperLedgerRow *is* the assertion: its validator raises
        # unless T1 + T2 + T3 + T4+ == session tokens. E.g. the first "alone"
        # row: 1403 + 698 + 601 + 1461 = 4163 (paper tab:tok_agent_context_gpt55).
        row = PaperLedgerRow(t1, t2, t3, t4p, session)
        assert row.session_tokens == session

    @pytest.mark.parametrize(
        ("config", "session", "usd", "dp"),
        [(r[0], r[5], r[6], r[7]) for r in GPT55_LEDGER_ROWS],
    )
    def test_usd_column_is_flat_12_dollars_per_mtok(
        self, config: str, session: int, usd: float, dp: int
    ) -> None:
        # The caption fixes the backend model and "its per-token price ...
        # identical across rows"; the constant that reproduces every printed
        # USD cell is $12.00 per million session tokens. Worked example, first
        # row: 4163 tok x 12e-6 = $0.049956 -> printed $0.05.
        computed = blended_session_dollars("gpt-5.5", session)
        assert abs(computed - usd) <= 0.51 * 10.0**-dp

    def test_turn4_plus_column_is_present_iff_session_ran_past_turn_3(self) -> None:
        # The 3-turn "Brief context" row prints "---" for Turn 4+; every row
        # whose outcome column reports >= 4 turns has a Turn 4+ entry.
        for _config, _t1, _t2, _t3, t4p, _sess, _usd, _dp, turns, _res in GPT55_LEDGER_ROWS:
            assert (t4p is None) == (turns <= 3)

    def test_brief_collapse_headline(self) -> None:
        # sec:tokecon: with Brief's context the session "collapses" — it is the
        # strictly cheapest published row (1835 session tokens), resolves, and
        # its convergence turn is 3 ("3 turns, resolved").
        brief = GPT55_LEDGER_ROWS[2]
        assert brief[5] == min(row[5] for row in GPT55_LEDGER_ROWS)
        assert all(row[5] > brief[5] for row in GPT55_LEDGER_ROWS if row[0] != brief[0])
        assert brief[8] == 3 and brief[9] is True


# ---------------------------------------------------------------------------
# tab:tok_agent_economics — "Quality x cost across context layers (pooled,
# fixed backend model)". Each row: system, resolution %, session tokens,
# printed USD (gpt-4o), printed USD decimal places, printed tokens per
# resolved point. All numbers verbatim from the paper table.
# ---------------------------------------------------------------------------
ECONOMICS_ROWS: list[tuple[str, float, int, float, int, float]] = [
    ("Brief", 48.0, 12400, 0.057, 3, 258.3),
    ("Mem0", 24.2, 45235, 0.2081, 4, 1869.2),
    ("Zep", 32.6, 45942, 0.2113, 4, 1409.3),
    ("Supermemory", 30.3, 46342, 0.2132, 4, 1529.4),
    ("MemGPT", 30.3, 45541, 0.2095, 4, 1503.0),
    ("GraphRAG", 28.1, 48382, 0.2226, 4, 1721.8),
    ("A-Mem", 32.6, 45250, 0.2082, 4, 1388.0),
    ("RAPTOR", 26.6, 48167, 0.2216, 4, 1810.8),
    ("ContextQ", 34.0, 45049, 0.2072, 4, 1325.0),
    ("Unabyss", 29.0, 47768, 0.2197, 4, 1647.2),
    ("ctx|", 31.1, 46290, 0.2129, 4, 1488.4),
    ("Driver", 31.7, 46512, 0.214, 3, 1467.3),
    ("Oiya", 27.3, 50298, 0.2314, 4, 1842.4),
    ("Kluris", 23.6, 51289, 0.2359, 4, 2173.3),
    ("Oracle Summary", 34.3, 44767, 0.2059, 4, 1305.2),
    ("OpenViking", 29.2, 47972, 0.2207, 4, 1642.9),
]


class TestGpt4oEconomicsGoldens:
    """All 16 tab:tok_agent_economics rows satisfy the two column identities."""

    @pytest.mark.parametrize(
        ("system", "session", "usd", "dp"),
        [(r[0], r[2], r[3], r[4]) for r in ECONOMICS_ROWS],
    )
    def test_usd_column_is_flat_4_60_dollars_per_mtok(
        self, system: str, session: int, usd: float, dp: int
    ) -> None:
        # The USD (gpt-4o) column is session tokens x $4.60/Mtok at printed
        # precision. Worked example, Brief row: 12400 x 4.6e-6 = $0.05704 ->
        # printed $0.057; Kluris: 51289 x 4.6e-6 = $0.2359294 -> printed $0.2359.
        computed = blended_session_dollars("gpt-4o", session)
        assert abs(computed - usd) <= 0.51 * 10.0**-dp

    @pytest.mark.parametrize(
        ("system", "resolution", "session", "tok_per_pt"),
        [(r[0], r[1], r[2], r[5]) for r in ECONOMICS_ROWS],
    )
    def test_tokens_per_resolved_point_column(
        self, system: str, resolution: float, session: int, tok_per_pt: float
    ) -> None:
        # Tok./res. pt = session tokens / resolution %. Worked example, Brief
        # row: 12400 / 48.0 = 258.33... -> printed 258.3 (paper prose: "about
        # 258 tokens per resolved point against 1300--2200 for everyone else").
        computed = tokens_per_resolved_point(session, resolution)
        assert abs(computed - tok_per_pt) <= 0.051

    def test_brief_sits_off_the_competitor_frontier(self) -> None:
        # sec:tokecon: Brief resolves most (48.0%) at ~a quarter of the session
        # tokens (12,400 vs 44,000--51,000); every competitor row is inside
        # 44,767..51,289 tokens and 23.6..34.3% resolution.
        brief = ECONOMICS_ROWS[0]
        competitors = ECONOMICS_ROWS[1:]
        assert brief[1] == max(row[1] for row in ECONOMICS_ROWS)
        assert brief[2] == min(row[2] for row in ECONOMICS_ROWS)
        assert all(44767 <= row[2] <= 51289 and 23.6 <= row[1] <= 34.3 for row in competitors)


class TestPaperSessionRates:
    """The pricing table holds paper-stated rates only, and fails loud otherwise."""

    def test_only_the_two_paper_priced_models(self) -> None:
        # sec:tokecon publishes USD columns for exactly two backends: GPT-5.5
        # (tab:tok_agent_context_gpt55) and gpt-4o (tab:tok_agent_economics).
        assert set(PAPER_SESSION_RATES) == {"gpt-5.5", "gpt-4o"}
        assert PAPER_SESSION_RATES["gpt-5.5"].usd_per_mtok == 12.00
        assert PAPER_SESSION_RATES["gpt-4o"].usd_per_mtok == 4.60
        assert PAPER_SESSION_RATES["gpt-5.5"].provenance == "tab:tok_agent_context_gpt55"
        assert PAPER_SESSION_RATES["gpt-4o"].provenance == "tab:tok_agent_economics"

    def test_unpriced_sweep_models_stay_unpriced(self) -> None:
        # Integrity gate: the sweep backends the paper names but never prices
        # must not silently acquire a paper rate.
        assert not UNPRICED_SWEEP_MODELS & set(PAPER_SESSION_RATES)

    def test_unknown_model_fails_loud(self) -> None:
        with pytest.raises(KeyError, match="no blended session rate"):
            blended_session_dollars("gpt-5.3-codex", 1000)

    def test_negative_tokens_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            blended_session_dollars("gpt-4o", -1)

    def test_vendor_table_untouched_by_session_rates(self) -> None:
        # The blended paper rates are a separate table; the vendor input/output
        # table keeps its original seven rows unchanged (behaviour-preserving).
        assert set(PRICING) == {
            "claude-opus-4-8",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
            "claude-fable-5",
            "gpt-5.1",
            "llama-4-maverick",
            "stub",
        }
