"""Tests for the Markdown report generator."""

from __future__ import annotations

import pytest

from membench.analysis.report import generate_report
from membench.analysis.tables import ScoredRow
from membench.competitors import load_vendor_claims


def _rows() -> list[ScoredRow]:
    rows: list[ScoredRow] = []
    for d in (1, 2, 3):
        rows.append(
            ScoredRow(
                "synthetic",
                "brief_graph",
                "claude-sonnet-4-6",
                d,
                "stripped",
                1.0,
                True,
                1.0,
                1.0,
                True,
                120,
                0.001,
            )
        )
        sim = max(0.0, 1.0 - 0.3 * d)
        rows.append(
            ScoredRow(
                "synthetic",
                "dense",
                "claude-sonnet-4-6",
                d,
                "stripped",
                sim,
                sim > 0.5,
                sim,
                sim,
                sim == 1.0,
                120,
                0.001,
            )
        )
        rows.append(
            ScoredRow(
                "synthetic",
                "none",
                "claude-sonnet-4-6",
                d,
                "stripped",
                0.0,
                False,
                0.0,
                0.0,
                False,
                50,
                0.0005,
            )
        )
    return rows


class TestReport:
    def test_has_expected_sections(self) -> None:
        md = generate_report(_rows(), vendor_claims=load_vendor_claims())
        for heading in (
            "# BriefBench Results",
            "## Executive summary",
            "## Depth crossover",
            "## Four-arm ablation",
            "## Competitor matrix",
            "### Measured",
            "### Vendor-reported",
            "Return on Tokens",
        ):
            assert heading in md

    def test_mentions_structured_advantage(self) -> None:
        md = generate_report(_rows())
        assert "brief_graph" in md
        # the deepest-depth structured number (1.00) and best baseline both appear
        assert "1.00" in md

    def test_vendor_claims_marked_pending_when_uncited(self) -> None:
        md = generate_report(_rows(), vendor_claims=load_vendor_claims())
        assert "pending citation" in md  # Mem0/Zep/GraphRAG placeholders

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="no scored rows"):
            generate_report([])
