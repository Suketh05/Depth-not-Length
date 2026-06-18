"""Tests for decision-compliance scoring."""

from __future__ import annotations

from membench.metrics.compliance import (
    decision_keywords,
    score_compliance,
    score_compliance_for_task,
)
from membench.types import MemoryItem, Scorer, SpecVariant, Task


def _corpus() -> dict[str, MemoryItem]:
    return {
        "D-1": MemoryItem("D-1", "All exports must call withAuditLog() before streaming."),
        "D-2": MemoryItem("D-2", "Use the DateRangePicker component for date ranges."),
        "D-3": MemoryItem("D-3", "see ADR-17"),  # no code identifier
    }


class TestDecisionKeywords:
    def test_extracts_identifiers_and_bare_form(self) -> None:
        kws = decision_keywords("call withAuditLog() and use DateRangePicker")
        assert "withAuditLog()" in kws and "withAuditLog" in kws
        assert "DateRangePicker" in kws


class TestScoreCompliance:
    def test_no_governing_is_vacuously_compliant(self) -> None:
        assert score_compliance("anything", (), {}).rate == 1.0

    def test_full_compliance(self) -> None:
        r = score_compliance(
            "I will call withAuditLog and use DateRangePicker", ("D-1", "D-2"), _corpus()
        )
        assert r.rate == 1.0 and r.honored == 2
        assert set(r.honored_ids) == {"D-1", "D-2"}

    def test_partial_compliance(self) -> None:
        r = score_compliance("I will call withAuditLog only", ("D-1", "D-2"), _corpus())
        assert r.honored == 1 and r.rate == 0.5
        assert r.honored_ids == ("D-1",)

    def test_zero_compliance(self) -> None:
        r = score_compliance("did something unrelated", ("D-1", "D-2"), _corpus())
        assert r.honored == 0 and r.rate == 0.0

    def test_bare_id_fallback_when_no_identifier(self) -> None:
        assert score_compliance("per D-3 we proceed", ("D-3",), _corpus()).honored == 1
        assert score_compliance("no reference here", ("D-3",), _corpus()).honored == 0

    def test_missing_decision_text_is_not_honored(self) -> None:
        # governing id absent from corpus and not echoed -> not honoured
        assert score_compliance("text", ("D-404",), _corpus()).honored == 0

    def test_for_task_wrapper(self) -> None:
        task = Task(
            task_id="t",
            dataset="dcbench",
            query="q",
            repo_ref="r",
            memory_corpus=tuple(_corpus().values()),
            governing_decisions=("D-1",),
            depth=2,
            spec_variant=SpecVariant.STRIPPED,
            scorer=Scorer.COMPLIANCE,
        )
        assert score_compliance_for_task("call withAuditLog()", task).rate == 1.0
