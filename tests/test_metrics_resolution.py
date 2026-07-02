"""Tests for SWE-ContextBench resolution scoring (paper sec:pubbench / tab:stdbench).

All rate goldens are hand-computed exact fractions; the paper's published
resolution numbers (Brief 47.3% etc.) are provenance in docstrings only and
are never asserted here.
"""

from __future__ import annotations

from membench.metrics.compliance import ComplianceResult
from membench.metrics.resolution import (
    TaskResolution,
    score_resolution,
)


def _compliance(honored: int, total: int) -> ComplianceResult:
    """Build a ComplianceResult with the given honoured/total counts."""
    rate = honored / total if total else 1.0
    ids = tuple(f"D-{i:03d}" for i in range(honored))
    return ComplianceResult(total=total, honored=honored, honored_ids=ids, rate=rate)


class TestScoreResolutionProxy:
    def test_fully_compliant_response_is_resolved(self) -> None:
        outcome = score_resolution("swe-1", "use withAuditLog() in the export", _compliance(2, 2))
        assert outcome == TaskResolution(
            task_id="swe-1",
            resolved=True,
            reason="all governing decisions honoured",
        )

    def test_empty_response_is_unresolved(self) -> None:
        outcome = score_resolution("swe-2", "   ", _compliance(2, 2))
        assert not outcome.resolved
        assert outcome.reason == "empty response"

    def test_refusal_is_unresolved(self) -> None:
        outcome = score_resolution("swe-3", "I can't modify that module.", _compliance(2, 2))
        assert not outcome.resolved
        assert outcome.reason == "refusal"

    def test_partial_compliance_is_unresolved_with_counts(self) -> None:
        # 1 of 2 governing decisions honoured -> unresolved, missed count named.
        outcome = score_resolution("swe-4", "some edit", _compliance(1, 2))
        assert not outcome.resolved
        assert outcome.reason == "1/2 governing decisions unmet"

    def test_no_governing_decisions_resolves_on_nonempty(self) -> None:
        # Vacuous compliance (rate 1.0 when total == 0): text-only bar remains.
        outcome = score_resolution("swe-5", "harmless edit", _compliance(0, 0))
        assert outcome.resolved


class TestScoreResolutionHarnessHook:
    def test_judge_verdict_overrides_proxy_negative(self) -> None:
        # Harness rejects (tests fail) even though the proxy would resolve.
        outcome = score_resolution(
            "swe-6", "fully compliant text", _compliance(2, 2), judge=lambda tid, text: False
        )
        assert not outcome.resolved
        assert outcome.reason == "harness verdict"

    def test_judge_verdict_overrides_proxy_positive(self) -> None:
        # Harness accepts (tests pass) even though compliance keywords missed:
        # on the judge path tests decide, not keywords.
        outcome = score_resolution(
            "swe-7", "patch body", _compliance(0, 2), judge=lambda tid, text: True
        )
        assert outcome.resolved

    def test_judge_receives_task_id_and_response_only(self) -> None:
        # The hook is arm-blind: exactly (task_id, response_text), nothing else.
        seen: list[tuple[str, str]] = []

        def spy_judge(task_id: str, response_text: str) -> bool:
            seen.append((task_id, response_text))
            return True

        score_resolution("swe-8", "diff --git a b", _compliance(1, 1), judge=spy_judge)
        assert seen == [("swe-8", "diff --git a b")]
