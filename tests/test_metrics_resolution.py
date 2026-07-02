"""Tests for SWE-ContextBench resolution scoring (paper sec:pubbench / tab:stdbench).

All rate goldens are hand-computed exact fractions; the paper's published
resolution numbers (Brief 47.3% etc.) are provenance in docstrings only and
are never asserted here.
"""

from __future__ import annotations

import pytest

from membench.metrics.compliance import ComplianceResult
from membench.metrics.resolution import (
    ResolutionRate,
    TaskResolution,
    resolution_by_slice,
    resolution_rate,
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


def _outcome(task_id: str, resolved: bool) -> TaskResolution:
    reason = "all governing decisions honoured" if resolved else "refusal"
    return TaskResolution(task_id=task_id, resolved=resolved, reason=reason)


class TestResolutionRate:
    def test_golden_seven_of_twelve(self) -> None:
        # Hand-computed on the paper's smallest slice size (swe3 has n = 12):
        # 7 resolved of 12 -> rate = 7/12 exactly.
        outcomes = [_outcome(f"t{i}", i < 7) for i in range(12)]
        result = resolution_rate(outcomes)
        assert result == ResolutionRate(resolved=7, total=12, rate=7 / 12)

    def test_all_resolved_and_none_resolved(self) -> None:
        assert resolution_rate([_outcome("t", True)]).rate == 1.0
        assert resolution_rate([_outcome("t", False)]).rate == 0.0

    def test_counts_ride_along(self) -> None:
        # The integer counts are part of the contract (small n stays visible).
        result = resolution_rate([_outcome("a", True), _outcome("b", False)])
        assert (result.resolved, result.total) == (1, 2)

    def test_empty_run_raises(self) -> None:
        # Zero tasks has no rate; silent 0 would fabricate a benchmark number.
        with pytest.raises(ValueError, match="at least one"):
            resolution_rate([])


class TestResolutionBySlice:
    def test_slices_keep_their_own_n(self) -> None:
        # swe1: 2 of 3 resolved -> 2/3; swe3: 1 of 2 -> 1/2. Slices never pool.
        outcomes = [
            _outcome("a", True),
            _outcome("b", True),
            _outcome("c", False),
            _outcome("d", True),
            _outcome("e", False),
        ]
        slices = {"a": "swe1", "b": "swe1", "c": "swe1", "d": "swe3", "e": "swe3"}
        by_slice = resolution_by_slice(outcomes, slices)
        assert by_slice == {
            "swe1": ResolutionRate(resolved=2, total=3, rate=2 / 3),
            "swe3": ResolutionRate(resolved=1, total=2, rate=1 / 2),
        }

    def test_unlabelled_task_fails_loud(self) -> None:
        # A task with no slice must not be silently dropped or pooled.
        with pytest.raises(ValueError, match="no slice label"):
            resolution_by_slice([_outcome("orphan", True)], {})

    def test_empty_run_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            resolution_by_slice([], {})

    def test_single_slice_matches_pooled_rate(self) -> None:
        # Degenerate consistency: one slice == the pooled statistic.
        outcomes = [_outcome("a", True), _outcome("b", False)]
        by_slice = resolution_by_slice(outcomes, {"a": "all", "b": "all"})
        assert by_slice["all"] == resolution_rate(outcomes)
