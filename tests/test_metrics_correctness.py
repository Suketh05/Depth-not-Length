"""Tests for merge-ready correctness scoring."""

from __future__ import annotations

from membench.metrics.compliance import ComplianceResult
from membench.metrics.correctness import score_correctness


def _compliance(rate: float, total: int = 2) -> ComplianceResult:
    honored = round(rate * total)
    return ComplianceResult(total=total, honored=honored, honored_ids=(), rate=rate)


def test_merge_ready_when_fully_compliant() -> None:
    r = score_correctness("call withAuditLog()", _compliance(1.0))
    assert r.merge_ready and "honoured" in r.reason


def test_empty_response_not_merge_ready() -> None:
    assert not score_correctness("   ", _compliance(1.0)).merge_ready


def test_refusal_not_merge_ready() -> None:
    r = score_correctness("I cannot help with that.", _compliance(1.0))
    assert not r.merge_ready and r.reason == "refusal"


def test_partial_compliance_not_merge_ready() -> None:
    r = score_correctness("did half of it", _compliance(0.5))
    assert not r.merge_ready and "1/2 governing decisions unmet" in r.reason
