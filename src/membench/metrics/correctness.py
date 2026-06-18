"""Merge-ready correctness for coding tasks: would this output go in as written.

Real merge-readiness depends on tests passing and a diff review; this harness runs
a deterministic stub agent (text, not a checked-out repo), so the proxy is: the
response is non-empty, is not a refusal, and honours every governing decision. The
dependency on :class:`ComplianceResult` is deliberate -- when a real diff+test
harness is wired in, it replaces this function without touching the compliance
contract. ``merge_ready`` is the binary the cost-to-correct metric treats as
"correct".
"""

from __future__ import annotations

from dataclasses import dataclass

from membench.metrics.compliance import ComplianceResult

__all__ = ["CorrectnessResult", "score_correctness"]

_REFUSAL_MARKERS = ("i can't", "i cannot", "i won't", "as an ai", "i'm not able to")


@dataclass(frozen=True, slots=True)
class CorrectnessResult:
    """Whether a response is merge-ready, with the reason."""

    merge_ready: bool
    reason: str


def score_correctness(response_text: str, compliance: ComplianceResult) -> CorrectnessResult:
    """Judge merge-readiness from the response and its compliance result."""
    text = response_text.strip()
    if not text:
        return CorrectnessResult(merge_ready=False, reason="empty response")
    if any(marker in text.lower() for marker in _REFUSAL_MARKERS):
        return CorrectnessResult(merge_ready=False, reason="refusal")
    if compliance.rate < 1.0:
        missed = compliance.total - compliance.honored
        return CorrectnessResult(
            merge_ready=False, reason=f"{missed}/{compliance.total} governing decisions unmet"
        )
    return CorrectnessResult(merge_ready=True, reason="all governing decisions honoured")
