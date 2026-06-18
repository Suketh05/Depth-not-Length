"""Merge-ready correctness for coding tasks: would this output actually go
in as written.

Real merge-readiness depends on tests passing and a diff review -- this repo
has no live coding agent yet (agent/llm_client.py produces text completions,
not diffs against a checked-out repo), so the proxy is: the response is
non-empty, isn't a refusal, and honors every governing decision. Swap this
for a real diff+test harness once one exists (PLAN.md Step 6+); the
ComplianceResult dependency is deliberate so that swap doesn't touch the
compliance contract.
"""

from dataclasses import dataclass

from benchmarks.task import Task
from scoring.compliance import ComplianceResult

_REFUSAL_MARKERS = ("i can't", "i cannot", "i won't", "as an ai", "i'm not able to")


@dataclass
class CorrectnessResult:
    task_id: str
    merge_ready: bool
    reason: str


def score_correctness(
    response_text: str, task: Task, compliance: ComplianceResult
) -> CorrectnessResult:
    text = response_text.strip()
    if not text:
        return CorrectnessResult(task.task_id, merge_ready=False, reason="empty response")

    lowered = text.lower()
    if any(marker in lowered for marker in _REFUSAL_MARKERS):
        return CorrectnessResult(task.task_id, merge_ready=False, reason="refusal")

    if compliance.rate < 1.0:
        missed = compliance.total - compliance.honored
        return CorrectnessResult(
            task.task_id,
            merge_ready=False,
            reason=f"{missed}/{compliance.total} governing decisions unmet",
        )

    return CorrectnessResult(
        task.task_id, merge_ready=True, reason="all governing decisions honored"
    )
