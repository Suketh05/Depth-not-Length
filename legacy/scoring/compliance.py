"""Decision compliance: of task.governing_decisions, how many does the
output honor.

The Task contract only carries gold decision IDs, not what honoring each one
actually requires -- a real coding agent has no reason to ever emit the
literal string "D-002". So for each gold ID we look up its decision text in
task.memory_corpus (present regardless of arm or spec-stripping -- it's the
ground truth, not what was retrieved) and check the output for the concrete
identifiers that decision names: PascalCase/camelCase symbols, function
calls, scoped package names. Falls back to the bare ID only when a decision's
text has no such identifier to extract.
"""

import re
from dataclasses import dataclass

from benchmarks.task import Task

_IDENTIFIER_PATTERN = re.compile(
    r"[A-Za-z][A-Za-z0-9_]*\(\)"  # function calls: withAuditLog()
    r"|@[\w-]+/[\w-]+"  # scoped packages: @t3-oss/env-nextjs
    r"|\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b"  # PascalCase: DateRangePicker
    r"|\b[a-z]+(?:[A-Z][a-z0-9]*)+\b"  # camelCase: withAuditLog
)


@dataclass
class ComplianceResult:
    task_id: str
    total: int
    honored: int
    honored_ids: list[str]
    rate: float  # honored / total; 1.0 when there are no governing decisions


def _decision_keywords(decision_text: str) -> set[str]:
    found = _IDENTIFIER_PATTERN.findall(decision_text)
    keywords = set(found)
    # A natural-language completion will say "use withAuditLog" without the
    # call syntax just as often as it will write the call itself -- credit
    # both forms rather than demanding the literal parens.
    keywords.update(kw[:-2] for kw in found if kw.endswith("()"))
    return keywords


def score_compliance(response_text: str, task: Task) -> ComplianceResult:
    if not task.governing_decisions:
        return ComplianceResult(task.task_id, total=0, honored=0, honored_ids=[], rate=1.0)

    decision_text_by_id = {item.item_id: item.text for item in task.memory_corpus}
    honored_ids = []
    for decision_id in task.governing_decisions:
        decision_text = decision_text_by_id.get(decision_id, "")
        keywords = _decision_keywords(decision_text)
        if keywords:
            if any(keyword in response_text for keyword in keywords):
                honored_ids.append(decision_id)
        elif decision_id in response_text:
            honored_ids.append(decision_id)

    total = len(task.governing_decisions)
    return ComplianceResult(
        task_id=task.task_id,
        total=total,
        honored=len(honored_ids),
        honored_ids=honored_ids,
        rate=len(honored_ids) / total,
    )
