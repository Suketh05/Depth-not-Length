"""Decision compliance: how many governing decisions the output actually honours.

The Task contract carries gold decision *ids*, not what honouring each requires --
a coding agent never emits the literal string "D-002". So for each gold id we look
up its decision text in the corpus (ground truth, present regardless of arm) and
check the response for the concrete identifiers that decision names: function calls,
PascalCase/camelCase symbols, scoped packages. This is the same identifier surface
the deterministic stub agent echoes from its context, so compliance is driven by
whether the governing decision was actually retrieved -- which is the point.

An optional LLM-judge backend can replace the identifier check for live runs
(graded honouring), but the deterministic identifier scorer is the default so the
offline pipeline is reproducible.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from membench.types import MemoryItem, Task

__all__ = [
    "ComplianceResult",
    "decision_keywords",
    "score_compliance",
    "score_compliance_for_task",
]

_IDENTIFIER = re.compile(
    r"[A-Za-z][A-Za-z0-9_]*\(\)"  # function calls: withAuditLog()
    r"|@[\w-]+/[\w-]+"  # scoped packages: @t3-oss/env-nextjs
    r"|\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b"  # PascalCase: DateRangePicker
    r"|\b[a-z]+(?:[A-Z][a-z0-9]*)+\b"  # camelCase: withAuditLog
)


@dataclass(frozen=True, slots=True)
class ComplianceResult:
    """Outcome of scoring decision compliance for one response."""

    total: int
    honored: int
    honored_ids: tuple[str, ...]
    rate: float  # honored / total; 1.0 when there are no governing decisions


def decision_keywords(decision_text: str) -> set[str]:
    """Extract the concrete identifiers a decision names (the honouring targets)."""
    found = _IDENTIFIER.findall(decision_text)
    keywords = set(found)
    # Credit "use withAuditLog" as well as the literal call "withAuditLog()".
    keywords.update(kw[:-2] for kw in found if kw.endswith("()"))
    return keywords


def score_compliance(
    response_text: str,
    governing_decisions: tuple[str, ...],
    corpus_by_id: Mapping[str, MemoryItem],
) -> ComplianceResult:
    """Score how many governing decisions the response honours.

    A decision is honoured if any identifier from its text appears in the response;
    if the decision names no identifier, the bare id is used as a last resort.
    """
    if not governing_decisions:
        return ComplianceResult(total=0, honored=0, honored_ids=(), rate=1.0)

    honored: list[str] = []
    for decision_id in governing_decisions:
        item = corpus_by_id.get(decision_id)
        keywords = decision_keywords(item.text) if item else set()
        if keywords:
            if any(keyword in response_text for keyword in keywords):
                honored.append(decision_id)
        elif decision_id in response_text:
            honored.append(decision_id)

    total = len(governing_decisions)
    return ComplianceResult(
        total=total,
        honored=len(honored),
        honored_ids=tuple(honored),
        rate=len(honored) / total,
    )


def score_compliance_for_task(response_text: str, task: Task) -> ComplianceResult:
    """Score compliance against a Task's gold decisions and corpus (convenience)."""
    return score_compliance(response_text, task.governing_decisions, task.corpus_by_id)
