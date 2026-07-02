"""Tests for the LLM-or-rubric compliance judge (paper sec:grader / sec:gradervalidity).

Goldens are hand-built:

* Stub-judge verdicts are read off the rubric by hand on constructed decisions
  (asserted / asserted-but-restated / not-asserted).
* Agreement-rate arithmetic is hand-computed as exact fractions.
* Wilson CI literals on the agreement proportion are quoted VERBATIM from the
  canonical golden pack (Section C.6 / E.1): Wilson-95% on (6, 8) and (8, 10),
  which the repo's ``metrics.factorization.wilson_interval`` produces.
"""

from __future__ import annotations

from membench.metrics.judge import (
    ComplianceVerdict,
    JudgeConfig,
    StubJudge,
)
from membench.types import MemoryItem, Scorer, SpecVariant, Task

# --- fixtures / builders ---------------------------------------------------

# A governing decision whose invariant names the identifiers withAuditLog() and
# DateRangePicker (the concrete honouring targets the rubric checks).
_DECISION_TEXT = (
    "All exports must call withAuditLog() before returning the DateRangePicker component."
)


def _task(
    governing: tuple[str, ...],
    *,
    corpus: tuple[MemoryItem, ...] | None = None,
    task_id: str = "t-1",
    depth: int = 1,
) -> Task:
    items = corpus if corpus is not None else (MemoryItem(item_id="D-001", text=_DECISION_TEXT),)
    return Task(
        task_id=task_id,
        dataset="dcbench",
        query="Add a CSV export endpoint for the users table.",
        repo_ref="repo@abc123",
        memory_corpus=items,
        governing_decisions=governing,
        depth=depth,
        spec_variant=SpecVariant.STRIPPED,
        scorer=Scorer.COMPLIANCE,
    )


# Three canonical answers against _DECISION_TEXT (D-001):
#   ENFORCED  -- asserts the invariant AND names a new identifier (a real edit).
#   RESTATED  -- asserts the invariant but names only decision identifiers (gaming).
#   ABSENT    -- names no invariant identifier at all.
_ANSWER_ENFORCED = "Wrapped the export in withAuditLog() inside exportUsers()."
_ANSWER_RESTATED = "The decision requires withAuditLog() on the DateRangePicker."
_ANSWER_ABSENT = "I updated the code as requested."


class TestStubJudgeGoldens:
    def test_enforced_answer_is_honored(self) -> None:
        # asserted (withAuditLog() present) AND enforced (answer names exportUsers(),
        # absent from the decision) => honored.
        verdict = StubJudge().grade(_task(("D-001",)), _ANSWER_ENFORCED)
        assert isinstance(verdict, ComplianceVerdict)
        assert verdict.total == 1
        assert verdict.honored_count == 1
        assert verdict.rate == 1.0
        assert verdict.compliant is True
        (decision,) = verdict.verdicts
        assert decision.decision_id == "D-001"
        assert decision.honored is True
        assert "exportUsers" in decision.rationale

    def test_restated_answer_is_gamed_not_honored(self) -> None:
        # asserted but NOT enforced (answer names only decision identifiers) => the
        # gaming guard of sec:gradervalidity marks it not honored.
        verdict = StubJudge().grade(_task(("D-001",)), _ANSWER_RESTATED)
        assert verdict.compliant is False
        (decision,) = verdict.verdicts
        assert decision.honored is False
        assert "gaming guard" in decision.rationale

    def test_absent_answer_is_not_asserted(self) -> None:
        verdict = StubJudge().grade(_task(("D-001",)), _ANSWER_ABSENT)
        assert verdict.compliant is False
        (decision,) = verdict.verdicts
        assert decision.honored is False
        assert "not asserted" in decision.rationale

    def test_gaming_guard_off_recovers_pure_assertion_rubric(self) -> None:
        # With enforcement disabled the restated answer is honored (invariant
        # asserted), matching the mechanical rule-based scorer exactly.
        judge = StubJudge(require_enforcement=False)
        verdict = judge.grade(_task(("D-001",)), _ANSWER_RESTATED)
        assert verdict.compliant is True

    def test_confidence_is_seed_reproducible_and_seed_sensitive(self) -> None:
        task = _task(("D-001",))
        conf_a = StubJudge(JudgeConfig(seed=7)).grade(task, _ANSWER_ENFORCED).verdicts[0].confidence
        conf_a2 = (
            StubJudge(JudgeConfig(seed=7)).grade(task, _ANSWER_ENFORCED).verdicts[0].confidence
        )
        conf_b = StubJudge(JudgeConfig(seed=8)).grade(task, _ANSWER_ENFORCED).verdicts[0].confidence
        assert conf_a == conf_a2  # deterministic under a fixed seed
        assert conf_a != conf_b  # different seed => different confidence
        assert 0.8 <= conf_a <= 1.0  # bimodal near-deterministic band
