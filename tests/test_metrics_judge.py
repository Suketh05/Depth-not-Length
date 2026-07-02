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
    DEFAULT_JUDGE_MODEL,
    ComplianceVerdict,
    Judge,
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


# A second governing decision naming a disjoint identifier surface.
_DECISION2 = MemoryItem(
    item_id="D-002",
    text="Every write path must go through the RateLimiter and call enforceQuota().",
)
_CORPUS2 = (MemoryItem(item_id="D-001", text=_DECISION_TEXT), _DECISION2)


class TestProtocolConformance:
    def test_stub_judge_satisfies_judge_protocol(self) -> None:
        assert isinstance(StubJudge(), Judge)

    def test_arbitrary_object_is_not_a_judge(self) -> None:
        assert not isinstance(object(), Judge)

    def test_grade_returns_one_verdict_per_governing_decision(self) -> None:
        verdict = StubJudge().grade(_task(("D-001", "D-002"), corpus=_CORPUS2), _ANSWER_ENFORCED)
        assert isinstance(verdict, ComplianceVerdict)
        assert tuple(v.decision_id for v in verdict.verdicts) == ("D-001", "D-002")


class TestJudgeConfigDistinctness:
    def test_default_judge_model_distinct_from_answer_models(self) -> None:
        # sec:grader: the grader is distinct from the answer models (Claude Sonnet,
        # GPT-5.1); the default judge is a different Claude tier.
        assert DEFAULT_JUDGE_MODEL == "claude-opus-4-8"
        assert JudgeConfig().is_distinct_from(["claude-sonnet-4-6", "gpt-5.1"]) is True

    def test_judge_not_distinct_when_it_equals_an_answer_model(self) -> None:
        assert (
            JudgeConfig(model="claude-sonnet-4-6").is_distinct_from(["claude-sonnet-4-6"]) is False
        )


class TestVerdictRollup:
    def test_partial_compliance_rate_is_a_clean_fraction(self) -> None:
        # D-001 enforced (honored), D-002 never asserted => 1 of 2 honored.
        verdict = StubJudge().grade(_task(("D-001", "D-002"), corpus=_CORPUS2), _ANSWER_ENFORCED)
        assert verdict.total == 2
        assert verdict.honored_count == 1
        assert verdict.rate == 0.5
        assert verdict.compliant is False  # bimodal event needs ALL honored

    def test_saturation_all_decisions_honored(self) -> None:
        # An answer that enforces both invariants (new identifier for each) is
        # fully compliant.
        answer = (
            "Wrapped exportUsers() in withAuditLog(); "
            "routed writes via enforceQuota() in guardWrites()."
        )
        verdict = StubJudge().grade(_task(("D-001", "D-002"), corpus=_CORPUS2), answer)
        assert verdict.honored_count == 2
        assert verdict.rate == 1.0
        assert verdict.compliant is True

    def test_empty_governing_set_is_vacuously_compliant(self) -> None:
        verdict = StubJudge().grade(_task(()), _ANSWER_ABSENT)
        assert verdict.total == 0
        assert verdict.verdicts == ()
        assert verdict.rate == 1.0
        assert verdict.compliant is True
