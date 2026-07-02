"""Tests for the LLM-judge compliance grader (paper sec:grader / sec:gradervalidity)."""

from __future__ import annotations

from membench.metrics.compliance import score_compliance
from membench.metrics.judge import (
    JudgeCase,
    StubComplianceJudge,
)
from membench.types import MemoryItem, Scorer, SpecVariant, Task

# Same hand-built decision corpus shape as tests/test_metrics_compliance.py, so the
# judge and the rule-based scorer are exercised on identical ground truth.
AUDIT = "All exports must call withAuditLog() before streaming."
PICKER = "Use the DateRangePicker component for date ranges."
ADR = "see ADR-17"  # names no code identifier -> bare-id fallback territory


def _corpus() -> dict[str, MemoryItem]:
    return {
        "D-1": MemoryItem("D-1", AUDIT),
        "D-2": MemoryItem("D-2", PICKER),
        "D-3": MemoryItem("D-3", ADR),
    }


def _task(governing: tuple[str, ...], task_id: str = "t") -> Task:
    return Task(
        task_id=task_id,
        dataset="dcbench",
        query="Export the users table to CSV.",
        repo_ref="repo@abc123",
        memory_corpus=tuple(_corpus().values()),
        governing_decisions=governing,
        depth=2,
        spec_variant=SpecVariant.STRIPPED,
        scorer=Scorer.COMPLIANCE,
    )


def _case(invariant: str, answer: str, decision_id: str | None = "D-1") -> JudgeCase:
    return JudgeCase(
        task_query="Export the users table to CSV.",
        invariant_text=invariant,
        answer_text=answer,
        decision_id=decision_id,
    )


class TestStubJudgeGoldens:
    """Hand-built verdict goldens for the deterministic rubric judge."""

    def test_enforced_identifier_is_compliant(self) -> None:
        verdict = StubComplianceJudge().judge(
            _case(AUDIT, "I will call withAuditLog() before streaming the rows.")
        )
        assert verdict.compliant is True
        assert verdict.parse_ok is True
        assert "withAuditLog" in verdict.rationale

    def test_bare_mention_without_call_parens_is_compliant(self) -> None:
        # The rubric shares the rule scorer's identifier surface: the bare
        # camelCase form credits the invariant just like the literal call.
        verdict = StubComplianceJudge().judge(_case(AUDIT, "Wrap it with withAuditLog first."))
        assert verdict.compliant is True

    def test_absent_identifier_is_noncompliant(self) -> None:
        verdict = StubComplianceJudge().judge(_case(AUDIT, "Just stream the rows directly."))
        assert verdict.compliant is False
        assert "withAuditLog" in verdict.rationale

    def test_negated_mention_is_noncompliant_where_rule_scorer_credits_it(self) -> None:
        # The gaming threat of sec:gradervalidity: restating the invariant inside
        # an opt-out clause. The judge refuses credit; the rule-based scorer
        # (identifier overlap only) credits the same unit -- this is exactly the
        # construct gap judge_rule_agreement exists to measure.
        gamed = "We will skip withAuditLog() for this export."
        verdict = StubComplianceJudge().judge(_case(AUDIT, gamed))
        assert verdict.compliant is False
        assert "asserted" in verdict.rationale or "restated" in verdict.rationale

        ruled = score_compliance(gamed, ("D-1",), _corpus())
        assert ruled.rate == 1.0  # the mechanical scorer is gameable here

    def test_negated_plus_clean_sentence_is_compliant(self) -> None:
        # One negated mention does not poison a separate enforcing sentence.
        answer = "Do not rely on manual review alone. Wrap the export in withAuditLog()."
        assert StubComplianceJudge().judge(_case(AUDIT, answer)).compliant is True

    def test_negation_cue_scoped_to_its_own_sentence(self) -> None:
        # Cue in sentence 1 must not leak into sentence 2 (sentence-local guard).
        answer = "There is no time to waste. Call withAuditLog() before streaming."
        assert StubComplianceJudge().judge(_case(AUDIT, answer)).compliant is True

    def test_bare_id_fallback_honoured(self) -> None:
        verdict = StubComplianceJudge().judge(_case(ADR, "per D-3 we proceed", "D-3"))
        assert verdict.compliant is True
        assert "D-3" in verdict.rationale

    def test_bare_id_fallback_missing(self) -> None:
        verdict = StubComplianceJudge().judge(_case(ADR, "no reference here", "D-3"))
        assert verdict.compliant is False

    def test_empty_invariant_uses_bare_id_only(self) -> None:
        # A governing id missing from the corpus yields an empty invariant: only
        # the bare id can honour it, mirroring score_compliance's last resort.
        assert StubComplianceJudge().judge(_case("", "mentions D-9", "D-9")).compliant is True
        assert StubComplianceJudge().judge(_case("", "mentions nothing", "D-9")).compliant is False

    def test_deterministic_across_calls(self) -> None:
        judge = StubComplianceJudge()
        case = _case(AUDIT, "I will call withAuditLog() before streaming.")
        assert judge.judge(case) == judge.judge(case)

    def test_seed_is_a_paraphrase_knob_never_a_verdict_knob(self) -> None:
        # sec:gradervalidity demands verdict stability under rubric paraphrase;
        # for the stub the seed selects the phrasing, and the boolean verdict is
        # invariant across seeds by construction.
        cases = [
            _case(AUDIT, "I will call withAuditLog() before streaming."),
            _case(AUDIT, "We will skip withAuditLog() for this export."),
            _case(AUDIT, "Just stream the rows."),
            _case(ADR, "per D-3 we proceed", "D-3"),
        ]
        baseline = [StubComplianceJudge(seed=0).judge(c).compliant for c in cases]
        for seed in (1, 2, 3, 17):
            assert [StubComplianceJudge(seed=seed).judge(c).compliant for c in cases] == baseline

    def test_seeds_select_different_rationale_phrasings(self) -> None:
        case = _case(AUDIT, "We will skip withAuditLog() for this export.")
        r0 = StubComplianceJudge(seed=0).judge(case).rationale
        r1 = StubComplianceJudge(seed=1).judge(case).rationale
        assert r0 != r1  # paraphrase differs ...
        v0 = StubComplianceJudge(seed=0).judge(case).compliant
        v1 = StubComplianceJudge(seed=1).judge(case).compliant
        assert v0 == v1  # ... verdict does not
