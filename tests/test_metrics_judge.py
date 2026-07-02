"""Tests for the LLM-judge compliance grader (paper sec:grader / sec:gradervalidity)."""

from __future__ import annotations

import dataclasses

import pytest

from membench.metrics.compliance import score_compliance, score_compliance_for_task
from membench.metrics.judge import (
    PROMPT_ORDERS,
    ComplianceJudge,
    JudgeCase,
    JudgeConfig,
    JudgeVerdict,
    LLMComplianceJudge,
    StubComplianceJudge,
    judge_task,
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


class TestProtocolConformance:
    """The judging contract of sec:grader, checked structurally."""

    def test_both_backends_implement_the_judge_contract(self) -> None:
        assert issubclass(StubComplianceJudge, ComplianceJudge)
        assert issubclass(LLMComplianceJudge, ComplianceJudge)
        assert isinstance(StubComplianceJudge(), ComplianceJudge)

    def test_abstract_contract_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            ComplianceJudge()  # type: ignore[abstract]

    def test_judge_case_is_arm_blind_by_construction(self) -> None:
        # sec:grader: the judge "sees the known invariant as reference but not
        # the arm identity". The case type has exactly these fields and nothing
        # that could carry an arm/system/retriever identity.
        field_names = {f.name for f in dataclasses.fields(JudgeCase)}
        assert field_names == {"task_query", "invariant_text", "answer_text", "decision_id"}
        for leak in ("arm", "system", "retriever", "model"):
            assert leak not in field_names

    def test_judge_case_requires_a_reference(self) -> None:
        with pytest.raises(ValueError, match="reference"):
            JudgeCase(task_query="q", invariant_text="", answer_text="a", decision_id=None)

    def test_verdict_records_full_judge_provenance(self) -> None:
        # sec:grader promises judge model, version, and prompt order are
        # detailed; every verdict carries them.
        verdict = StubComplianceJudge(prompt_order="answer_first").judge(_case(AUDIT, "text"))
        assert verdict.config.model == "stub-rubric-judge"
        assert verdict.config.version == "rubric-v1"
        assert verdict.config.prompt_order == "answer_first"
        assert verdict.config.temperature == 0.0

    def test_verdicts_and_cases_are_immutable(self) -> None:
        verdict = StubComplianceJudge().judge(_case(AUDIT, "text"))
        with pytest.raises(dataclasses.FrozenInstanceError):
            verdict.compliant = True  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            _case(AUDIT, "text").answer_text = "other"  # type: ignore[misc]

    def test_prompt_orders_constant(self) -> None:
        assert PROMPT_ORDERS == ("invariant_first", "answer_first")


class TestJudgeConfigValidation:
    def test_rejects_unknown_prompt_order(self) -> None:
        with pytest.raises(ValueError, match="prompt_order"):
            JudgeConfig(prompt_order="answer_last")

    def test_rejects_negative_temperature(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            JudgeConfig(temperature=-0.1)

    def test_rejects_nonpositive_max_tokens(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            JudgeConfig(max_tokens=0)

    def test_rejects_empty_model(self) -> None:
        with pytest.raises(ValueError, match="model"):
            JudgeConfig(model="")

    def test_defaults_are_the_offline_stub_identity(self) -> None:
        config = JudgeConfig()
        assert config.model == "stub-rubric-judge"
        assert config.prompt_order == "invariant_first"
        assert config.temperature == 0.0


class TestJudgeTask:
    def test_rate_arithmetic_one_of_two(self) -> None:
        # D-1 enforced, D-2 never mentioned: rate = 1/2 = 0.5 (hand-computed).
        result = judge_task(
            StubComplianceJudge(),
            "I will call withAuditLog() before streaming.",
            _task(("D-1", "D-2")),
        )
        assert result.total == 2
        assert result.compliant == 1
        assert result.compliant_ids == ("D-1",)
        assert result.rate == 0.5
        assert [decision_id for decision_id, _ in result.verdicts] == ["D-1", "D-2"]
        assert all(isinstance(v, JudgeVerdict) for _, v in result.verdicts)

    def test_vacuous_task_is_fully_compliant(self) -> None:
        # Same convention as the rule-based ComplianceResult: no governing
        # decisions -> rate 1.0 on zero units.
        result = judge_task(StubComplianceJudge(), "anything", _task(()))
        assert result.total == 0
        assert result.compliant == 0
        assert result.rate == 1.0
        assert result.verdicts == ()

    def test_governing_id_missing_from_corpus_uses_bare_id_fallback(self) -> None:
        assert judge_task(StubComplianceJudge(), "per D-404 we act", _task(("D-404",))).rate == 1.0
        assert judge_task(StubComplianceJudge(), "unrelated", _task(("D-404",))).rate == 0.0

    def test_parity_with_rule_scorer_on_ungamed_responses(self) -> None:
        # On answers with no negated restatements the two graders share the
        # identifier surface, so per-task rates coincide exactly.
        task = _task(("D-1", "D-2", "D-3"))
        for response in (
            "I will call withAuditLog() and use DateRangePicker, per D-3.",
            "Use DateRangePicker for the date range.",
            "Nothing relevant here.",
        ):
            judged = judge_task(StubComplianceJudge(), response, task)
            ruled = score_compliance_for_task(response, task)
            assert judged.rate == ruled.rate
            assert judged.compliant_ids == ruled.honored_ids
