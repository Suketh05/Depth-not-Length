"""Tests for the LLM-judge compliance grader (paper sec:grader / sec:gradervalidity)."""

from __future__ import annotations

import dataclasses

import pytest

from membench.agents.llm.base import LLMClient, LLMResponse
from membench.metrics.compliance import score_compliance, score_compliance_for_task
from membench.metrics.judge import (
    PROMPT_ORDERS,
    RUBRIC_SYSTEM_PROMPT,
    ComplianceJudge,
    JudgeCase,
    JudgeConfig,
    JudgeVerdict,
    LLMComplianceJudge,
    StubComplianceJudge,
    build_judge_prompt,
    grader_agreement,
    judge_rule_agreement,
    judge_task,
    parse_judge_verdict,
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

    def test_contraction_negations_are_cues(self) -> None:
        # "don't"/"won't" negate without containing the bare word "not".
        for answer in (
            "We don't need withAuditLog() for this internal export.",
            "This export won't call withAuditLog().",
            "We shouldn\u2019t bother with withAuditLog() here.",  # curly apostrophe
        ):
            assert StubComplianceJudge().judge(_case(AUDIT, answer)).compliant is False

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


class _FakeJudgeClient(LLMClient):
    """Offline LLMClient double returning a canned judge reply; records prompts."""

    def __init__(self, reply: str, model: str = "fake-judge-model") -> None:
        self.reply = reply
        self._model = model
        self.calls: list[tuple[str, str, int]] = []

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        self.calls.append((system, user, max_tokens))
        return LLMResponse(
            text=self.reply,
            input_tokens=(len(system) + len(user)) // 4,
            output_tokens=max(1, len(self.reply) // 4),
            dollars=0.0,
            model=self._model,
        )


class TestBuildJudgePrompt:
    def test_system_prompt_is_the_fixed_rubric(self) -> None:
        system, _ = build_judge_prompt(_case(AUDIT, "answer"), JudgeConfig())
        assert system == RUBRIC_SYSTEM_PROMPT
        # The rubric encodes the two protocol requirements verbatim-checkable
        # against sec:grader / sec:gradervalidity.
        assert "HONOURS the invariant" in system
        assert "restating" in system.lower()
        assert "not told which memory" in system

    def test_invariant_first_order(self) -> None:
        case = _case(AUDIT, "the produced answer text")
        _, user = build_judge_prompt(case, JudgeConfig(prompt_order="invariant_first"))
        assert user.index("TASK:") == 0
        assert user.index("GOVERNING DECISION") < user.index("AGENT ANSWER:")
        assert AUDIT in user
        assert "the produced answer text" in user

    def test_answer_first_order_swaps_blocks(self) -> None:
        # sec:gradervalidity names swapped presentation order as a stability
        # probe; the swap must be a config knob, not a caller reformat.
        case = _case(AUDIT, "the produced answer text")
        _, user = build_judge_prompt(case, JudgeConfig(prompt_order="answer_first"))
        assert user.index("TASK:") == 0
        assert user.index("AGENT ANSWER:") < user.index("GOVERNING DECISION")

    def test_decision_id_shown_only_when_present(self) -> None:
        with_id = build_judge_prompt(_case(AUDIT, "a", "D-1"), JudgeConfig())[1]
        without_id = build_judge_prompt(_case(AUDIT, "a", None), JudgeConfig())[1]
        assert "(id D-1)" in with_id
        assert "(id" not in without_id

    def test_prompt_carries_no_arm_identity(self) -> None:
        # The prompt is a pure render of the arm-blind JudgeCase: every line is
        # either fixed scaffolding or one of the case's own fields.
        case = _case(AUDIT, "answer body", "D-1")
        _, user = build_judge_prompt(case, JudgeConfig())
        scaffolding = {"TASK:", "GOVERNING DECISION (id D-1):", "AGENT ANSWER:", ""}
        payload = {case.task_query, case.invariant_text, case.answer_text}
        assert set(user.split("\n")) == scaffolding | payload


class TestParseJudgeVerdict:
    CONFIG = JudgeConfig(model="fake-judge-model", version="llm-judge-v1")

    def test_compliant_verdict_with_rationale(self) -> None:
        verdict = parse_judge_verdict(
            "VERDICT: COMPLIANT\nRATIONALE: the export is wrapped in withAuditLog().",
            self.CONFIG,
        )
        assert verdict.compliant is True
        assert verdict.parse_ok is True
        assert verdict.rationale == "the export is wrapped in withAuditLog()."
        assert verdict.config == self.CONFIG

    def test_noncompliant_verdict(self) -> None:
        verdict = parse_judge_verdict("VERDICT: NONCOMPLIANT\nRATIONALE: never applied.", self.CONFIG)
        assert verdict.compliant is False
        assert verdict.parse_ok is True

    def test_tolerates_case_and_hyphen_variants(self) -> None:
        assert parse_judge_verdict("verdict: compliant", self.CONFIG).compliant is True
        assert parse_judge_verdict("Verdict: Non-Compliant", self.CONFIG).compliant is False
        assert parse_judge_verdict("VERDICT: NON COMPLIANT", self.CONFIG).compliant is False

    def test_rationale_falls_back_to_full_text(self) -> None:
        verdict = parse_judge_verdict("VERDICT: COMPLIANT because it is applied", self.CONFIG)
        assert verdict.compliant is True
        assert "because it is applied" in verdict.rationale

    def test_unparseable_output_fails_closed(self) -> None:
        # A garbage grade must never be promoted to a success (it would inflate
        # the compliance numerator of eq:factor) -- but it stays flagged.
        verdict = parse_judge_verdict("I think this looks fine overall!", self.CONFIG)
        assert verdict.compliant is False
        assert verdict.parse_ok is False
        assert "unparseable" in verdict.rationale

    def test_empty_output_fails_closed(self) -> None:
        verdict = parse_judge_verdict("", self.CONFIG)
        assert verdict.compliant is False
        assert verdict.parse_ok is False


class TestLLMComplianceJudge:
    def test_judges_via_injected_client(self) -> None:
        client = _FakeJudgeClient("VERDICT: COMPLIANT\nRATIONALE: invariant enforced.")
        judge = LLMComplianceJudge(client)
        verdict = judge.judge(_case(AUDIT, "wraps the export in withAuditLog()"))
        assert verdict.compliant is True
        assert verdict.rationale == "invariant enforced."
        assert verdict.config.model == "fake-judge-model"

    def test_client_receives_rubric_and_arm_blind_prompt(self) -> None:
        client = _FakeJudgeClient("VERDICT: NONCOMPLIANT\nRATIONALE: not applied.")
        judge = LLMComplianceJudge(client, max_tokens=128)
        case = _case(AUDIT, "just stream the rows")
        judge.judge(case)
        (system, user, max_tokens), = client.calls
        assert system == RUBRIC_SYSTEM_PROMPT
        assert max_tokens == 128
        assert (system, user) == (RUBRIC_SYSTEM_PROMPT, build_judge_prompt(case, judge.config)[1])

    def test_prompt_order_knob_reaches_the_prompt(self) -> None:
        client = _FakeJudgeClient("VERDICT: COMPLIANT")
        judge = LLMComplianceJudge(client, prompt_order="answer_first")
        judge.judge(_case(AUDIT, "answer body"))
        _, user, _ = client.calls[0]
        assert user.index("AGENT ANSWER:") < user.index("GOVERNING DECISION")
        assert judge.config.prompt_order == "answer_first"

    def test_config_records_snapshot_identity(self) -> None:
        judge = LLMComplianceJudge(_FakeJudgeClient("VERDICT: COMPLIANT", model="judge-snap-01"))
        assert judge.config.model == "judge-snap-01"
        assert judge.config.version == "llm-judge-v1"
        assert judge.config.temperature == 0.0

    def test_instantiation_without_client_is_offline_safe(self) -> None:
        # Lazy construction: no SDK import, no keys needed until .judge() runs.
        judge = LLMComplianceJudge(model_key="claude")
        assert judge.config.model == "claude"
        assert isinstance(judge, ComplianceJudge)

    def test_works_end_to_end_with_judge_task(self) -> None:
        client = _FakeJudgeClient("VERDICT: COMPLIANT\nRATIONALE: fine.")
        result = judge_task(LLMComplianceJudge(client), "any response", _task(("D-1", "D-2")))
        assert result.rate == 1.0  # canned judge approves both units
        assert len(client.calls) == 2

    @pytest.mark.live
    def test_live_judge_smoke(self) -> None:
        # Requires network + ANTHROPIC_API_KEY; the offline suite never runs it.
        judge = LLMComplianceJudge(model_key="claude")
        verdict = judge.judge(
            _case(AUDIT, "I wrapped the export in withAuditLog() before streaming.")
        )
        assert verdict.parse_ok is True


class TestJudgeRuleAgreementArithmetic:
    """Hand-computed agreement goldens (never produced by running the module)."""

    def test_golden_four_of_five(self) -> None:
        # judge = [T, T, F, F, T]  (p_J = 3/5)
        # rule  = [T, F, F, F, T]  (p_R = 2/5)
        # agree on units 1, 3, 4, 5 -> p_o = 4/5 = 0.8
        # p_e = (3/5)(2/5) + (2/5)(3/5) = 6/25 + 6/25 = 12/25 = 0.48
        # kappa = (4/5 - 12/25) / (1 - 12/25) = (8/25) / (13/25) = 8/13
        #       = 0.6153846153846154 (exact fraction 8/13)
        result = judge_rule_agreement(
            [True, True, False, False, True], [True, False, False, False, True]
        )
        assert result.n == 5
        assert result.n_agree == 4
        assert result.agreement_rate == 0.8
        assert result.judge_positive_rate == 0.6
        assert result.rule_positive_rate == 0.4
        assert result.cohen_kappa == pytest.approx(8 / 13, abs=1e-12)

    def test_golden_seven_of_ten(self) -> None:
        # judge = [T,T,T,T,T,T,F,F,F,F]  (p_J = 6/10)
        # rule  = [T,T,T,T,F,F,F,F,T,F]  (p_R = 5/10)
        # agreements at units 1,2,3,4,7,8,10 -> p_o = 7/10
        # p_e = 0.6*0.5 + 0.4*0.5 = 0.5
        # kappa = (0.7 - 0.5) / (1 - 0.5) = 0.2 / 0.5 = 2/5 = 0.4
        judge = [True] * 6 + [False] * 4
        rule = [True, True, True, True, False, False, False, False, True, False]
        result = judge_rule_agreement(judge, rule)
        assert result.n_agree == 7
        assert result.agreement_rate == 0.7
        assert result.cohen_kappa == pytest.approx(0.4, abs=1e-12)

    def test_chance_floor_gives_zero_kappa(self) -> None:
        # judge = [T,T,F,F], rule = [T,F,T,F]: p_o = 2/4 = 0.5,
        # p_e = 0.5*0.5 + 0.5*0.5 = 0.5 -> kappa = 0/0.5 = 0.0 exactly.
        # Raw agreement 50% but ZERO beyond chance -- the reason kappa is
        # reported next to the raw rate.
        result = judge_rule_agreement([True, True, False, False], [True, False, True, False])
        assert result.agreement_rate == 0.5
        assert result.cohen_kappa == 0.0

    def test_perfect_agreement_is_kappa_one(self) -> None:
        flags = [True, False, True, True, False]
        result = judge_rule_agreement(flags, list(flags))
        assert result.agreement_rate == 1.0
        assert result.cohen_kappa == 1.0

    def test_degenerate_all_positive_agreement_is_kappa_one(self) -> None:
        # Both raters constant-True: p_o = 1 and p_e = 1 (0/0 in the raw
        # formula); defined as 1.0 at perfect agreement.
        result = judge_rule_agreement([True, True, True], [True, True, True])
        assert result.agreement_rate == 1.0
        assert result.cohen_kappa == 1.0

    def test_constant_but_opposite_raters_score_zero(self) -> None:
        # judge all-True vs rule all-False: p_o = 0, p_e = 1*0 + 0*1 = 0
        # -> kappa = (0-0)/(1-0) = 0.0 (no agreement, but none expected either).
        result = judge_rule_agreement([True, True, True], [False, False, False])
        assert result.agreement_rate == 0.0
        assert result.cohen_kappa == 0.0

    def test_systematic_disagreement_is_kappa_minus_one(self) -> None:
        # judge = [T,F], rule = [F,T]: p_o = 0, p_e = 0.5*0.5 + 0.5*0.5 = 0.5
        # -> kappa = (0 - 0.5)/(1 - 0.5) = -1.0 (worse than chance).
        result = judge_rule_agreement([True, False], [False, True])
        assert result.cohen_kappa == -1.0

    def test_empty_units_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero units"):
            judge_rule_agreement([], [])

    def test_misaligned_vectors_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="align"):
            judge_rule_agreement([True, False], [True])


class TestGraderAgreementEndToEnd:
    def test_hand_computed_agreement_over_tasks(self) -> None:
        # Four (task, decision) units, hand-graded:
        #   unit 1 (t1, D-1): "call withAuditLog() ..." in a clean sentence
        #             -> judge T, rule T (agree)
        #   unit 2 (t1, D-2): "skip DateRangePicker" -- negated restatement
        #             -> judge F, rule T (the gaming divergence; DISAGREE)
        #   unit 3 (t2, D-3): bare-id "per D-3" -> judge T, rule T (agree)
        #   unit 4 (t3, D-1): identifier absent -> judge F, rule F (agree)
        # p_o = 3/4 = 0.75; p_J = 2/4 = 0.5; p_R = 3/4 = 0.75
        # p_e = 0.5*0.75 + 0.5*0.25 = 0.375 + 0.125 = 0.5
        # kappa = (0.75 - 0.5)/(1 - 0.5) = 0.25/0.5 = 0.5 exactly
        graded = [
            (
                "I will call withAuditLog() before streaming. We can skip DateRangePicker here.",
                _task(("D-1", "D-2"), task_id="t1"),
            ),
            ("per D-3 we proceed", _task(("D-3",), task_id="t2")),
            ("Just stream the rows.", _task(("D-1",), task_id="t3")),
            ("vacuous tasks are skipped entirely", _task((), task_id="t4")),
        ]
        result = grader_agreement(StubComplianceJudge(), graded)
        assert result.n == 4  # t4 contributed no units
        assert result.n_agree == 3
        assert result.agreement_rate == 0.75
        assert result.judge_positive_rate == 0.5
        assert result.rule_positive_rate == 0.75
        assert result.cohen_kappa == 0.5

    def test_ungamed_batch_agrees_perfectly(self) -> None:
        graded = [
            ("call withAuditLog() and use DateRangePicker", _task(("D-1", "D-2"), task_id="t1")),
            ("nothing relevant", _task(("D-1",), task_id="t2")),
        ]
        result = grader_agreement(StubComplianceJudge(), graded)
        assert result.n == 3
        assert result.agreement_rate == 1.0
        assert result.cohen_kappa == 1.0

    def test_only_vacuous_tasks_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="zero units"):
            grader_agreement(StubComplianceJudge(), [("anything", _task(()))])
