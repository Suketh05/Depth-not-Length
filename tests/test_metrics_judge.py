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

import pytest

from membench.agents.llm.base import LLMClient, LLMResponse
from membench.metrics.judge import (
    ANSWER_FIRST,
    DEFAULT_JUDGE_MODEL,
    INVARIANT_FIRST,
    ComplianceVerdict,
    DecisionVerdict,
    Judge,
    JudgeConfig,
    LLMJudge,
    StubJudge,
    agreement_rate,
    build_judge_prompt,
    judge_vs_rule_agreement,
    parse_verdict,
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


class TestAgreementRateArithmetic:
    def test_perfect_agreement(self) -> None:
        judge = [True, True, False, True, False]
        rule = [True, True, False, True, False]
        result = agreement_rate(judge, rule)
        assert result.n_agree == 5
        assert result.n_total == 5
        assert result.rate == 1.0  # 5/5
        assert result.judge_stricter == 0
        assert result.rule_stricter == 0

    def test_six_of_eight_agreement_wilson_verbatim(self) -> None:
        # Hand-counted: indices 1 and 7 disagree (rule compliant, judge not) => the
        # gaming-guard direction; 6 of 8 agree.
        judge = [True, False, False, True, False, False, True, False]
        rule = [True, True, False, True, False, False, True, True]
        result = agreement_rate(judge, rule)
        assert result.n_agree == 6
        assert result.n_total == 8
        assert result.rate == 0.75  # 6/8
        assert result.judge_stricter == 2  # rule says yes, judge says no
        assert result.rule_stricter == 0
        # Wilson 95% on (6, 8) -- golden pack Section C.6 / E.1, verbatim.
        assert result.ci == (0.40927543031016883, 0.9285207872478909)

    def test_eight_of_ten_agreement_wilson_verbatim(self) -> None:
        judge = [True] * 10
        rule = [True] * 8 + [False, False]
        result = agreement_rate(judge, rule)
        assert result.n_agree == 8
        assert result.n_total == 10
        assert result.rate == 0.8  # 8/10
        assert result.rule_stricter == 2  # judge says yes, rule says no
        assert result.judge_stricter == 0
        # Wilson 95% on (8, 10) -- golden pack Section C.6, verbatim.
        assert result.ci == (0.49016247153664183, 0.9433178485456247)

    def test_total_disagreement_is_a_chance_floor(self) -> None:
        result = agreement_rate([True, False, True, False], [False, True, False, True])
        assert result.n_agree == 0
        assert result.rate == 0.0  # 0/4

    def test_empty_agreement_is_vacuously_one_with_no_ci(self) -> None:
        result = agreement_rate([], [])
        assert result.n_total == 0
        assert result.rate == 1.0
        assert result.ci is None

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="align"):
            agreement_rate([True, False], [True])


class TestJudgeVsRuleAgreementEndToEnd:
    def test_gaming_guard_drives_the_only_disagreement(self) -> None:
        # Enforced and absent answers agree with the rule scorer; the restated
        # (gamed) answer is the one place the judge is stricter than the rule.
        tasks = [_task(("D-001",)), _task(("D-001",)), _task(("D-001",))]
        answers = [_ANSWER_ENFORCED, _ANSWER_RESTATED, _ANSWER_ABSENT]
        result = judge_vs_rule_agreement(StubJudge(), tasks, answers)
        assert result.n_total == 3
        assert result.n_agree == 2  # enforced + absent agree; restated does not
        assert result.rate == pytest.approx(2 / 3)
        assert result.judge_stricter == 1
        assert result.rule_stricter == 0

    def test_pure_assertion_rubric_agrees_with_rule_by_construction(self) -> None:
        # Disabling the gaming guard makes the stub judge identical to the
        # rule-based scorer, so agreement is 1.0 on the same three answers.
        tasks = [_task(("D-001",)), _task(("D-001",)), _task(("D-001",))]
        answers = [_ANSWER_ENFORCED, _ANSWER_RESTATED, _ANSWER_ABSENT]
        result = judge_vs_rule_agreement(StubJudge(require_enforcement=False), tasks, answers)
        assert result.n_agree == 3
        assert result.rate == 1.0

    def test_tasks_answers_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="align"):
            judge_vs_rule_agreement(StubJudge(), [_task(("D-001",))], [])


class _CannedClient(LLMClient):
    """An offline LLMClient that returns a fixed reply (no network, not live)."""

    def __init__(self, reply: str, model: str = "judge-test") -> None:
        self._reply = reply
        self._model = model
        self.last_system: str | None = None
        self.last_user: str | None = None

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> LLMResponse:
        self.last_system = system
        self.last_user = user
        return LLMResponse(
            text=self._reply,
            input_tokens=(len(system) + len(user)) // 4,
            output_tokens=max(1, len(self._reply) // 4),
            dollars=0.0,
            model=self._model,
        )


class TestLLMJudgeOffline:
    def test_llm_judge_parses_injected_json_verdict(self) -> None:
        client = _CannedClient('{"honored": true, "confidence": 0.9, "rationale": "enforced"}')
        judge = LLMJudge(client)
        assert isinstance(judge, Judge)
        verdict = judge.grade(_task(("D-001",)), _ANSWER_ENFORCED)
        (decision,) = verdict.verdicts
        assert decision.honored is True
        assert decision.confidence == 0.9
        assert decision.rationale == "enforced"
        assert verdict.compliant is True

    def test_llm_judge_config_defaults_to_client_model(self) -> None:
        judge = LLMJudge(_CannedClient("{}", model="claude-opus-4-8"))
        assert judge.config.model == "claude-opus-4-8"

    def test_llm_judge_prompt_is_arm_blind_and_carries_the_invariant(self) -> None:
        client = _CannedClient('{"honored": false, "rationale": "n/a"}')
        LLMJudge(client, JudgeConfig(prompt_order=INVARIANT_FIRST)).grade(
            _task(("D-001",)), _ANSWER_RESTATED
        )
        assert client.last_system is not None and client.last_user is not None
        blob = (client.last_system + client.last_user).lower()
        # The invariant reference and candidate answer are both present ...
        assert "withauditlog" in blob
        assert "the decision requires" in blob
        # ... but no arm identity leaks into the grading prompt (arm-blindness).
        for arm in ("brief", "bm25", "tfidf", "dense", "raptor", "rerank_ce", "hybrid_rrf"):
            assert arm not in blob


class TestBuildJudgePromptOrder:
    def test_invariant_first_places_reference_before_answer(self) -> None:
        system, user = build_judge_prompt(
            _DECISION_TEXT, "D-001", _ANSWER_ENFORCED, prompt_order=INVARIANT_FIRST
        )
        assert isinstance(system, str) and system  # non-empty rubric instructions
        assert user.index("invariant reference") < user.index("Candidate answer")

    def test_answer_first_swaps_the_two_blocks(self) -> None:
        _, user = build_judge_prompt(
            _DECISION_TEXT, "D-001", _ANSWER_ENFORCED, prompt_order=ANSWER_FIRST
        )
        # The presentation-order lever of sec:gradervalidity: same content, swapped.
        assert user.index("Candidate answer") < user.index("invariant reference")


class TestParseVerdict:
    def test_strict_json(self) -> None:
        verdict = parse_verdict('{"honored": true, "confidence": 0.9, "rationale": "ok"}', "D-1")
        assert verdict == DecisionVerdict("D-1", True, "ok", 0.9)

    def test_json_embedded_in_prose(self) -> None:
        verdict = parse_verdict('Sure: {"honored": false, "rationale": "restated"} done.', "D-1")
        assert verdict.honored is False
        assert verdict.rationale == "restated"
        assert verdict.confidence == 1.0  # default when the field is omitted

    def test_confidence_is_clamped_to_unit_interval(self) -> None:
        assert parse_verdict('{"honored": true, "confidence": 1.5}', "D-1").confidence == 1.0
        assert parse_verdict('{"honored": true, "confidence": -0.2}', "D-1").confidence == 0.0

    def test_keyword_fallback_when_no_json(self) -> None:
        assert parse_verdict("The invariant is not honored here.", "D-1").honored is False
        assert parse_verdict("This clearly honors the constraint.", "D-1").honored is True

    def test_unparseable_reply_fails_closed(self) -> None:
        # A judge whose reply cannot be read must NOT be credited as compliant.
        verdict = parse_verdict("<<< garbage with no verdict >>>", "D-1")
        assert verdict.honored is False
        assert verdict.confidence == 0.0
        assert "unparseable" in verdict.rationale


# A supersession chain: D-new supersedes D-old (the paper's governance edge). Only
# the CURRENT decision governs, so honouring the superseded one is NOT compliance.
_SUPER_CORPUS = (
    MemoryItem(
        item_id="D-old",
        text="Route all reads through OldGateway using fetchDirect().",
    ),
    MemoryItem(
        item_id="D-new",
        text="Route all reads through NewGateway using fetchViaGateway().",
        metadata={"node_type": "constraint", "supersedes": "D-old"},
    ),
)


class TestSupersessionConflict:
    def test_honouring_the_superseded_decision_is_not_compliant(self) -> None:
        # The answer honours the stale D-old (OldGateway/fetchDirect) but the task
        # is governed by the current D-new -- whose invariant is never asserted.
        task = _task(("D-new",), corpus=_SUPER_CORPUS, task_id="super-1")
        answer = "Kept the read path on OldGateway via fetchDirect()."
        verdict = StubJudge().grade(task, answer)
        assert verdict.compliant is False
        assert "not asserted" in verdict.verdicts[0].rationale

    def test_honouring_the_current_decision_is_compliant(self) -> None:
        task = _task(("D-new",), corpus=_SUPER_CORPUS, task_id="super-2")
        answer = "Migrated reads to NewGateway using fetchViaGateway() in migrateReads()."
        verdict = StubJudge().grade(task, answer)
        assert verdict.compliant is True


class TestArmInvariance:
    def test_grade_is_a_function_of_content_only(self) -> None:
        # sec:gradervalidity's arm-blind construct: the verdict depends only on
        # (task, answer). The signature admits no arm, so the same answer grades
        # identically every time -- deterministic and label-free.
        task = _task(("D-001",))
        first = StubJudge(JudgeConfig(seed=3)).grade(task, _ANSWER_ENFORCED)
        second = StubJudge(JudgeConfig(seed=3)).grade(task, _ANSWER_ENFORCED)
        assert first == second
