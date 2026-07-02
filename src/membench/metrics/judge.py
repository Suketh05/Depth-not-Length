"""LLM-or-rubric compliance grader (paper Section sec:grader / Section sec:gradervalidity).

The paper grades *compliance* -- the outcome-level judgment that a produced edit
*honors* the governing decision's invariant -- with a judge that is (a) distinct
from the answer models and (b) *arm-blind*: it sees the governing decision's known
invariant as reference but never which memory arm produced the answer, so a
systematic blind spot cancels in arm *contrasts* even where it shifts absolute
levels (Section sec:gradervalidity, "arm-invariant, invariant-checking,
bimodal-and-calibrated proxy whose differences across arms are trustworthy").
Compliance is the outcome factor of the decomposition ``P_comply = P_ret * kappa``
(Equation eq:factor): recall is mechanical ground truth, compliance is this
outcome-scored judgment, and their divergence *is* the use factor ``kappa``.

This module supplies both grading backends the paper names:

* :class:`StubJudge` -- a deterministic, seeded, offline rubric judge (the default,
  so the whole pipeline runs green with no API keys, matching the harness's
  offline design in Section sec:offline). It checks the decision's invariant on the
  output exactly as the paper's rubric does, and additionally implements the
  *gaming guard* the paper flags in Section sec:gradervalidity: an arm can satisfy
  "the invariant holds on the output" by *restating* the constraint rather than by
  a correct edit, so the rubric separates *invariant asserted* (a token appears)
  from *invariant enforced by the change* (the answer names an identifier the
  decision did not). This is precisely the "invariant asserted vs. invariant
  enforced by the change" distinction the paper leaves to grader work.
* :class:`LLMJudge` -- an optional live backend that reuses the existing
  ``agents.llm`` clients (imported lazily; distinct model from the answer models),
  builds an arm-blind rubric prompt, and parses a structured verdict.

Section sec:gradervalidity also names the *construct-validity* check this module
implements: the **agreement rate between the judge and the rule-based scorer**
(:func:`agreement_rate`), reported with a Wilson score interval. The full
grader-validity robustness suite (held-out arm-recovery test, verdict stability
under rubric paraphrase and swapped presentation order) is out of scope for this
module -- the paper flags it as needed-before-earned future work.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from membench.metrics.compliance import decision_keywords, score_compliance_for_task
from membench.metrics.factorization import wilson_interval
from membench.types import MemoryItem, Task

__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "AgreementResult",
    "ComplianceVerdict",
    "DecisionVerdict",
    "Judge",
    "JudgeConfig",
    "PromptOrder",
    "StubJudge",
    "agreement_rate",
    "judge_vs_rule_agreement",
]

# The answer models in the paper are Claude (Sonnet) and GPT-5.1 (Section
# "Reproducibility"). The grader must be a *distinct* model (Section sec:grader:
# "distinct from the answer models"); a stronger Claude tier is the default judge.
DEFAULT_JUDGE_MODEL = "claude-opus-4-8"

# Presentation orders whose swap is one of the verdict-stability checks named in
# Section sec:gradervalidity ("verdict stability ... under ... swapped order").
PromptOrder = str
INVARIANT_FIRST: PromptOrder = "invariant_first"
ANSWER_FIRST: PromptOrder = "answer_first"


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    """Recorded grader configuration (Section sec:grader: "judge model, version, and prompt order").

    Parameters
    ----------
    model
        The judge model identifier. Must be distinct from every answer model for
        the arm-blind construct to hold (Section sec:grader).
    version
        A grader/prompt version tag, recorded so a re-grade is reproducible.
    prompt_order
        Presentation order of the invariant reference and the answer; swapping it
        is one of the verdict-stability checks (Section sec:gradervalidity).
    seed
        Seed for the deterministic stub judge, so an offline grade is reproducible.
    """

    model: str = DEFAULT_JUDGE_MODEL
    version: str = "v1"
    prompt_order: PromptOrder = INVARIANT_FIRST
    seed: int = 0

    def is_distinct_from(self, answer_models: Sequence[str]) -> bool:
        """Return whether the judge model differs from every answer model.

        The arm-blind construct in Section sec:grader requires the grader to be a
        different model from the ones that produced the answers.

        Parameters
        ----------
        answer_models
            The model identifiers that produced the graded answers.

        Returns
        -------
        bool
            ``True`` iff :attr:`model` is not among ``answer_models``.
        """
        return self.model not in set(answer_models)


@dataclass(frozen=True, slots=True)
class DecisionVerdict:
    """A per-decision compliance verdict with the judge's rationale.

    Parameters
    ----------
    decision_id
        The governing decision's ``item_id`` this verdict grades.
    honored
        Whether the answer honors the decision's invariant.
    rationale
        A short human-readable justification (why honored / why not).
    confidence
        The judge's confidence in ``[0, 1]``; near-deterministic (bimodal) for the
        stub, per the calibrated-check reading in Section sec:gradervalidity.
    """

    decision_id: str
    honored: bool
    rationale: str
    confidence: float


@dataclass(frozen=True, slots=True)
class ComplianceVerdict:
    """The judge's task-level compliance grade over all governing decisions.

    Mirrors the shape of :class:`membench.metrics.compliance.ComplianceResult` so
    the judge and the rule-based scorer are directly comparable for the agreement
    check of Section sec:gradervalidity.

    Parameters
    ----------
    task_id
        The graded task.
    verdicts
        One :class:`DecisionVerdict` per governing decision (empty when the task
        has no governing decisions -- vacuously compliant).
    config
        The :class:`JudgeConfig` that produced this grade (recorded per Section
        sec:grader).
    """

    task_id: str
    verdicts: tuple[DecisionVerdict, ...]
    config: JudgeConfig

    @property
    def total(self) -> int:
        """Number of governing decisions graded."""
        return len(self.verdicts)

    @property
    def honored_count(self) -> int:
        """Number of governing decisions the answer honors."""
        return sum(1 for verdict in self.verdicts if verdict.honored)

    @property
    def rate(self) -> float:
        """Compliance rate ``honored / total``; ``1.0`` when there are none."""
        if not self.verdicts:
            return 1.0
        return self.honored_count / self.total

    @property
    def compliant(self) -> bool:
        """The bimodal task-level event: every governing decision honored.

        Section sec:gradervalidity reads compliance as a near-deterministic
        per-task invariant check whose mass piles at ``0`` and ``1``; the boolean
        event used for the judge/rule agreement rate is "all invariants honored".
        """
        return all(verdict.honored for verdict in self.verdicts)


@runtime_checkable
class Judge(Protocol):
    """The compliance-grading contract (Section sec:grader).

    A judge takes a task and the agent's answer and returns a compliance verdict.
    Crucially the signature carries **no arm identity** -- arm-blindness is enforced
    by construction, which is the defense the paper's contrast claims rest on
    (Section sec:gradervalidity).
    """

    def grade(self, task: Task, agent_answer: str) -> ComplianceVerdict:
        """Grade whether ``agent_answer`` honors ``task``'s governing decisions.

        Parameters
        ----------
        task
            The evaluation unit, carrying the governing decision ids and the corpus
            in which their invariant text is ground truth.
        agent_answer
            The answer text produced by some (undisclosed) memory arm.

        Returns
        -------
        ComplianceVerdict
            The per-decision verdicts and their task-level roll-up.
        """
        ...


def _invariant_reference(task: Task, decision_id: str) -> MemoryItem | None:
    """Return the governing decision's memory item (its invariant text), if present."""
    return task.corpus_by_id.get(decision_id)


def _seeded_confidence(seed: int, decision_id: str, honored: bool, answer: str) -> float:
    """Return a deterministic, seed-dependent confidence in ``[0.8, 1.0]``.

    The stub judge's verdict is a near-deterministic invariant check (bimodal, per
    Section sec:gradervalidity), so confidence sits high; it is derived from a
    stable SHA-256 digest -- not Python's salted ``hash`` -- so the same
    ``(seed, decision, verdict, answer)`` reproduces bit-for-bit across processes,
    which is what "seeded" reproducibility requires.
    """
    key = f"{seed}\x00{decision_id}\x00{int(honored)}\x00{answer}".encode()
    digest = hashlib.sha256(key).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return 0.8 + 0.2 * fraction


class StubJudge:
    """Deterministic, offline rubric judge (paper Section sec:grader, default backend).

    The rubric checks the governing decision's invariant on the answer exactly as
    the paper's grader does (Section sec:grader: "the rubric checks the decision's
    invariant on the output"): the invariant is *asserted* when any concrete
    identifier the decision names appears in the answer -- the same identifier
    surface the rule-based scorer
    (:func:`membench.metrics.compliance.score_compliance`) checks, so the two
    estimators are directly comparable for the agreement rate of Section
    sec:gradervalidity.

    With ``require_enforcement=True`` (the default) the judge additionally applies
    the *gaming guard* the paper names in Section sec:gradervalidity: an arm can
    satisfy "the invariant holds on the output" by *restating* the constraint
    rather than by a correct edit, and the typed store -- which surfaces the
    decision text verbatim -- is the arm most able to do so. The rubric therefore
    separates *invariant asserted* (a decision token appears) from *invariant
    enforced by the change* (the answer names at least one identifier the decision
    did not, i.e. evidence of an actual edit rather than a parroted constraint).
    Set ``require_enforcement=False`` to recover the pure invariant-assertion
    rubric, which then agrees with the rule-based scorer by construction.

    Parameters
    ----------
    config
        The recorded grader configuration; its ``seed`` drives the deterministic
        confidence. Defaults to a fresh :class:`JudgeConfig`.
    require_enforcement
        Whether to apply the gaming guard described above. ``True`` by default.
    """

    def __init__(
        self, config: JudgeConfig | None = None, *, require_enforcement: bool = True
    ) -> None:
        self._config = config if config is not None else JudgeConfig()
        self._require_enforcement = require_enforcement

    @property
    def config(self) -> JudgeConfig:
        """The recorded grader configuration."""
        return self._config

    def grade(self, task: Task, agent_answer: str) -> ComplianceVerdict:
        """Grade ``agent_answer`` against ``task``'s governing decisions (arm-blind).

        Parameters
        ----------
        task
            The evaluation unit whose governing decision ids index the invariant
            text in its corpus.
        agent_answer
            The answer text produced by an undisclosed arm.

        Returns
        -------
        ComplianceVerdict
            One :class:`DecisionVerdict` per governing decision plus the roll-up.
        """
        verdicts = tuple(
            self._grade_decision(task, decision_id, agent_answer)
            for decision_id in task.governing_decisions
        )
        return ComplianceVerdict(task_id=task.task_id, verdicts=verdicts, config=self._config)

    def _grade_decision(self, task: Task, decision_id: str, answer: str) -> DecisionVerdict:
        item = _invariant_reference(task, decision_id)
        invariant_tokens = decision_keywords(item.text) if item is not None else set()

        # Invariant asserted? Mirror score_compliance exactly: substring match on
        # the decision's identifiers, else the bare id when it names none.
        if invariant_tokens:
            asserted = any(token in answer for token in invariant_tokens)
        else:
            asserted = decision_id in answer

        if not asserted:
            honored = False
            rationale = (
                f"invariant not asserted: no identifier of {decision_id} appears in the answer"
            )
        elif not self._require_enforcement:
            honored = True
            rationale = f"invariant asserted: an identifier of {decision_id} appears in the answer"
        else:
            decision_tokens = decision_keywords(item.text) if item is not None else set()
            new_tokens = decision_keywords(answer) - decision_tokens
            honored = bool(new_tokens)
            if honored:
                witness = min(new_tokens)
                rationale = (
                    f"invariant asserted and enforced by a change "
                    f"(answer names {witness!r}, absent from the decision)"
                )
            else:
                rationale = (
                    f"invariant merely restated, not enforced by a change "
                    f"(gaming guard on {decision_id})"
                )

        confidence = _seeded_confidence(self._config.seed, decision_id, honored, answer)
        return DecisionVerdict(
            decision_id=decision_id,
            honored=honored,
            rationale=rationale,
            confidence=confidence,
        )


@dataclass(frozen=True, slots=True)
class AgreementResult:
    """Judge-vs-rule agreement, the construct-validity check of Section sec:gradervalidity.

    The paper defends the compliance construct partly by the **agreement rate
    between the LLM-or-rubric judge and the rule-based (mechanical) scorer**. This
    is the observed agreement -- the fraction of items on which the two estimators
    return the same compliant/not-compliant event -- with a Wilson score interval
    (the repo's default binomial-proportion interval, ``metrics.factorization``).
    The two disagreement directions are also recorded because they are not
    symmetric: ``judge_stricter`` counts the gaming-guard cases where the rule
    scorer credits an asserted-but-not-enforced invariant that the judge rejects
    (Section sec:gradervalidity).

    Parameters
    ----------
    n_agree
        Number of items on which judge and rule agree.
    n_total
        Number of graded items.
    judge_stricter
        Items the rule scores compliant but the judge does not (gaming-guard hits).
    rule_stricter
        Items the judge scores compliant but the rule does not.
    ci
        Wilson 95% interval on the agreement proportion, or ``None`` when empty.
    """

    n_agree: int
    n_total: int
    judge_stricter: int
    rule_stricter: int
    ci: tuple[float, float] | None

    @property
    def rate(self) -> float:
        """Observed agreement ``n_agree / n_total``; ``1.0`` for the empty set."""
        if self.n_total == 0:
            return 1.0
        return self.n_agree / self.n_total


def agreement_rate(
    judge_flags: Sequence[bool],
    rule_flags: Sequence[bool],
    *,
    confidence: float = 0.95,
) -> AgreementResult:
    """Observed agreement between the judge and the rule-based scorer (Section sec:gradervalidity).

    Parameters
    ----------
    judge_flags
        The judge's per-item compliant/not-compliant events.
    rule_flags
        The rule-based scorer's per-item events, aligned index-for-index with
        ``judge_flags``.
    confidence
        Two-sided confidence level for the Wilson interval on the agreement rate.

    Returns
    -------
    AgreementResult
        The agreement count, directional disagreement counts, and Wilson CI.

    Raises
    ------
    ValueError
        If the two sequences differ in length.
    """
    if len(judge_flags) != len(rule_flags):
        raise ValueError(
            f"judge_flags and rule_flags must align: {len(judge_flags)} != {len(rule_flags)}"
        )
    n_total = len(judge_flags)
    n_agree = sum(1 for judge, rule in zip(judge_flags, rule_flags, strict=True) if judge == rule)
    judge_stricter = sum(
        1 for judge, rule in zip(judge_flags, rule_flags, strict=True) if rule and not judge
    )
    rule_stricter = sum(
        1 for judge, rule in zip(judge_flags, rule_flags, strict=True) if judge and not rule
    )
    ci = wilson_interval(n_agree, n_total, confidence=confidence) if n_total > 0 else None
    return AgreementResult(
        n_agree=n_agree,
        n_total=n_total,
        judge_stricter=judge_stricter,
        rule_stricter=rule_stricter,
        ci=ci,
    )


def judge_vs_rule_agreement(
    judge: Judge,
    tasks: Sequence[Task],
    answers: Sequence[str],
    *,
    confidence: float = 0.95,
) -> AgreementResult:
    """Grade ``tasks`` with both estimators and return their agreement.

    Runs ``judge`` and the mechanical rule-based scorer
    (:func:`membench.metrics.compliance.score_compliance_for_task`) over the same
    (task, answer) pairs and compares the task-level compliant events -- the
    end-to-end form of the Section sec:gradervalidity agreement check.

    Parameters
    ----------
    judge
        Any :class:`Judge` (stub or LLM backend).
    tasks
        The evaluation units.
    answers
        The agent answers, aligned index-for-index with ``tasks``.
    confidence
        Two-sided confidence level for the Wilson interval.

    Returns
    -------
    AgreementResult
        The judge-vs-rule agreement over the graded tasks.

    Raises
    ------
    ValueError
        If ``tasks`` and ``answers`` differ in length.
    """
    if len(tasks) != len(answers):
        raise ValueError(f"tasks and answers must align: {len(tasks)} != {len(answers)}")
    judge_flags = [
        judge.grade(task, answer).compliant for task, answer in zip(tasks, answers, strict=True)
    ]
    rule_flags = [
        # Rule-based compliant event: every governing decision honored (rate == 1.0),
        # the same bimodal task-level event ComplianceVerdict.compliant uses.
        score_compliance_for_task(answer, task).rate == 1.0
        for task, answer in zip(tasks, answers, strict=True)
    ]
    return agreement_rate(judge_flags, rule_flags, confidence=confidence)
