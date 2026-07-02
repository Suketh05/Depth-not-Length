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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from membench.types import MemoryItem, Task

__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "ComplianceVerdict",
    "DecisionVerdict",
    "Judge",
    "JudgeConfig",
    "PromptOrder",
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
