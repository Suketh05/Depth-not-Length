"""LLM-judge compliance grading protocol (paper Section ``sec:grader`` / ``sec:gradervalidity``).

The paper grades *compliance* -- "the outcome-level judgment that the produced edit
honors the governing decision" -- with "an LLM-or-rubric-assisted judge that is
distinct from the answer models and sees the known invariant as reference but not
the arm identity" (Section ``sec:grader``/``sec:offline``; judge model, version, and
prompt order are the construct-validity surface of Section ``sec:gradervalidity``).
This module implements that protocol:

* :class:`JudgeCase` -- the exact evidence the judge sees: task, governing
  decision's invariant (reference), and the agent answer. The dataclass has **no
  arm field**, so arm-blindness holds by construction rather than by discipline
  (Section ``sec:grader``: the judge sees "the known invariant as reference but
  not the arm identity").
* :class:`JudgeConfig` -- records judge model, version, prompt order, and decoding
  configuration with every verdict, because "the friendly names ... are not
  reproducible identifiers; the camera-ready pins the exact snapshot strings and
  decoding configuration ... for both agent and grader" (Section ``sec:stats``).
* :class:`StubComplianceJudge` -- the deterministic, offline, rubric-based default:
  it checks the decision's invariant on the output via the same identifier surface
  as the rule-based scorer (:mod:`membench.metrics.compliance`), plus a negation
  guard that separates "invariant asserted" from "invariant enforced", the gaming
  threat Section ``sec:gradervalidity`` flags ("an arm could satisfy 'the invariant
  holds on the output' by *restating* the decision"). Verdicts are near-
  deterministic booleans, matching the bimodality reading of
  Figure ``fig:ecdf_compliance``.
* :class:`LLMComplianceJudge` -- the optional live backend reusing the
  :mod:`membench.agents.llm` clients (lazily constructed, so the offline pipeline
  never imports an SDK). Per Section ``sec:stats``, "the grader is a distinct model
  from the answer models" -- callers are responsible for picking a judge model
  disjoint from the answer models under test.
* :func:`judge_rule_agreement` / :func:`grader_agreement` -- the grader-validity
  machinery of Section ``sec:gradervalidity``: raw agreement rate and Cohen's
  kappa between the judge and the mechanical rule-based scorer on the same
  (task, decision) units, quantifying how far the judged construct drifts from
  the identifier-overlap proxy. Divergences feed the use factor
  :math:`\\kappa = P_{comply} / P_{ret}` of Equation ``eq:factor`` only through
  compliance, so this agreement check is the guard on that numerator.

The paper does not publish a numeric judge-vs-rule agreement value, so no paper
constant is asserted here; the machinery exists so a rerun can report one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

__all__ = [
    "PROMPT_ORDERS",
    "ComplianceJudge",
    "JudgeCase",
    "JudgeConfig",
    "JudgeVerdict",
]

#: Presentation orders for the reference invariant vs. the agent answer inside the
#: judge prompt. Section ``sec:gradervalidity`` names "swapped presentation order"
#: as a verdict-stability check, so the order is a first-class, recorded knob
#: rather than an accident of string formatting.
PROMPT_ORDERS: tuple[str, ...] = ("invariant_first", "answer_first")


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    """The judge's identity and decoding configuration, recorded with every verdict.

    Section ``sec:grader`` promises "judge model, version, and prompt order detailed
    in Section ``sec:gradervalidity``", and Section ``sec:stats`` requires pinning
    "the exact snapshot strings and decoding configuration (temperature, top-p,
    max tokens, ...) for both agent and grader". This object is that record.

    Parameters
    ----------
    model
        Judge model identifier (an exact snapshot string for live judges, or the
        stub rubric's label for offline runs).
    version
        Version tag of the judge prompt/rubric pair, bumped whenever the rubric
        wording changes so verdicts remain attributable.
    prompt_order
        One of :data:`PROMPT_ORDERS`: whether the reference invariant or the agent
        answer is presented first in the judge prompt.
    temperature
        Intended decoding temperature. The default ``0.0`` matches the paper's
        deterministic-grading intent; hosted models "can drift across snapshots
        even at temperature 0" (Section ``sec:stats``), which is why the model
        snapshot is recorded alongside it.
    max_tokens
        Maximum verdict length requested from a live judge.
    """

    model: str = "stub-rubric-judge"
    version: str = "rubric-v1"
    prompt_order: str = "invariant_first"
    temperature: float = 0.0
    max_tokens: int = 256

    def __post_init__(self) -> None:
        if self.prompt_order not in PROMPT_ORDERS:
            raise ValueError(
                f"prompt_order must be one of {PROMPT_ORDERS}, got {self.prompt_order!r}"
            )
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if not self.model:
            raise ValueError("model must be a non-empty string")


@dataclass(frozen=True, slots=True)
class JudgeCase:
    """Exactly what the judge is shown for one (task, governing decision) unit.

    Section ``sec:grader``: the judge "sees the known invariant as reference but
    not the arm identity". This dataclass is that contract made structural: there
    is deliberately **no arm/system/retriever field**, so a caller cannot leak the
    arm identity into the judge prompt without changing this type. (Section
    ``sec:gradervalidity`` still flags residual arm-inference from answer *style*;
    that threat lives in the answer text and cannot be typed away.)

    Parameters
    ----------
    task_query
        The task prompt the agent was answering.
    invariant_text
        The governing decision's text -- the reference invariant the rubric checks
        on the output (e.g. "the PII column does not appear in the export").
    answer_text
        The agent's produced answer/edit being graded.
    decision_id
        Optional ground-truth id of the governing decision (e.g. ``"D-002"``).
        Part of the *invariant reference*, not the arm identity; used only as the
        last-resort honouring target when the invariant names no code identifier,
        mirroring :func:`membench.metrics.compliance.score_compliance`.
    """

    task_query: str
    invariant_text: str
    answer_text: str
    decision_id: str | None = None

    def __post_init__(self) -> None:
        if not self.invariant_text and not self.decision_id:
            raise ValueError(
                "JudgeCase needs a reference: invariant_text and decision_id are both empty"
            )


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """One compliance verdict with its rationale and full judge provenance.

    The verdict is boolean because the paper reads the per-task compliance
    distribution as "sharply bimodal ... indicating the grader is making a
    near-deterministic invariant check rather than scoring a continuous quality"
    (Section ``sec:gradervalidity``, Figure ``fig:ecdf_compliance``).

    Parameters
    ----------
    compliant
        Whether the answer honours the governing decision's invariant.
    rationale
        The judge's stated reason (rubric step for the stub, model text for the
        live judge).
    config
        The :class:`JudgeConfig` that produced this verdict -- model, version,
        prompt order, decoding -- so every verdict is attributable
        (Section ``sec:stats``).
    parse_ok
        ``False`` when a live judge's output did not contain a parseable verdict
        and the grader failed *closed* to non-compliant. Kept on the record so
        parse failures are auditable rather than silently folded into misses.
    """

    compliant: bool
    rationale: str
    config: JudgeConfig
    parse_ok: bool = True


class ComplianceJudge(ABC):
    """The judging contract: (task, governing invariant, answer) -> verdict.

    Section ``sec:grader`` describes one judge role with two admissible
    implementations -- "an LLM-***or-rubric***-assisted judge" -- so the contract is
    an ABC (mirroring :class:`membench.retrieval.base.MemorySystem` and
    :class:`membench.agents.llm.base.LLMClient`) with the deterministic rubric stub
    and the live LLM backend as interchangeable subclasses. Everything downstream
    (per-task judging, agreement machinery) is written against this contract only.
    """

    @property
    @abstractmethod
    def config(self) -> JudgeConfig:
        """The judge's recorded identity: model, version, prompt order, decoding."""

    @abstractmethod
    def judge(self, case: JudgeCase) -> JudgeVerdict:
        """Grade one (task, governing decision, answer) unit into a verdict."""
        raise NotImplementedError
