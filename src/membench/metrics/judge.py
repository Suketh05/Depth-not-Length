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

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from membench.metrics.compliance import decision_keywords

__all__ = [
    "PROMPT_ORDERS",
    "ComplianceJudge",
    "JudgeCase",
    "JudgeConfig",
    "JudgeVerdict",
    "StubComplianceJudge",
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


# Two semantically identical phrasings per rubric outcome. The active phrasing is
# selected by the stub judge's seed, so reruns with different seeds are a built-in
# rubric-paraphrase probe: Section ``sec:gradervalidity`` demands "verdict stability
# under rubric paraphrase", and for the stub that stability holds by construction
# (the seed can only change the rationale wording, never the boolean verdict).
_RATIONALE_TEMPLATES: dict[str, tuple[str, str]] = {
    "honored": (
        "invariant identifier {target!r} is applied in the answer",
        "the answer enforces the invariant via {target!r}",
    ),
    "bare_id": (
        "decision id {target!r} is referenced (invariant names no code identifier)",
        "no code identifier in the invariant; the answer cites decision id {target!r}",
    ),
    "missing": (
        "no invariant identifier ({target}) appears in the answer",
        "the answer never applies any invariant identifier ({target})",
    ),
    "no_reference": (
        "decision id {target!r} is never referenced (invariant names no code identifier)",
        "no code identifier in the invariant and the answer omits decision id {target!r}",
    ),
    "negated": (
        "invariant identifier {target!r} is only mentioned in a negated/opt-out clause "
        "(asserted, not enforced)",
        "every mention of {target!r} sits in a negation/opt-out clause; the invariant is "
        "restated but not applied",
    ),
}

# Sentence-level negation / opt-out cues. Section ``sec:gradervalidity`` flags the
# gaming threat that "an arm could satisfy 'the invariant holds on the output' by
# *restating* the decision (echoing the constraint or adding a guard comment)
# rather than by a correct edit" and notes that "cleanly separating 'invariant
# asserted' from 'invariant enforced by the change' is left to future grader
# work". This guard is the deterministic first cut: a mention of an invariant
# identifier counts only if at least one of its host sentences carries no
# negation cue. The rule-based scorer has no such guard, which is exactly the
# construct gap :func:`judge_rule_agreement` measures.
_NEGATION_CUES = re.compile(
    r"\b(?:not|never|no|without|skip(?:s|ped|ping)?|ignor(?:e|es|ed|ing)|"
    r"avoid(?:s|ed|ing)?|omit(?:s|ted|ting)?|bypass(?:es|ed|ing)?|"
    r"instead\s+of|rather\s+than)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"[.!?;\n]+")


def _sentences(text: str) -> list[str]:
    """Split text into rough sentences (grading unit for the negation guard)."""
    return [part for part in _SENTENCE_SPLIT.split(text) if part.strip()]


def _enforced(keyword: str, answer_text: str) -> bool | None:
    """Classify a keyword's mentions: True enforced, False only-negated, None absent.

    A keyword is *enforced* when at least one sentence containing it carries no
    negation/opt-out cue; *asserted-but-negated* when it appears only inside
    negated sentences ("we will skip withAuditLog()"); absent otherwise.
    """
    hosts = [sentence for sentence in _sentences(answer_text) if keyword in sentence]
    if not hosts:
        return None
    return any(not _NEGATION_CUES.search(sentence) for sentence in hosts)


class StubComplianceJudge(ComplianceJudge):
    """Deterministic offline rubric judge -- the default grader for the pipeline.

    Implements the rubric half of the paper's "LLM-or-rubric-assisted judge"
    (Section ``sec:grader``): it "checks the decision's invariant on the output",
    using the same concrete-identifier surface as the rule-based scorer
    (:func:`membench.metrics.compliance.decision_keywords`: function calls,
    PascalCase/camelCase symbols, scoped packages), with the bare decision id as
    the last-resort target when the invariant names no identifier. The rubric is
    a pure function of the case text, so verdicts are exactly reproducible
    offline -- no API, no sampling -- and boolean, matching the near-deterministic
    bimodal grading the paper reports (Section ``sec:gradervalidity``,
    Figure ``fig:ecdf_compliance``).

    Parameters
    ----------
    seed
        Selects which of the equivalent rationale phrasings is emitted (a
        deterministic rubric-paraphrase knob, see :data:`_RATIONALE_TEMPLATES`).
        The verdict boolean is independent of the seed by construction.
    prompt_order
        Recorded in :attr:`config` for parity with live judges; the stub reads
        the structured :class:`JudgeCase` directly, so order cannot affect it.
    """

    def __init__(self, seed: int = 0, *, prompt_order: str = "invariant_first") -> None:
        self._seed = seed
        self._config = JudgeConfig(
            model="stub-rubric-judge",
            version="rubric-v1",
            prompt_order=prompt_order,
            temperature=0.0,
            max_tokens=256,
        )

    @property
    def config(self) -> JudgeConfig:
        """The stub's recorded identity (model label, rubric version, order)."""
        return self._config

    def _phrase(self, outcome: str, target: str) -> str:
        templates = _RATIONALE_TEMPLATES[outcome]
        return templates[self._seed % len(templates)].format(target=target)

    def judge(self, case: JudgeCase) -> JudgeVerdict:
        """Apply the invariant rubric to the answer; deterministic and seeded.

        Rubric steps (in order):

        1. Extract the invariant's concrete identifiers via
           :func:`membench.metrics.compliance.decision_keywords`.
        2. If any identifier occurs in a non-negated sentence of the answer, the
           invariant is *enforced* -> compliant.
        3. If identifiers occur but only inside negated/opt-out sentences, the
           invariant is merely *asserted* -> non-compliant (the gaming guard of
           Section ``sec:gradervalidity``; the rule-based scorer credits this).
        4. If the invariant names no identifier, fall back to the bare decision
           id (when provided), mirroring the rule-based scorer's last resort.
        5. Otherwise the invariant is not honoured.
        """
        keywords = decision_keywords(case.invariant_text)
        if keywords:
            negated: list[str] = []
            for keyword in sorted(keywords):
                status = _enforced(keyword, case.answer_text)
                if status is True:
                    return JudgeVerdict(
                        compliant=True,
                        rationale=self._phrase("honored", keyword),
                        config=self._config,
                    )
                if status is False:
                    negated.append(keyword)
            if negated:
                return JudgeVerdict(
                    compliant=False,
                    rationale=self._phrase("negated", ", ".join(negated)),
                    config=self._config,
                )
            targets = ", ".join(sorted(keywords))
            return JudgeVerdict(
                compliant=False,
                rationale=self._phrase("missing", targets),
                config=self._config,
            )
        if case.decision_id is not None and case.decision_id in case.answer_text:
            return JudgeVerdict(
                compliant=True,
                rationale=self._phrase("bare_id", case.decision_id),
                config=self._config,
            )
        return JudgeVerdict(
            compliant=False,
            rationale=self._phrase("no_reference", case.decision_id or "<none>"),
            config=self._config,
        )
