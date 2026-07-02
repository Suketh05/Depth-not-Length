r"""LLM-judge compliance grading protocol (paper Section ``sec:grader`` / ``sec:gradervalidity``).

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
  :math:`\kappa = P_{comply} / P_{ret}` of Equation ``eq:factor`` only through
  compliance, so this agreement check is the guard on that numerator.

The paper does not publish a numeric judge-vs-rule agreement value, so no paper
constant is asserted here; the machinery exists so a rerun can report one.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from membench.metrics.compliance import decision_keywords, score_compliance
from membench.types import Task

if TYPE_CHECKING:  # imported lazily at runtime: the offline pipeline never needs it
    from membench.agents.llm.base import LLMClient

__all__ = [
    "PROMPT_ORDERS",
    "RUBRIC_SYSTEM_PROMPT",
    "ComplianceJudge",
    "GraderAgreement",
    "JudgeCase",
    "JudgeConfig",
    "JudgeVerdict",
    "LLMComplianceJudge",
    "StubComplianceJudge",
    "TaskJudgeResult",
    "build_judge_prompt",
    "grader_agreement",
    "judge_rule_agreement",
    "judge_task",
    "parse_judge_verdict",
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
    r"instead\s+of|rather\s+than)\b"
    r"|n['\u2019]t\b",  # contractions: don't / won't / shouldn't (either apostrophe)
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


@dataclass(frozen=True, slots=True)
class TaskJudgeResult:
    """Judge-graded compliance for one task, shaped like ``ComplianceResult``.

    Same accounting contract as the rule-based
    :class:`membench.metrics.compliance.ComplianceResult` (``rate = compliant /
    total``, vacuous ``1.0`` with no governing decisions), so the two graders are
    drop-in comparable per task -- which is what the grader-validity agreement
    machinery of Section ``sec:gradervalidity`` compares.

    Parameters
    ----------
    total
        Number of governing decisions graded.
    compliant
        Number judged honoured.
    compliant_ids
        Ids of the honoured decisions, in governing order.
    rate
        ``compliant / total``; ``1.0`` when there are no governing decisions.
    verdicts
        Per-decision ``(decision_id, verdict)`` pairs, in governing order, so the
        rationale and judge provenance of every unit remain auditable.
    """

    total: int
    compliant: int
    compliant_ids: tuple[str, ...]
    rate: float
    verdicts: tuple[tuple[str, JudgeVerdict], ...]


def judge_task(judge: ComplianceJudge, response_text: str, task: Task) -> TaskJudgeResult:
    """Judge every governing decision of a task against one response.

    Builds one arm-blind :class:`JudgeCase` per governing decision -- the task
    query, the decision's ground-truth text from the corpus (present regardless of
    arm, exactly like the rule-based scorer's lookup), and the response -- and
    aggregates the boolean verdicts into a per-task compliance rate
    (Section ``sec:grader``). A decision id missing from the corpus contributes an
    empty invariant, so only the bare-id fallback can honour it, mirroring
    :func:`membench.metrics.compliance.score_compliance`.
    """
    if not task.governing_decisions:
        return TaskJudgeResult(total=0, compliant=0, compliant_ids=(), rate=1.0, verdicts=())

    corpus = task.corpus_by_id
    verdicts: list[tuple[str, JudgeVerdict]] = []
    compliant_ids: list[str] = []
    for decision_id in task.governing_decisions:
        item = corpus.get(decision_id)
        case = JudgeCase(
            task_query=task.query,
            invariant_text=item.text if item is not None else "",
            answer_text=response_text,
            decision_id=decision_id,
        )
        verdict = judge.judge(case)
        verdicts.append((decision_id, verdict))
        if verdict.compliant:
            compliant_ids.append(decision_id)

    total = len(task.governing_decisions)
    return TaskJudgeResult(
        total=total,
        compliant=len(compliant_ids),
        compliant_ids=tuple(compliant_ids),
        rate=len(compliant_ids) / total,
        verdicts=tuple(verdicts),
    )


#: The arm-blind judging rubric given to a live judge as the system prompt.
#: It operationalises Section ``sec:grader``: check the decision's invariant on
#: the output ("the rubric checks the decision's invariant on the output"), demand
#: enforcement rather than restatement (the gaming threat of Section
#: ``sec:gradervalidity``), and never speculate about which memory system produced
#: the answer (arm-blindness). The fixed VERDICT/RATIONALE shape keeps parsing
#: mechanical and the verdict boolean (bimodal grading, Figure
#: ``fig:ecdf_compliance``).
RUBRIC_SYSTEM_PROMPT = (
    "You are a compliance grader for coding-agent outputs. You will see a TASK, a "
    "GOVERNING DECISION stating an invariant that any acceptable answer must honour, "
    "and an AGENT ANSWER. Decide whether the answer HONOURS the invariant: the "
    "invariant must be enforced by the proposed change itself. Merely restating the "
    "invariant, quoting it, or promising to skip it does NOT count as honouring it. "
    "Judge only invariant compliance, not overall answer quality or correctness. "
    "You are not told which memory or retrieval system produced the answer; do not "
    "try to guess or reward any particular style. "
    "Reply in exactly two lines:\n"
    "VERDICT: COMPLIANT or VERDICT: NONCOMPLIANT\n"
    "RATIONALE: <one sentence>"
)

_VERDICT_PATTERN = re.compile(r"VERDICT\s*:\s*(NON[\s-]?COMPLIANT|COMPLIANT)", re.IGNORECASE)
_RATIONALE_PATTERN = re.compile(r"RATIONALE\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def build_judge_prompt(case: JudgeCase, config: JudgeConfig) -> tuple[str, str]:
    """Render the (system, user) judge prompt for one case, arm-blind by input.

    The user message contains only the three :class:`JudgeCase` fields (plus the
    ground-truth decision id when present), laid out per ``config.prompt_order``:
    ``"invariant_first"`` presents the governing decision before the agent answer,
    ``"answer_first"`` swaps them. Section ``sec:gradervalidity`` names swapped
    presentation order as a stability probe, so the order is an explicit, recorded
    argument instead of a formatting accident. The task always leads, since both
    blocks are read relative to it.
    """
    id_line = f" (id {case.decision_id})" if case.decision_id else ""
    invariant_block = f"GOVERNING DECISION{id_line}:\n{case.invariant_text}"
    answer_block = f"AGENT ANSWER:\n{case.answer_text}"
    if config.prompt_order == "invariant_first":
        first, second = invariant_block, answer_block
    else:
        first, second = answer_block, invariant_block
    user = f"TASK:\n{case.task_query}\n\n{first}\n\n{second}"
    return RUBRIC_SYSTEM_PROMPT, user


def parse_judge_verdict(text: str, config: JudgeConfig) -> JudgeVerdict:
    """Parse a live judge's raw output into a verdict; fail *closed* on garbage.

    Accepts ``VERDICT: COMPLIANT`` / ``VERDICT: NONCOMPLIANT`` (case-insensitive,
    tolerating ``NON-COMPLIANT``/``NON COMPLIANT``) anywhere in the text and takes
    the rest of the ``RATIONALE:`` line as the rationale. Output with no parseable
    verdict is graded non-compliant with ``parse_ok=False``: an unparseable grade
    must never be silently promoted to a success (that would inflate the
    compliance numerator of Equation ``eq:factor``), but it stays flagged on the
    record so parse failures are auditable rather than folded into real misses.
    """
    match = _VERDICT_PATTERN.search(text)
    if match is None:
        snippet = text.strip()[:200]
        return JudgeVerdict(
            compliant=False,
            rationale=f"unparseable judge output (failed closed): {snippet!r}",
            config=config,
            parse_ok=False,
        )
    compliant = match.group(1).upper() == "COMPLIANT"
    rationale_match = _RATIONALE_PATTERN.search(text)
    rationale = rationale_match.group(1).strip() if rationale_match else text.strip()
    return JudgeVerdict(compliant=compliant, rationale=rationale, config=config)


class LLMComplianceJudge(ComplianceJudge):
    """Live LLM judge backend (the "LLM-assisted" half of Section ``sec:grader``).

    Reuses the :class:`membench.agents.llm.base.LLMClient` contract, so any
    configured backend (Anthropic, OpenAI, or an injected fake in tests) can act
    as the judge. When no client is injected, one is built lazily from
    :func:`membench.agents.combos.make_llm_client` with ``offline=False`` on the
    first :meth:`judge` call -- construction, not just SDK import, is deferred, so
    merely instantiating this class stays offline-safe. Real judging requires
    network and keys and is exercised only by ``live``-marked tests.

    Two protocol obligations from the paper are the caller's responsibility and
    are recorded rather than enforced here: the judge must be "distinct from the
    answer models" (Sections ``sec:grader``, ``sec:stats``), and ``config.model``
    should be an exact snapshot string in reported runs. ``config.temperature``
    records the intended decoding configuration; the shared ``LLMClient.complete``
    contract does not expose a temperature knob, so backends run at their
    default and the recorded value documents intent (Section ``sec:stats``).

    Parameters
    ----------
    client
        An :class:`~membench.agents.llm.base.LLMClient` to use as the judge. When
        ``None``, ``model_key`` is resolved lazily via ``make_llm_client``.
    model_key
        Model-grid key (e.g. ``"claude"``) used for lazy construction and, when no
        client is injected, as the recorded model identity.
    prompt_order
        Presentation order for :func:`build_judge_prompt`.
    version
        Judge prompt/rubric version tag recorded on every verdict.
    max_tokens
        Verdict-length cap passed to the client.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        model_key: str = "claude",
        prompt_order: str = "invariant_first",
        version: str = "llm-judge-v1",
        max_tokens: int = 256,
    ) -> None:
        self._client = client
        self._model_key = model_key
        self._config = JudgeConfig(
            model=client.model_name if client is not None else model_key,
            version=version,
            prompt_order=prompt_order,
            temperature=0.0,
            max_tokens=max_tokens,
        )

    @property
    def config(self) -> JudgeConfig:
        """The judge's recorded identity (model, version, prompt order, decoding)."""
        return self._config

    def _ensure_client(self) -> LLMClient:
        if self._client is None:
            # Lazy: only a live judging call pays the import and needs keys.
            from membench.agents.combos import make_llm_client

            self._client = make_llm_client(self._model_key, offline=False)
            self._config = JudgeConfig(
                model=self._client.model_name,
                version=self._config.version,
                prompt_order=self._config.prompt_order,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
        return self._client

    def judge(self, case: JudgeCase) -> JudgeVerdict:
        """Prompt the judge model with the arm-blind case and parse its verdict."""
        client = self._ensure_client()
        system, user = build_judge_prompt(case, self._config)
        response = client.complete(system, user, max_tokens=self._config.max_tokens)
        return parse_judge_verdict(response.text, self._config)


@dataclass(frozen=True, slots=True)
class GraderAgreement:
    """Judge-vs-rule-based agreement over matched (task, decision) units.

    The grader-validity summary of Section ``sec:gradervalidity``: how often the
    outcome-level judge and the mechanical identifier-overlap scorer
    (:mod:`membench.metrics.compliance`) return the same boolean on the same
    units, plus Cohen's kappa to correct raw agreement for chance (two fixed
    raters -- the 2-rater specialisation of the Fleiss statistic in
    :mod:`membench.metrics.agreement`).

    Parameters
    ----------
    n
        Number of matched units compared.
    n_agree
        Units where both graders returned the same boolean.
    agreement_rate
        ``n_agree / n`` (observed agreement, :math:`p_o`).
    cohen_kappa
        :math:`(p_o - p_e) / (1 - p_e)` with :math:`p_e = p_J p_R +
        (1-p_J)(1-p_R)`; ``1.0`` is perfect, ``0.0`` chance-level. Defined as
        ``1.0`` at perfect agreement (which subsumes the degenerate
        :math:`p_e = 1` case, only reachable when :math:`p_o = 1`).
    judge_positive_rate
        Fraction of units the judge graded compliant (:math:`p_J`).
    rule_positive_rate
        Fraction of units the rule-based scorer graded compliant (:math:`p_R`).
    """

    n: int
    n_agree: int
    agreement_rate: float
    cohen_kappa: float
    judge_positive_rate: float
    rule_positive_rate: float


def judge_rule_agreement(
    judge_flags: Sequence[bool], rule_flags: Sequence[bool]
) -> GraderAgreement:
    r"""Agreement rate and Cohen's kappa between judge and rule-based verdicts.

    Standard two-rater Cohen (1960) kappa on aligned boolean verdict vectors:

    .. math::

        p_o = \frac{\#\{i : J_i = R_i\}}{n},\qquad
        p_e = p_J p_R + (1-p_J)(1-p_R),\qquad
        \kappa = \frac{p_o - p_e}{1 - p_e}

    where :math:`p_J, p_R` are the two raters' compliant-rates. Perfect agreement
    returns :math:`\kappa = 1` directly (this also covers the :math:`p_e = 1`
    degeneracy: :math:`p_e = 1` forces both marginals onto the same constant
    label, hence :math:`p_o = 1`). This quantifies the construct gap Section
    ``sec:gradervalidity`` worries about -- the judge measuring "honoring the
    decision" vs. the rule scorer's surface identifier overlap.

    Raises
    ------
    ValueError
        If the vectors are empty (agreement undefined; a chance-agreement floor
        must never be invented from zero evidence) or of different lengths.
    """
    if len(judge_flags) != len(rule_flags):
        raise ValueError(
            f"verdict vectors must align: {len(judge_flags)} judge vs {len(rule_flags)} rule"
        )
    n = len(judge_flags)
    if n == 0:
        raise ValueError("agreement is undefined on zero units")

    n_agree = sum(1 for j, r in zip(judge_flags, rule_flags, strict=True) if bool(j) == bool(r))
    p_o = n_agree / n
    p_judge = sum(map(bool, judge_flags)) / n
    p_rule = sum(map(bool, rule_flags)) / n
    p_e = p_judge * p_rule + (1.0 - p_judge) * (1.0 - p_rule)
    kappa = 1.0 if p_o == 1.0 else (p_o - p_e) / (1.0 - p_e)
    return GraderAgreement(
        n=n,
        n_agree=n_agree,
        agreement_rate=p_o,
        cohen_kappa=kappa,
        judge_positive_rate=p_judge,
        rule_positive_rate=p_rule,
    )


def grader_agreement(judge: ComplianceJudge, graded: Sequence[tuple[str, Task]]) -> GraderAgreement:
    """Run judge and rule-based scorer over (response, task) pairs and compare.

    The comparison unit is one (task, governing decision) pair: the judge grades
    it via :func:`judge_task` and the rule-based scorer via
    :func:`membench.metrics.compliance.score_compliance` on the identical
    response, corpus, and gold ids. Tasks with no governing decisions are skipped
    (both graders define them as vacuously compliant, so counting them would
    inflate agreement with units that grade nothing).

    Raises
    ------
    ValueError
        If no (task, decision) units remain after skipping vacuous tasks.
    """
    judge_flags: list[bool] = []
    rule_flags: list[bool] = []
    for response_text, task in graded:
        if not task.governing_decisions:
            continue
        judged = judge_task(judge, response_text, task)
        ruled = score_compliance(response_text, task.governing_decisions, task.corpus_by_id)
        judged_ids = set(judged.compliant_ids)
        ruled_ids = set(ruled.honored_ids)
        for decision_id in task.governing_decisions:
            judge_flags.append(decision_id in judged_ids)
            rule_flags.append(decision_id in ruled_ids)
    return judge_rule_agreement(judge_flags, rule_flags)
