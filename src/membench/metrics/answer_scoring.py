r"""Answer-vs-gold scoring for the public-benchmark sweep (Section ``sec:pubbench``).

The unified-harness sweep of Section~\ref{sec:pubbench} (Table~\ref{tab:stdbench},
Figure~\ref{fig:stdbench}) runs every arm through each public benchmark's *own*
scoring protocol and reports one headline number per benchmark:

* **LoCoMo** -- LLM-judge accuracy (``tab:stdbench`` row ``LoCoMo``; ``Brief 87.6``).
  A judge model, *distinct from the answer model* and blind to the arm identity,
  reads the gold reference answer and returns a binary correct / incorrect verdict;
  accuracy is the mean verdict (Section~\ref{sec:grader}, judge protocol).
* **DMR** -- deep-memory *fact F1* (``tab:stdbench`` row ``DMR``; ``Brief 94.2``): the
  harmonic mean of precision and recall over the set of gold facts recovered in the
  answer.

This module implements the answer-side scorers those rows need, which the repo did
not previously have (loaders emit ``gold_answers`` but nothing compared an answer to
gold). SWE-ContextBench *resolution* -- the third headline row of ``tab:stdbench``
(``Brief 47.3``) -- is a test-execution outcome, not an answer-vs-gold comparison, so
it lives in the sibling module :mod:`membench.metrics.resolution`.

Design decisions
----------------
* **Deterministic fallback + judge hook.** The offline pipeline must be reproducible
  with no network (see ``pyproject`` "core install is offline-capable"), so the
  default LoCoMo judge is a deterministic token-overlap scorer
  (:class:`TokenOverlapJudge`); a live LLM judge is supplied through the
  :class:`AnswerJudge` protocol without changing the aggregation.
* **Arm-blind by construction.** The :class:`AnswerJudge` call signature carries only
  ``(question, gold_answer, predicted_answer)`` and never the arm identity, encoding
  the "sees the known invariant as reference but not the arm identity" requirement of
  Section~\ref{sec:grader} in the type, not merely the prompt.

Normalization
-------------
Text is normalized with the standard SQuAD/HotpotQA answer normalization
[Rajpurkar2016]_ (lowercase, strip punctuation, drop the articles ``a``/``an``/``the``,
collapse whitespace) before any token or fact comparison, so that scores are invariant
to casing, punctuation, and article noise -- the same normalization LoCoMo and DMR use.

References
----------
.. [Rajpurkar2016] Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016).
   *SQuAD: 100,000+ Questions for Machine Comprehension of Text.* EMNLP 2016.
   (The ``normalize_answer`` / token-F1 protocol reused across extractive-QA and
   conversational-memory benchmarks.)
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = [
    "AnswerJudge",
    "AnswerRecord",
    "FactF1",
    "JudgeAccuracy",
    "JudgeVerdict",
    "TokenOverlapJudge",
    "answer_exact_match",
    "answer_tokens",
    "fact_f1",
    "llm_judge_accuracy",
    "normalize_answer",
    "token_f1",
]

# Articles stripped by the SQuAD normalization (matched as whole words only).
_ARTICLES = re.compile(r"\b(?:a|an|the)\b")
_PUNCTUATION = frozenset(string.punctuation)


def normalize_answer(text: str) -> str:
    """Return the SQuAD-normalized form of an answer string.

    Lowercases, removes punctuation, drops the articles ``a``/``an``/``the`` as whole
    words, and collapses runs of whitespace to single spaces (SQuAD normalization
    [Rajpurkar2016]_). Idempotent: ``normalize_answer(normalize_answer(x)) ==
    normalize_answer(x)``.

    Parameters
    ----------
    text
        Raw answer text.

    Returns
    -------
    str
        Normalized answer; the empty string for input that is all punctuation,
        articles, and whitespace.
    """
    lowered = text.lower()
    without_punctuation = "".join(ch for ch in lowered if ch not in _PUNCTUATION)
    without_articles = _ARTICLES.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def answer_tokens(text: str) -> list[str]:
    """Return the whitespace tokens of :func:`normalize_answer` applied to ``text``.

    Parameters
    ----------
    text
        Raw answer text.

    Returns
    -------
    list of str
        Normalized tokens (empty list when the normalized answer is empty).
    """
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


def answer_exact_match(prediction: str, gold: str) -> bool:
    """Return whether ``prediction`` and ``gold`` are equal after normalization.

    The SQuAD/HotpotQA exact-match criterion [Rajpurkar2016]_: two answers match iff
    their :func:`normalize_answer` forms are identical.

    Parameters
    ----------
    prediction
        Model-produced answer.
    gold
        Reference (gold) answer.

    Returns
    -------
    bool
        ``True`` when the normalized strings are equal.
    """
    return normalize_answer(prediction) == normalize_answer(gold)


def token_f1(prediction: str, gold: str) -> float:
    r"""Return the token-level F1 between a prediction and a gold answer.

    The standard SQuAD token F1 [Rajpurkar2016]_ over the *multisets* of normalized
    tokens (:func:`answer_tokens`): with ``num_same`` the size of the multiset
    intersection,

    .. math::

        \text{precision} = \frac{\text{num\_same}}{|\text{pred tokens}|},\quad
        \text{recall} = \frac{\text{num\_same}}{|\text{gold tokens}|},\quad
        F_1 = \frac{2\,P\,R}{P + R}.

    This is the per-answer quantity averaged into the LoCoMo token-overlap fallback
    judge (:class:`TokenOverlapJudge`).

    Parameters
    ----------
    prediction
        Model-produced answer.
    gold
        Reference (gold) answer.

    Returns
    -------
    float
        Token F1 in ``[0, 1]``. Boundary conventions (SQuAD): if either side has no
        tokens, returns ``1.0`` iff *both* are empty else ``0.0``; ``0.0`` when the
        two token multisets are disjoint.
    """
    pred_tokens = answer_tokens(prediction)
    gold_tokens = answer_tokens(gold)
    if not pred_tokens or not gold_tokens:
        # Both empty -> vacuously perfect; exactly one empty -> no overlap possible.
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True, slots=True)
class FactF1:
    """Precision / recall / F1 over a set of gold facts (DMR fact-F1 protocol)."""

    precision: float
    recall: float
    f1: float
    true_positives: int
    n_predicted: int
    n_gold: int

    @property
    def f1_percent(self) -> float:
        r"""F1 as a percentage -- the ``tab:stdbench`` ``DMR / fact F1`` unit (%)."""
        return 100.0 * self.f1


def fact_f1(predicted_facts: Iterable[str], gold_facts: Iterable[str]) -> FactF1:
    r"""Return the DMR deep-memory *fact F1* of a predicted fact set against gold.

    Implements the ``DMR / fact F1`` row of Table~\ref{tab:stdbench}
    (Section~\ref{sec:pubbench}; ``Brief 94.2``). Each fact is reduced to its
    :func:`normalize_answer` form and de-duplicated into a set, so that
    ``true_positives`` counts *distinct* gold facts recovered:

    .. math::

        P = \frac{|\hat F \cap F|}{|\hat F|},\quad
        R = \frac{|\hat F \cap F|}{|F|},\quad
        F_1 = \frac{2\,P\,R}{P + R},

    with :math:`\hat F` the predicted and :math:`F` the gold fact set.

    Parameters
    ----------
    predicted_facts
        Facts extracted from the model answer.
    gold_facts
        Reference facts the answer should recover.

    Returns
    -------
    FactF1
        Precision, recall, F1, and the underlying counts. Boundary conventions: an
        empty prediction *and* empty gold is vacuously perfect (all three ``1.0``);
        if exactly one side is empty every rate is ``0.0``; disjoint sets give
        ``0.0``.
    """
    predicted = {normalize_answer(fact) for fact in predicted_facts}
    gold = {normalize_answer(fact) for fact in gold_facts}
    predicted.discard("")
    gold.discard("")

    if not predicted and not gold:
        return FactF1(precision=1.0, recall=1.0, f1=1.0, true_positives=0, n_predicted=0, n_gold=0)

    true_positives = len(predicted & gold)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(gold) if gold else 0.0
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom > 0 else 0.0
    return FactF1(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        n_predicted=len(predicted),
        n_gold=len(gold),
    )


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """One LoCoMo judge decision on whether a predicted answer matches the gold."""

    correct: bool
    score: float  # graded confidence in [0, 1]; the fallback uses token F1
    rationale: str


@runtime_checkable
class AnswerJudge(Protocol):
    r"""Arm-blind judge of a single answer against its gold reference.

    The call signature is the type-level encoding of Section~\ref{sec:grader}: the
    judge "sees the known invariant as reference but not the arm identity". It
    receives only the ``question``, the ``gold_answer`` (the reference), and the
    ``predicted_answer`` -- never which memory arm produced it -- so a judge
    *cannot* condition on the arm even accidentally. A live LLM judge (a model
    distinct from the answer model) implements this protocol; the offline default is
    :class:`TokenOverlapJudge`.
    """

    def __call__(self, question: str, gold_answer: str, predicted_answer: str) -> JudgeVerdict:
        """Return the correctness verdict for one predicted answer."""
        ...


@dataclass(frozen=True, slots=True)
class TokenOverlapJudge:
    r"""Deterministic LoCoMo judge: correct iff token overlap clears a threshold.

    The reproducible, network-free default for the LoCoMo LLM-judge-accuracy row of
    Table~\ref{tab:stdbench} (``sec:pubbench``). A prediction is judged correct when
    it is a normalized exact match *or* its :func:`token_f1` against the gold answer
    is at least ``threshold``. This is a deterministic *proxy* for the live LLM
    judge, not a reproduction of it: it keeps the offline pipeline scoring without an
    API key (see the ``pyproject`` offline-install note), and any live judge is
    dropped in through the :class:`AnswerJudge` protocol without touching
    aggregation.

    Parameters
    ----------
    threshold
        Minimum token F1 (in ``[0, 1]``) for a non-exact answer to count as correct.
        Defaults to ``0.5``; this is an implementation choice, not a paper value.
    """

    threshold: float = 0.5

    def __post_init__(self) -> None:
        """Validate the threshold lies in the unit interval."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")

    def __call__(self, question: str, gold_answer: str, predicted_answer: str) -> JudgeVerdict:
        """Return a correctness verdict from normalized token overlap with gold."""
        if answer_exact_match(predicted_answer, gold_answer):
            return JudgeVerdict(correct=True, score=1.0, rationale="normalized exact match")
        score = token_f1(predicted_answer, gold_answer)
        correct = score >= self.threshold
        return JudgeVerdict(
            correct=correct,
            score=score,
            rationale=f"token F1 {score:.3f} {'>=' if correct else '<'} threshold {self.threshold}",
        )


@dataclass(frozen=True, slots=True)
class AnswerRecord:
    """One benchmark item: a question, its gold answer, and a model prediction."""

    question: str
    gold_answer: str
    predicted_answer: str


@dataclass(frozen=True, slots=True)
class JudgeAccuracy:
    """Aggregate LoCoMo LLM-judge accuracy over a set of answered questions."""

    accuracy: float
    n_correct: int
    n_total: int
    verdicts: tuple[JudgeVerdict, ...]

    @property
    def percent(self) -> float:
        r"""Accuracy as a percentage -- the ``tab:stdbench`` LoCoMo unit (%)."""
        return 100.0 * self.accuracy


def llm_judge_accuracy(records: Sequence[AnswerRecord], judge: AnswerJudge) -> JudgeAccuracy:
    r"""Return LoCoMo LLM-judge accuracy: the mean correctness verdict over answers.

    Implements the ``LoCoMo / LLM-judge acc.`` row of Table~\ref{tab:stdbench}
    (Section~\ref{sec:pubbench}; ``Brief 87.6``). Each record is judged independently
    by the arm-blind ``judge`` (an :class:`AnswerJudge`; the reproducible default is
    :class:`TokenOverlapJudge`), and accuracy is the fraction of correct verdicts.
    The judge never receives the arm identity, so the same call scores every arm
    identically -- the arm-invariance Section~\ref{sec:grader} relies on.

    Parameters
    ----------
    records
        The answered questions to score (at least one).
    judge
        An arm-blind answer judge.

    Returns
    -------
    JudgeAccuracy
        Accuracy in ``[0, 1]``, the correct/total counts, and the per-record
        verdicts (in input order) for inspection.

    Raises
    ------
    ValueError
        If ``records`` is empty (accuracy over no questions is undefined).
    """
    if not records:
        raise ValueError("llm_judge_accuracy needs at least one record")
    verdicts = tuple(
        judge(record.question, record.gold_answer, record.predicted_answer) for record in records
    )
    n_correct = sum(1 for verdict in verdicts if verdict.correct)
    return JudgeAccuracy(
        accuracy=n_correct / len(verdicts),
        n_correct=n_correct,
        n_total=len(verdicts),
        verdicts=verdicts,
    )
