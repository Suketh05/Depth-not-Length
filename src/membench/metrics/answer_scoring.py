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
from collections.abc import Iterable
from dataclasses import dataclass

__all__ = [
    "FactF1",
    "answer_exact_match",
    "answer_tokens",
    "fact_f1",
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
