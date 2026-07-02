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

__all__ = [
    "answer_tokens",
    "normalize_answer",
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
