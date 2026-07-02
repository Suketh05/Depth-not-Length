"""Answer-vs-gold scoring for the public benchmarks (paper ``sec:pubbench``).

The paper's Table ``tab:stdbench`` ("Standard benchmarks on a unified harness")
reports, for every memory system, *LoCoMo LLM-judge accuracy* (Brief 87.6%),
*DMR fact F1* (Brief 94.2%), and *SWE-ContextBench resolution* (Brief 47.3%),
each produced by "each benchmark's own harness under one fixed configuration"
(identical corpora, budgets, answer model, and graders per benchmark). The
loaders (:mod:`membench.datasets.locomo`, :mod:`membench.datasets.dmr`) emit
tasks under the :attr:`~membench.types.Scorer.QA_MATCH` contract and expose
``gold_answers()``; this module supplies the missing half of that contract --
the answer-vs-gold scorers themselves:

* :func:`normalize_answer` / :func:`exact_match` / :func:`token_f1` -- the
  standard SQuAD-style normalization and token-overlap F1 (Rajpurkar et al.,
  2016) that conversational-memory harnesses grade free-text answers with.
* :func:`fact_f1` / :func:`mean_fact_f1` -- the DMR "fact F1" row of
  ``tab:stdbench``: precision/recall/F1 over *extracted facts* after
  normalization, macro-averaged over questions.
* :func:`token_overlap_judge` / :func:`llm_judge_accuracy` -- the LoCoMo
  "LLM-judge acc." row: per-question binary verdicts from a judge that is
  *distinct from the answer models* and *arm-blind* (paper ``sec:grader`` /
  ``sec:gradervalidity``: the judge "sees the known invariant as reference but
  not the arm identity"). The judge is a pluggable hook; the default is a
  deterministic token-overlap verdict so the offline pipeline stays
  reproducible, mirroring the design of :mod:`membench.metrics.compliance`.

Numbers from ``tab:stdbench`` are cited for provenance only; nothing in this
module asserts them -- they are outputs of live runs, not of these formulas.
The companion resolution scorer lives in :mod:`membench.metrics.resolution`.

References
----------
Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). SQuAD: 100,000+
Questions for Machine Comprehension of Text. EMNLP 2016. (Defines the
normalize/exact-match/token-F1 protocol reused by conversational QA harnesses.)
Maharana, A. et al. (2024). Evaluating Very Long-Term Conversational Memory of
LLM Agents (LoCoMo). ACL 2024. (LLM-judge accuracy protocol.)
Packer, C. et al. (2023). MemGPT: Towards LLMs as Operating Systems.
arXiv:2310.08560. (Introduces the Deep Memory Retrieval task scored here.)
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = [
    "F1Result",
    "exact_match",
    "fact_f1",
    "mean_fact_f1",
    "normalize_answer",
    "split_facts",
    "token_f1",
    "tokenize_answer",
]

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WHITESPACE = re.compile(r"\s+", re.UNICODE)
_PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Normalize a free-text answer the SQuAD way before comparison.

    Applies, in order: lowercase, punctuation removal, English article removal
    (``a``/``an``/``the`` as whole words), and whitespace collapse. This is the
    exact ``normalize_answer`` of the SQuAD evaluation script (Rajpurkar et
    al., 2016), the de-facto grading normalization for conversational QA
    benchmarks; LoCoMo and DMR harnesses grade against gold under it. The
    function is idempotent: ``normalize_answer(normalize_answer(t)) ==
    normalize_answer(t)``.

    Parameters
    ----------
    text
        Raw answer text (prediction or gold).

    Returns
    -------
    str
        The normalized form; ``""`` when nothing but punctuation/articles/
        whitespace remains ("no answer").
    """
    lowered = text.lower()
    no_punct = lowered.translate(_PUNCTUATION)
    no_articles = _ARTICLES.sub(" ", no_punct)
    return _WHITESPACE.sub(" ", no_articles).strip()


def tokenize_answer(text: str) -> list[str]:
    """Split a *normalized* answer into whitespace tokens.

    Parameters
    ----------
    text
        Answer text; it is normalized via :func:`normalize_answer` first, so
        callers may pass raw strings.

    Returns
    -------
    list of str
        Normalized tokens; empty when the normalized answer is ``""``.
    """
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


@dataclass(frozen=True, slots=True)
class F1Result:
    """Precision / recall / F1 with the integer overlap counts that produced them.

    The counts are retained so every ratio is auditable: ``precision =
    overlap / n_predicted``, ``recall = overlap / n_gold`` and, by the
    harmonic-mean identity,

    .. math::

        F_1 = \\frac{2PR}{P + R} = \\frac{2 \\cdot \\text{overlap}}
              {n_\\text{predicted} + n_\\text{gold}},

    which is how :attr:`f1` is computed -- one exact integer division, no
    intermediate float rounding, so tiny hand-computed goldens are exact
    fractions (e.g. overlap 2 of 3 predicted / 4 gold gives F1 = 4/7).
    """

    precision: float
    recall: float
    f1: float
    overlap: int
    n_predicted: int
    n_gold: int


def _f1_from_counts(overlap: int, n_predicted: int, n_gold: int) -> F1Result:
    """Assemble an :class:`F1Result` from integer counts (single-division ratios)."""
    precision = overlap / n_predicted if n_predicted else 0.0
    recall = overlap / n_gold if n_gold else 0.0
    f1 = 2 * overlap / (n_predicted + n_gold) if (n_predicted + n_gold) else 0.0
    return F1Result(
        precision=precision,
        recall=recall,
        f1=f1,
        overlap=overlap,
        n_predicted=n_predicted,
        n_gold=n_gold,
    )


def exact_match(prediction: str, gold: str) -> bool:
    """Return whether prediction and gold agree after normalization.

    The SQuAD exact-match event: equality of :func:`normalize_answer` forms.
    Two "no answer" strings (both normalize to ``""``) match.

    Parameters
    ----------
    prediction
        Model answer text.
    gold
        Gold answer text.

    Returns
    -------
    bool
        ``True`` iff the normalized strings are identical.
    """
    return normalize_answer(prediction) == normalize_answer(gold)


def token_f1(prediction: str, gold: str) -> F1Result:
    """Token-overlap F1 between a predicted and a gold answer (SQuAD protocol).

    Both strings are normalized and whitespace-tokenized; the overlap is the
    *multiset* (bag) intersection of the two token bags, exactly as in the
    SQuAD evaluation script: a token predicted twice only counts twice if it
    also appears twice in gold. Precision divides by the predicted-token
    count, recall by the gold-token count.

    Degenerate "no answer" rule (also SQuAD's): if either side normalizes to
    the empty string, F1 (and P and R) is 1.0 when *both* are empty and 0.0
    otherwise -- token overlap is meaningless against an empty bag.

    Parameters
    ----------
    prediction
        Model answer text.
    gold
        Gold answer text.

    Returns
    -------
    F1Result
        Precision, recall, F1 and the integer counts behind them.
    """
    pred_tokens = tokenize_answer(prediction)
    gold_tokens = tokenize_answer(gold)
    if not pred_tokens or not gold_tokens:
        # SQuAD degenerate rule: empty-vs-empty is a perfect match, else zero.
        score = 1.0 if pred_tokens == gold_tokens else 0.0
        return F1Result(
            precision=score,
            recall=score,
            f1=score,
            overlap=0,
            n_predicted=len(pred_tokens),
            n_gold=len(gold_tokens),
        )
    overlap_counts = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(overlap_counts.values())
    return _f1_from_counts(overlap, len(pred_tokens), len(gold_tokens))


_FACT_DELIMITERS = re.compile(r"[\n;]+|(?<=[.!?])\s+")


def split_facts(answer_text: str) -> tuple[str, ...]:
    """Split a free-text answer into candidate fact strings, deterministically.

    Facts are separated on newlines, semicolons, and sentence-final
    punctuation followed by whitespace -- the delimiters DMR-style answers
    actually use to list facts. Empty fragments are dropped; fragments are
    *not* normalized here (that happens inside :func:`fact_f1`), so the raw
    fact text stays available for error inspection.

    Parameters
    ----------
    answer_text
        A model or gold answer that lists one or more facts.

    Returns
    -------
    tuple of str
        The non-empty fact fragments, in order of appearance.
    """
    fragments = (fragment.strip() for fragment in _FACT_DELIMITERS.split(answer_text))
    return tuple(fragment for fragment in fragments if fragment)


def _normalized_fact_set(facts: Iterable[str]) -> set[str]:
    """Normalize each fact and deduplicate; empty-normalizing facts are dropped."""
    normalized = (normalize_answer(fact) for fact in facts)
    return {fact for fact in normalized if fact}


def fact_f1(predicted_facts: Iterable[str], gold_facts: Iterable[str]) -> F1Result:
    """DMR fact F1: precision/recall/F1 over normalized extracted facts.

    This is the "DMR fact F1" metric of the paper's ``tab:stdbench`` row
    (Brief 94.2% vs. Kluris 87.0%, cited for provenance -- not asserted
    here). Each fact string is normalized with :func:`normalize_answer` and
    the two sides are compared as *sets* (duplicate statements of the same
    fact neither help nor hurt):

    * precision = matched / predicted -- fraction of asserted facts that are
      in the gold fact set (penalizes hallucinated facts);
    * recall = matched / gold -- fraction of gold facts the answer recovered
      (penalizes forgotten facts);
    * F1 = harmonic mean, computed as ``2*matched / (n_predicted + n_gold)``.

    A fact "matches" iff its normalized form is identical to a gold fact's
    normalized form -- the deterministic criterion; graded/semantic matching
    belongs to the LLM-judge path (:func:`llm_judge_accuracy`), never here.

    Degenerate rule (mirrors :func:`token_f1`): if either side is empty after
    normalization, the score is 1.0 when both are empty and 0.0 otherwise.

    Parameters
    ----------
    predicted_facts
        Facts extracted from the model answer (e.g. via :func:`split_facts`).
    gold_facts
        The benchmark's gold facts for the question.

    Returns
    -------
    F1Result
        Set precision/recall/F1 with the integer counts behind them.
    """
    predicted = _normalized_fact_set(predicted_facts)
    gold = _normalized_fact_set(gold_facts)
    if not predicted or not gold:
        score = 1.0 if predicted == gold else 0.0
        return F1Result(
            precision=score,
            recall=score,
            f1=score,
            overlap=0,
            n_predicted=len(predicted),
            n_gold=len(gold),
        )
    return _f1_from_counts(len(predicted & gold), len(predicted), len(gold))


def mean_fact_f1(results: Sequence[F1Result]) -> float:
    """Macro-average the per-question fact F1 into one benchmark score.

    ``tab:stdbench`` reports a single "fact F1 (%)" per system; it is the
    unweighted mean of per-question F1 (every question counts equally,
    regardless of how many facts it carries), scaled by the caller to percent.

    Parameters
    ----------
    results
        Per-question :class:`F1Result` values from :func:`fact_f1`.

    Returns
    -------
    float
        Mean F1 in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``results`` is empty -- an empty benchmark has no score, and
        silently returning 0 would fabricate one.
    """
    if not results:
        raise ValueError("mean_fact_f1 needs at least one per-question result")
    return sum(result.f1 for result in results) / len(results)
