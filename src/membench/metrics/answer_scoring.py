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

__all__ = [
    "normalize_answer",
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
