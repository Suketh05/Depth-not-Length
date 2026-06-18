r"""Retrieval-quality metrics scored against the gold governing decisions.

The diagnostic layer of the benchmark: given an arm's ordered retrieved ids and the
gold ids a task's output must honour, compute the standard information-retrieval
quantities -- precision/recall@k, reciprocal rank, average precision, nDCG -- plus
the one the depth thesis cares about most, **full-chain recovery** (did the arm
recover *every* gold id, the analogue of :math:`P_\text{sim}` / :math:`P_\text{struct}`).

Binary relevance (an id is gold or it is not) is used throughout, which is the right
model here: a governing decision is either recovered or not. All metrics follow the
standard conventions (recall on empty gold is vacuously 1; precision on empty
retrieval is 1 only if there was nothing to retrieve).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "RetrievalScore",
    "average_precision",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "score_retrieval",
]


def _dedup(retrieved: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for r in retrieved:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def precision_at_k(retrieved: Sequence[str], gold: set[str], k: int | None = None) -> float:
    """Fraction of the top-``k`` retrieved ids that are gold (k defaults to all)."""
    items = _dedup(retrieved)
    cut = items if k is None else items[:k]
    if not cut:
        return 1.0 if not gold else 0.0
    return sum(1 for r in cut if r in gold) / len(cut)


def recall_at_k(retrieved: Sequence[str], gold: set[str], k: int | None = None) -> float:
    """Fraction of gold ids found in the top-``k`` retrieved (vacuously 1 if no gold)."""
    if not gold:
        return 1.0
    cut = _dedup(retrieved) if k is None else _dedup(retrieved)[:k]
    return sum(1 for g in gold if g in cut) / len(gold)


def reciprocal_rank(retrieved: Sequence[str], gold: set[str]) -> float:
    """Reciprocal of the rank of the first gold id (0.0 if none retrieved)."""
    for rank, r in enumerate(_dedup(retrieved), start=1):
        if r in gold:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved: Sequence[str], gold: set[str]) -> float:
    """Average precision: mean of precision@k at each gold hit, over all gold ids."""
    if not gold:
        return 1.0
    items = _dedup(retrieved)
    hits = 0
    cumulative = 0.0
    for rank, r in enumerate(items, start=1):
        if r in gold:
            hits += 1
            cumulative += hits / rank
    return cumulative / len(gold)


def ndcg_at_k(retrieved: Sequence[str], gold: set[str], k: int | None = None) -> float:
    r"""Normalised discounted cumulative gain with binary relevance.

    :math:`DCG = \sum_i \mathbb{1}[r_i \in gold] / \log_2(i+1)`, normalised by the
    ideal DCG (all gold ranked first). Returns 1.0 when there is no gold.
    """
    if not gold:
        return 1.0
    items = _dedup(retrieved)
    cut = items if k is None else items[:k]
    dcg = sum(1.0 / math.log2(rank + 1) for rank, r in enumerate(cut, start=1) if r in gold)
    ideal_hits = min(len(gold), len(cut)) if k is not None else len(gold)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass(frozen=True, slots=True)
class RetrievalScore:
    """All retrieval-quality metrics for one query."""

    precision: float
    recall: float
    mrr: float
    average_precision: float
    ndcg: float
    chain_recovered: bool
    retrieved: int
    gold: int
    true_positives: tuple[str, ...]


def score_retrieval(
    retrieved_ids: Sequence[str],
    governing_decisions: Sequence[str],
    *,
    k: int | None = None,
) -> RetrievalScore:
    """Compute the full retrieval-quality bundle for one query.

    ``chain_recovered`` is the load-bearing diagnostic: True only if *every* gold id
    was retrieved (full causal-chain recovery).
    """
    items = _dedup(retrieved_ids)
    gold = set(governing_decisions)
    true_positives = tuple(sorted(gold & set(items)))
    return RetrievalScore(
        precision=precision_at_k(items, gold, k),
        recall=recall_at_k(items, gold, k),
        mrr=reciprocal_rank(items, gold),
        average_precision=average_precision(items, gold),
        ndcg=ndcg_at_k(items, gold, k),
        chain_recovered=bool(gold) and gold <= set(items),
        retrieved=len(items),
        gold=len(gold),
        true_positives=true_positives,
    )
