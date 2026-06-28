r"""Score-combination fusion (CombSUM / CombMNZ / CombANZ) over multiple rankers.

Where the ``hybrid_rrf`` arm fuses *ranks*, this arm fuses *scores*: it takes the
raw relevance scores of several rankers (here a lexical BM25 ranker and a dense
embedding ranker), puts them on a common scale, and combines them additively.
This is the original family of fusion operators introduced by Fox & Shaw, with
the min-max score normalisation popularised by Lee. Score fusion keeps the
*magnitude* of agreement between rankers (how strongly each ranker liked a
document), which rank fusion deliberately discards, so it is a complementary
similarity baseline against the structured arm.

Given a document ``d`` and rankers ``i = 1..m`` with raw scores ``s_i(d)``, each
ranker is min-max normalised over the documents it scored::

    n_i(d) = (s_i(d) - min_i) / (max_i - min_i)        in [0, 1]

(a ranker whose scores are all equal contributes ``0`` for every document, i.e.
it provides no discriminating signal). Writing ``z(d) = |{ i : n_i(d) > 0 }|``
for the number of rankers that gave ``d`` a strictly positive normalised score,
the three combiners are::

    CombSUM(d) = sum_i n_i(d)
    CombMNZ(d) = CombSUM(d) * z(d)
    CombANZ(d) = CombSUM(d) / z(d)         (0 when z(d) = 0)

CombMNZ rewards documents found (with signal) by *many* rankers; CombANZ rewards
documents found by *few* rankers but strongly. A document absent from a ranker is
treated as score ``0`` for that ranker and does not contribute to ``z(d)``.

References
----------
Fox, E. A., & Shaw, J. A. (1994). Combination of multiple searches.
In *The Second Text REtrieval Conference (TREC-2)*, NIST Special Publication
500-215, pp. 243-252.

Lee, J. H. (1997). Analyses of multiple evidence combination.
In *Proceedings of the 20th Annual International ACM SIGIR Conference on Research
and Development in Information Retrieval (SIGIR '97)*, pp. 267-276.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from enum import Enum

from rank_bm25 import BM25Okapi

from membench.retrieval._text import tokenize
from membench.retrieval.base import ArmFamily, MemorySystem, pack_to_budget, register_arm
from membench.retrieval.embeddings import (
    EmbeddingProvider,
    HashingEmbeddingProvider,
    cosine_similarity_matrix,
)
from membench.types import MemoryItem, RetrievedContext

__all__ = [
    "CombANZMemory",
    "CombMNZMemory",
    "CombSUMMemory",
    "FusionMethod",
    "fuse_scores",
    "min_max_normalize",
]


class FusionMethod(str, Enum):
    """The three Fox & Shaw score-combination operators."""

    COMBSUM = "combsum"
    COMBMNZ = "combmnz"
    COMBANZ = "combanz"


def min_max_normalize(scores: Mapping[str, float]) -> dict[str, float]:
    """Min-max normalise a ranker's scores to ``[0, 1]``.

    Each score is mapped to ``(s - min) / (max - min)``. A ranker whose scores are
    all equal (zero range) carries no discriminating information and is mapped to
    ``0.0`` for every document, rather than raising on a divide-by-zero.

    Parameters
    ----------
    scores
        Mapping of ``item_id`` to that ranker's raw relevance score.

    Returns
    -------
    dict[str, float]
        Mapping of ``item_id`` to a normalised score in ``[0, 1]``, in the
        iteration order of ``scores``.
    """
    if not scores:
        return {}
    values = scores.values()
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 0.0:
        return dict.fromkeys(scores, 0.0)
    return {item_id: (score - lo) / span for item_id, score in scores.items()}


def fuse_scores(
    ranker_scores: Sequence[Mapping[str, float]],
    method: FusionMethod = FusionMethod.COMBSUM,
) -> dict[str, float]:
    """Fuse several rankers' raw scores with a Fox & Shaw combiner.

    Each ranker is min-max normalised independently (see :func:`min_max_normalize`),
    then the per-document normalised scores are combined according to ``method``.
    A document absent from a ranker contributes ``0`` for that ranker and is not
    counted toward the non-zero multiplier ``z(d)``.

    Parameters
    ----------
    ranker_scores
        One mapping of ``item_id -> raw score`` per ranker.
    method
        Which combiner to apply (CombSUM, CombMNZ, or CombANZ).

    Returns
    -------
    dict[str, float]
        Mapping of ``item_id`` to fused score, keyed in first-seen order across the
        rankers (so a stable sort reproduces the input ordering on ties).
    """
    normalised = [min_max_normalize(scores) for scores in ranker_scores]

    # Union of all document ids, preserving first-seen order for stable tie-breaks.
    fused_sum: dict[str, float] = {}
    nonzero: dict[str, int] = {}
    for norm in normalised:
        for item_id, value in norm.items():
            fused_sum[item_id] = fused_sum.get(item_id, 0.0) + value
            count = nonzero.get(item_id, 0)
            nonzero[item_id] = count + 1 if value > 0.0 else count

    if method is FusionMethod.COMBSUM:
        return dict(fused_sum)
    if method is FusionMethod.COMBMNZ:
        return {item_id: total * nonzero[item_id] for item_id, total in fused_sum.items()}
    return {
        item_id: (total / nonzero[item_id] if nonzero[item_id] else 0.0)
        for item_id, total in fused_sum.items()
    }


class _ScoreFusionMemory(MemorySystem):
    """Fuse BM25 and dense embedding scores with a Fox & Shaw combiner.

    Parameters
    ----------
    method
        Which score-combination operator to apply.
    provider
        Embedding backend for the dense ranker (defaults to the deterministic
        offline hashing provider).
    """

    _method: FusionMethod = FusionMethod.COMBSUM

    def __init__(
        self,
        method: FusionMethod | None = None,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        self._fusion_method = method if method is not None else self._method
        self._provider = provider or HashingEmbeddingProvider()
        self._items: list[MemoryItem] = []
        self._tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Ingest the corpus and build the BM25 index over its tokenised texts."""
        self._items.extend(items)
        self._tokens = [tokenize(item.text) for item in self._items]
        self._bm25 = BM25Okapi(self._tokens) if any(self._tokens) else None

    def _lexical_scores(self, query: str) -> dict[str, float]:
        query_tokens = tokenize(query)
        if self._bm25 is None or not query_tokens:
            return {item.item_id: 0.0 for item in self._items}
        raw = self._bm25.get_scores(query_tokens)
        return {self._items[i].item_id: float(raw[i]) for i in range(len(self._items))}

    def _dense_scores(self, query: str) -> dict[str, float]:
        matrix = self._provider.embed([item.text for item in self._items])
        query_vec = self._provider.embed_one(query)
        sims = cosine_similarity_matrix(query_vec, matrix)[0]
        return {self._items[i].item_id: float(sims[i]) for i in range(len(self._items))}

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Return the score-fused ranking packed into the budget."""
        if not self._items:
            return RetrievedContext.empty()
        text_by_id = {item.item_id: item.text for item in self._items}
        fused = fuse_scores(
            [self._lexical_scores(query), self._dense_scores(query)],
            method=self._fusion_method,
        )
        # ``fused`` is keyed in corpus order, so a stable sort by -score breaks ties
        # by original corpus order.
        ordered = sorted(fused, key=lambda item_id: -fused[item_id])
        ranked = [(item_id, text_by_id[item_id]) for item_id in ordered]
        return pack_to_budget(ranked, budget_tokens)


@register_arm("comb_sum", family=ArmFamily.HYBRID, follows_links=False, uses_embeddings=True)
class CombSUMMemory(_ScoreFusionMemory):
    """Fuse BM25 and dense scores with CombSUM (sum of normalised scores)."""

    _method = FusionMethod.COMBSUM


@register_arm("comb_mnz", family=ArmFamily.HYBRID, follows_links=False, uses_embeddings=True)
class CombMNZMemory(_ScoreFusionMemory):
    """Fuse BM25 and dense scores with CombMNZ (CombSUM times non-zero count)."""

    _method = FusionMethod.COMBMNZ


@register_arm("comb_anz", family=ArmFamily.HYBRID, follows_links=False, uses_embeddings=True)
class CombANZMemory(_ScoreFusionMemory):
    """Fuse BM25 and dense scores with CombANZ (CombSUM over non-zero count)."""

    _method = FusionMethod.COMBANZ
