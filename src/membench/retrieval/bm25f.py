r"""Fielded BM25F lexical retrieval arm.

BM25F is the standard extension of Okapi BM25 to documents made of several
weighted fields (e.g. a short *title* and a longer *body*). Rather than scoring
each field independently and combining the field scores afterwards, BM25F first
combines per-field term frequencies *before* the non-linear saturation, so that a
single occurrence in a heavily weighted short field is not over-rewarded by the
saturation curve. A single global ``k1`` governs the saturation of the combined
pseudo term frequency.

For a query term :math:`t` and document :math:`d` with fields :math:`f`:

.. math::

    \tilde{tf}(t, d) = \sum_{f} w_f \,
        \frac{tf_f(t, d)}{1 - b + b \, \frac{l_f(d)}{\bar{l}_f}}

    \mathrm{score}(d, q) = \sum_{t \in q} \mathrm{idf}(t) \,
        \frac{\tilde{tf}(t, d)}{k_1 + \tilde{tf}(t, d)}

where :math:`w_f` is the weight of field :math:`f`, :math:`tf_f(t, d)` the raw
term frequency of :math:`t` in that field, :math:`l_f(d)` the field length,
:math:`\bar{l}_f` the corpus-average length of that field, :math:`b` the
length-normalisation parameter, and :math:`k_1` the (single, global) saturation
parameter. The inverse document frequency uses the non-negative Lucene/ATIRE form
:math:`\mathrm{idf}(t) = \ln\!\bigl(1 + \frac{N - n_t + 0.5}{n_t + 0.5}\bigr)`,
with :math:`N` documents and :math:`n_t` documents containing :math:`t` in any
field.

A single field with weight ``1`` reduces exactly to standard BM25 (up to the
rank-preserving constant ``k1 + 1`` that BM25F omits from the numerator):
:math:`\frac{tf / B}{k_1 + tf / B} = \frac{tf}{k_1 B + tf}` with
:math:`B = 1 - b + b\,l/\bar{l}`.

Reference
---------
S. Robertson, H. Zaragoza, and M. Taylor, "Simple BM25 Extension to Multiple
Weighted Fields," in *Proc. 13th ACM Int. Conf. on Information and Knowledge
Management (CIKM '04)*, 2004, pp. 42-49. See also S. Robertson and H. Zaragoza,
"The Probabilistic Relevance Framework: BM25 and Beyond," *Foundations and Trends
in Information Retrieval*, vol. 3, no. 4, pp. 333-389, 2009.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

from membench.retrieval._text import tokenize
from membench.retrieval.base import ArmFamily, MemorySystem, pack_to_budget, register_arm
from membench.types import MemoryItem, RetrievedContext

__all__ = ["BM25FMemory", "bm25_idf", "bm25f_document_score"]

_DEFAULT_K1 = 1.2
_DEFAULT_B = 0.75
_DEFAULT_TITLE_WEIGHT = 2.0
_DEFAULT_BODY_WEIGHT = 1.0
_TITLE_FIELD = "title"
_BODY_FIELD = "body"


def bm25_idf(n_docs: int, doc_freq: int) -> float:
    r"""Non-negative BM25 inverse document frequency (Lucene/ATIRE ``+1`` form).

    Parameters
    ----------
    n_docs
        Total number of documents in the corpus, :math:`N`.
    doc_freq
        Number of documents containing the term, :math:`n_t` (in any field).

    Returns
    -------
    float
        :math:`\\ln\\bigl(1 + (N - n_t + 0.5) / (n_t + 0.5)\\bigr)`, which is
        always :math:`\\ge 0` (unlike the classic Robertson-Sparck-Jones form,
        which can go negative for very common terms).
    """
    return math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))


def _weighted_field_tf(
    term: str,
    field_term_freqs: Mapping[str, Mapping[str, int]],
    field_lengths: Mapping[str, int],
    avg_field_lengths: Mapping[str, float],
    field_weights: Mapping[str, float],
    b: float,
) -> float:
    r"""Combine a term's per-field, length-normalised frequencies into one value.

    This is the BM25F pseudo term frequency :math:`\tilde{tf}(t, d)` computed
    *before* the saturation, summing ``w_f * tf_f / (1 - b + b * l_f / mean_l_f)``
    over the fields. A field with zero average length contributes its raw weighted
    frequency (no normalisation is defined when the field never occurs).
    """
    total = 0.0
    for field_name, weight in field_weights.items():
        inner = field_term_freqs.get(field_name)
        tf = inner.get(term, 0) if inner is not None else 0
        if tf == 0:
            continue
        avg = avg_field_lengths.get(field_name, 0.0)
        # No length normalisation is defined for a field that never occurs (avg 0).
        norm = 1.0 - b + b * (field_lengths.get(field_name, 0) / avg) if avg > 0.0 else 1.0
        total += weight * tf / norm
    return total


def bm25f_document_score(
    query_terms: Sequence[str],
    field_term_freqs: Mapping[str, Mapping[str, int]],
    field_lengths: Mapping[str, int],
    *,
    avg_field_lengths: Mapping[str, float],
    field_weights: Mapping[str, float],
    doc_freqs: Mapping[str, int],
    n_docs: int,
    k1: float,
    b: float,
) -> float:
    """Score one document against a query under BM25F.

    Each distinct query term contributes ``idf(t) * x / (k1 + x)`` where ``x`` is
    the field-weighted, length-normalised pseudo term frequency
    (:func:`_weighted_field_tf`). The saturation uses a single global ``k1``.

    Parameters
    ----------
    query_terms
        The query's tokens. Repeated terms are de-duplicated (each distinct term
        contributes once) and processed in sorted order for determinism.
    field_term_freqs
        ``field -> (term -> raw frequency)`` for this document.
    field_lengths
        ``field -> token length`` for this document.
    avg_field_lengths
        ``field -> mean token length across the corpus``.
    field_weights
        ``field -> weight`` :math:`w_f`. Fields absent here are ignored.
    doc_freqs
        ``term -> number of documents containing it`` (in any field).
    n_docs
        Total number of documents in the corpus.
    k1
        Global saturation parameter (typically ``1.2``).
    b
        Length-normalisation parameter in ``[0, 1]`` (typically ``0.75``).

    Returns
    -------
    float
        The BM25F relevance score (``0.0`` if no query term occurs).
    """
    score = 0.0
    for term in sorted(set(query_terms)):
        df = doc_freqs.get(term, 0)
        if df <= 0:
            continue
        x = _weighted_field_tf(
            term, field_term_freqs, field_lengths, avg_field_lengths, field_weights, b
        )
        if x == 0.0:
            continue
        score += bm25_idf(n_docs, df) * x / (k1 + x)
    return score


@register_arm("bm25f", family=ArmFamily.LEXICAL, follows_links=False, uses_embeddings=False)
class BM25FMemory(MemorySystem):
    """Rank corpus items by fielded BM25F (title/body) similarity, then pack to budget."""

    def __init__(
        self,
        *,
        k1: float = _DEFAULT_K1,
        b: float = _DEFAULT_B,
        title_weight: float = _DEFAULT_TITLE_WEIGHT,
        body_weight: float = _DEFAULT_BODY_WEIGHT,
    ) -> None:
        self._k1 = k1
        self._b = b
        self._field_weights: dict[str, float] = {
            _TITLE_FIELD: title_weight,
            _BODY_FIELD: body_weight,
        }
        self._items: list[MemoryItem] = []
        self._field_term_freqs: list[dict[str, dict[str, int]]] = []
        self._field_lengths: list[dict[str, int]] = []
        self._avg_field_lengths: dict[str, float] = {}
        self._doc_freqs: dict[str, int] = {}
        self._n_docs = 0

    @staticmethod
    def _split_fields(text: str) -> dict[str, list[str]]:
        """Derive two fields from flat text: the first line is the title, rest is body."""
        first_line, _, rest = text.partition("\n")
        return {_TITLE_FIELD: tokenize(first_line), _BODY_FIELD: tokenize(rest)}

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Ingest the corpus; build per-field frequency, length, and doc-frequency tables."""
        self._items.extend(items)
        self._n_docs = len(self._items)
        self._field_term_freqs = []
        self._field_lengths = []
        length_sums: dict[str, int] = dict.fromkeys(self._field_weights, 0)
        doc_freqs: dict[str, int] = {}
        for item in self._items:
            fields = self._split_fields(item.text)
            per_field_tf: dict[str, dict[str, int]] = {}
            per_field_len: dict[str, int] = {}
            terms_in_doc: set[str] = set()
            for field_name in self._field_weights:
                tokens = fields.get(field_name, [])
                tf: dict[str, int] = {}
                for token in tokens:
                    tf[token] = tf.get(token, 0) + 1
                per_field_tf[field_name] = tf
                per_field_len[field_name] = len(tokens)
                length_sums[field_name] += len(tokens)
                terms_in_doc.update(tf)
            self._field_term_freqs.append(per_field_tf)
            self._field_lengths.append(per_field_len)
            for term in terms_in_doc:
                doc_freqs[term] = doc_freqs.get(term, 0) + 1
        self._doc_freqs = doc_freqs
        n = self._n_docs
        self._avg_field_lengths = {
            field_name: (total / n if n > 0 else 0.0) for field_name, total in length_sums.items()
        }

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Return the BM25F-ranked corpus packed into the budget."""
        if not self._items:
            return RetrievedContext.empty()
        query_terms = tokenize(query)
        if not query_terms or self._n_docs == 0:
            # Degenerate query: fall back to corpus order rather than crashing.
            order: list[int] = list(range(len(self._items)))
        else:
            scores = [
                bm25f_document_score(
                    query_terms,
                    self._field_term_freqs[i],
                    self._field_lengths[i],
                    avg_field_lengths=self._avg_field_lengths,
                    field_weights=self._field_weights,
                    doc_freqs=self._doc_freqs,
                    n_docs=self._n_docs,
                    k1=self._k1,
                    b=self._b,
                )
                for i in range(len(self._items))
            ]
            # Stable descending sort: ties keep corpus order (index tie-break).
            order = sorted(range(len(self._items)), key=lambda i: (-scores[i], i))
        ranked = [(self._items[i].item_id, self._items[i].text) for i in order]
        return pack_to_budget(ranked, budget_tokens)
