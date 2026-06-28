r"""Extra rank-quality information-retrieval metrics over graded-relevance vectors.

The companion to :mod:`membench.metrics.retrieval` (which scores id-sets against gold).
This module operates on a **relevance vector** -- the per-rank relevance of the
retrieved list, best-first -- and supplies the user-model-based and parametric ranking
measures the depth thesis uses to weigh *where* on the list a gold id landed, not just
whether it was recovered. The four metrics here are deliberately distinct from the
``precision@k`` / ``reciprocal_rank`` / ``ndcg_at_k`` already in
:mod:`membench.metrics.retrieval`.

Relevance vectors are sequences of floats in rank order (index 0 = top result).
Binary relevance (0/1) is the common case; Expected Reciprocal Rank additionally
supports graded relevance.

Metrics
-------
Rank-Biased Precision (RBP)
    A persistence-model utility: a user steps down the ranking, continuing to the next
    result with probability ``p`` and stopping with probability ``1 - p``. The expected
    utility is

    .. math:: \mathrm{RBP} = (1 - p) \sum_{i=1}^{d} r_i \, p^{\,i-1}

    with :math:`r_i \in [0, 1]` the relevance at rank :math:`i`. Computed over the
    observed (finite) list this is a lower bound on the full-list RBP; the unjudged
    tail forms the *residual* :math:`p^{d}` (see :func:`rank_biased_precision`).
    Reference: A. Moffat and J. Zobel, "Rank-biased precision for measurement of
    retrieval effectiveness", *ACM Trans. Inf. Syst.* 27(1), 2008.

Expected Reciprocal Rank (ERR)
    A cascade user model: the user scans top-down and stops at rank :math:`r` with
    probability :math:`R_r \prod_{i<r}(1 - R_i)`, where the per-grade stop (satisfaction)
    probability is :math:`R_g = (2^{g} - 1) / 2^{g_{\max}}`. ERR is the expected
    reciprocal of the stopping rank,

    .. math:: \mathrm{ERR} = \sum_{r=1}^{d} \frac{1}{r} R_r \prod_{i=1}^{r-1} (1 - R_i).

    Reference: O. Chapelle, D. Metlzer, Y. Zhang and P. Grinspan, "Expected reciprocal
    rank for graded relevance", *CIKM* 2009.

Average Precision (AP) / Mean Average Precision (MAP)
    The mean of the precision evaluated at each relevant rank, normalised by the number
    of relevant items; MAP averages AP across queries. Reference: standard IR (e.g.
    Manning, Raghavan and Schutze, *Introduction to Information Retrieval*, 2008, ch. 8).

R-precision
    Precision at cutoff :math:`R`, where :math:`R` is the number of relevant items: the
    fraction of the top-:math:`R` results that are relevant. Reference: same, ch. 8.

All metrics return a value in :math:`[0, 1]`; for a fixed relevance multiset, ranking
the relevant items first maximises each of them.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "average_precision",
    "expected_reciprocal_rank",
    "mean_average_precision",
    "r_precision",
    "rank_biased_precision",
]


def rank_biased_precision(relevances: Sequence[float], p: float = 0.8) -> float:
    r"""Rank-Biased Precision of a relevance vector (Moffat & Zobel 2008).

    Implements :math:`\mathrm{RBP} = (1 - p) \sum_{i=1}^{d} r_i \, p^{\,i-1}` over the
    observed list. Because the list is finite this is the lower-bound (base) RBP; the
    mass on the unjudged tail is the residual :math:`p^{d}` and is *not* added here.

    Parameters
    ----------
    relevances : Sequence[float]
        Per-rank relevance, best-first (index 0 is the top result). Each value should
        lie in ``[0, 1]``; binary 0/1 relevance is the usual case.
    p : float, optional
        Persistence (continuation) probability, by default ``0.8``. Must satisfy
        ``0 < p < 1``: larger ``p`` models a more patient user and weights deeper ranks.

    Returns
    -------
    float
        The base RBP in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``p`` is not strictly inside ``(0, 1)``.
    """
    if not 0.0 < p < 1.0:
        msg = f"persistence p must satisfy 0 < p < 1, got {p}"
        raise ValueError(msg)
    weighted = sum(rel * p**i for i, rel in enumerate(relevances))
    return (1.0 - p) * weighted


def expected_reciprocal_rank(
    relevances: Sequence[float],
    *,
    max_grade: float | None = None,
) -> float:
    r"""Compute Expected Reciprocal Rank of a graded relevance vector (Chapelle 2009).

    Maps each grade ``g`` to a stop probability :math:`R_g = (2^{g} - 1) / 2^{g_{\max}}`
    and returns :math:`\sum_{r} \frac{1}{r} R_r \prod_{i<r}(1 - R_i)`.

    Parameters
    ----------
    relevances : Sequence[float]
        Per-rank graded relevance, best-first. Values must be non-negative; binary 0/1
        relevance is supported as the two-grade special case.
    max_grade : float or None, optional
        The maximum attainable grade :math:`g_{\max}` used to normalise the stop
        probabilities. If ``None`` (default) it is taken as the maximum observed grade
        (falling back to ``1.0`` when every grade is zero).

    Returns
    -------
    float
        The ERR in ``[0, 1]``.

    Raises
    ------
    ValueError
        If any relevance is negative, or if a non-positive ``max_grade`` is supplied.
    """
    if any(rel < 0.0 for rel in relevances):
        msg = "ERR requires non-negative graded relevances"
        raise ValueError(msg)
    if max_grade is None:
        observed = max(relevances, default=0.0)
        g_max = observed if observed > 0.0 else 1.0
    else:
        if max_grade <= 0.0:
            msg = f"max_grade must be positive, got {max_grade}"
            raise ValueError(msg)
        g_max = max_grade
    denom = 2.0**g_max
    err = 0.0
    remaining = 1.0  # probability the user has not yet stopped before this rank
    for rank, grade in enumerate(relevances, start=1):
        stop_prob = (2.0**grade - 1.0) / denom
        err += remaining * stop_prob / rank
        remaining *= 1.0 - stop_prob
    return err


def average_precision(relevances: Sequence[float]) -> float:
    """Average Precision of a relevance vector under binary relevance.

    An item is relevant when its relevance is strictly positive. AP is the mean of the
    precision evaluated at each relevant rank, divided by the number of relevant items.

    Parameters
    ----------
    relevances : Sequence[float]
        Per-rank relevance, best-first; any value ``> 0`` counts as relevant.

    Returns
    -------
    float
        The average precision in ``[0, 1]``; ``0.0`` when there are no relevant items.
    """
    n_relevant = sum(1 for rel in relevances if rel > 0.0)
    if n_relevant == 0:
        return 0.0
    hits = 0
    cumulative = 0.0
    for rank, rel in enumerate(relevances, start=1):
        if rel > 0.0:
            hits += 1
            cumulative += hits / rank
    return cumulative / n_relevant


def mean_average_precision(rankings: Sequence[Sequence[float]]) -> float:
    """Mean Average Precision across several relevance vectors.

    Parameters
    ----------
    rankings : Sequence[Sequence[float]]
        One relevance vector per query, each best-first.

    Returns
    -------
    float
        The mean of :func:`average_precision` over the queries in ``[0, 1]``; ``0.0``
        when no queries are supplied.
    """
    if not rankings:
        return 0.0
    return sum(average_precision(r) for r in rankings) / len(rankings)


def r_precision(relevances: Sequence[float], n_relevant: int | None = None) -> float:
    """R-precision: precision at the cutoff ``R`` equal to the relevant-item count.

    Parameters
    ----------
    relevances : Sequence[float]
        Per-rank relevance, best-first; any value ``> 0`` counts as relevant.
    n_relevant : int or None, optional
        The total number of relevant items ``R``. If ``None`` (default) it is taken as
        the number of positive entries in ``relevances`` (useful when the full relevant
        set is known to be entirely contained in the ranking).

    Returns
    -------
    float
        The fraction of the top-``R`` results that are relevant, in ``[0, 1]``; ``0.0``
        when ``R`` is zero.

    Raises
    ------
    ValueError
        If ``n_relevant`` is negative.
    """
    if n_relevant is None:
        r = sum(1 for rel in relevances if rel > 0.0)
    else:
        if n_relevant < 0:
            msg = f"n_relevant must be non-negative, got {n_relevant}"
            raise ValueError(msg)
        r = n_relevant
    if r == 0:
        return 0.0
    top_r = relevances[:r]
    return sum(1 for rel in top_r if rel > 0.0) / r
