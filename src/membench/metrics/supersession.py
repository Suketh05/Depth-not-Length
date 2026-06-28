r"""Recency-correct supersession rate: did the system apply the *current* decision.

Many tasks carry a *chain* of decisions in which a later decision supersedes an
earlier one (a policy is revised, a constraint is relaxed, a rule is overruled).
The only correct behaviour is to honour the **current** decision -- the unique
decision that nothing else supersedes -- and never a stale, already-overruled one.

Modelled formally, the ``supersedes`` relation over a case's decisions is a strict
partial order; "current" is its unique maximal element (the head of the chain). The
supersession rate is then plain accuracy against that target:

.. math::

    \mathrm{rate} = \frac{1}{N} \sum_{i=1}^{N}
        \mathbb{1}\!\left[\, a_i = c_i \,\right],

where :math:`a_i` is the decision the system applied in case :math:`i` and
:math:`c_i` is that case's current (latest non-superseded) decision. This is the
empirical "recency" principle of belief revision -- when beliefs conflict, the most
recent information takes precedence -- specialised to an explicit supersession DAG.

A uniformly random system that picks one decision from a chain of length
:math:`m_i` is right with probability :math:`1/m_i`, so the documented chance
baseline is :math:`\frac{1}{N}\sum_i 1/m_i` (reported on
:class:`SupersessionReport`). Beating that baseline is the evidence that a memory
system tracks supersession rather than guessing.

References
----------
.. [1] Alchourron, C. E., Gardenfors, P., & Makinson, D. (1985). "On the Logic of
       Theory Change: Partial Meet Contraction and Revision Functions." The Journal
       of Symbolic Logic, 50(2), 510-530. (The AGM recency/primacy-of-new-information
       principle that grounds "apply the current, not the superseded, decision".)
.. [2] Manning, C. D., Raghavan, P., & Schutze, H. (2008). Introduction to
       Information Retrieval, Ch. 8. (Accuracy as the fraction of correctly
       resolved cases, and the uniform-guess chance baseline.)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from membench.types import MemoryItem

__all__ = [
    "SUPERSEDES_EDGE",
    "SupersessionCase",
    "SupersessionReport",
    "case_from_items",
    "current_decision",
    "score_supersession",
    "supersession_rate",
]

#: Edge label (in ``MemoryItem.metadata["edges"]``) meaning "this item supersedes
#: the target item". An edge ``(newer_id) --SUPERSEDES_EDGE--> (older_id)`` records
#: that ``newer_id`` overrules ``older_id``.
SUPERSEDES_EDGE = "supersedes"


@dataclass(frozen=True, slots=True)
class SupersessionCase:
    """One decision chain plus the decision the system actually applied.

    Parameters
    ----------
    chain
        The decision ids participating in this case (any order). Must be
        non-empty and contain no duplicates.
    supersedes
        Directed edges ``(newer_id, older_id)`` meaning ``newer_id`` supersedes
        ``older_id``. Both endpoints must be members of ``chain``.
    applied
        The decision id the system actually applied, or ``None`` if it applied
        none of the chain's decisions (always counted as not recency-correct).
    """

    chain: tuple[str, ...]
    supersedes: tuple[tuple[str, str], ...]
    applied: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "chain", tuple(self.chain))
        object.__setattr__(self, "supersedes", tuple((n, o) for n, o in self.supersedes))
        if not self.chain:
            raise ValueError("SupersessionCase.chain must be non-empty")
        members = set(self.chain)
        if len(members) != len(self.chain):
            raise ValueError(f"SupersessionCase.chain has duplicate ids: {self.chain}")
        for newer, older in self.supersedes:
            if newer == older:
                raise ValueError(f"a decision cannot supersede itself: {newer!r}")
            if newer not in members or older not in members:
                raise ValueError(
                    f"supersedes edge ({newer!r}, {older!r}) references an id outside the chain"
                )


def current_decision(case: SupersessionCase) -> str:
    """Return the current (latest non-superseded) decision of a case.

    The current decision is the unique maximal element of the ``supersedes``
    partial order: the one decision that no other decision supersedes.

    Parameters
    ----------
    case
        The decision chain and its supersession edges.

    Returns
    -------
    str
        The id of the current decision.

    Raises
    ------
    ValueError
        If there is not exactly one non-superseded decision -- i.e. the chain is
        malformed (a cycle leaves none, or a branch leaves several with no single
        latest decision).
    """
    superseded = {older for _newer, older in case.supersedes}
    live = [decision for decision in case.chain if decision not in superseded]
    if len(live) != 1:
        raise ValueError(
            f"chain {case.chain} has {len(live)} non-superseded decisions ({live}); "
            "expected exactly one current decision"
        )
    return live[0]


def case_from_items(
    items: Sequence[MemoryItem],
    applied: str | None,
    *,
    edge_label: str = SUPERSEDES_EDGE,
) -> SupersessionCase:
    """Build a :class:`SupersessionCase` from memory items carrying edge metadata.

    Reads each item's ``metadata["edges"]`` (an iterable of ``(target_id, label)``
    pairs, per the repo's edge convention) and keeps only the edges labelled
    ``edge_label`` as supersession edges, where the item is the newer decision and
    the target is the older one.

    Parameters
    ----------
    items
        The decision nodes forming the chain.
    applied
        The decision id the system applied, or ``None``.
    edge_label
        The edge label that denotes supersession (defaults to
        :data:`SUPERSEDES_EDGE`).

    Returns
    -------
    SupersessionCase
        The reconstructed case.
    """
    chain = tuple(item.item_id for item in items)
    edges: list[tuple[str, str]] = []
    for item in items:
        raw_edges: Iterable[tuple[str, str]] = item.metadata.get("edges", ())
        for target_id, label in raw_edges:
            if label == edge_label:
                edges.append((item.item_id, str(target_id)))
    return SupersessionCase(chain=chain, supersedes=tuple(edges), applied=applied)


@dataclass(frozen=True, slots=True)
class SupersessionReport:
    """Aggregate recency-correctness over a set of supersession cases.

    Attributes
    ----------
    total
        Number of cases scored.
    recency_correct
        Number of cases in which the system applied the current decision.
    rate
        ``recency_correct / total`` in ``[0, 1]`` (0.0 on empty input).
    chance_baseline
        Mean per-case uniform-guess accuracy ``(1/total) * sum(1 / len(chain))``;
        the expected rate of a system that picks a decision at random. 0.0 on
        empty input.
    """

    total: int
    recency_correct: int
    rate: float
    chance_baseline: float


def supersession_rate(cases: Sequence[SupersessionCase]) -> float:
    """Fraction of cases in which the system applied the current decision.

    Parameters
    ----------
    cases
        The supersession cases to score.

    Returns
    -------
    float
        The recency-correct rate in ``[0, 1]``. Returns ``0.0`` on empty input
        (vacuously, keeping the result inside ``[0, 1]``).
    """
    if not cases:
        return 0.0
    correct = sum(
        1 for case in cases if case.applied is not None and case.applied == current_decision(case)
    )
    return correct / len(cases)


def score_supersession(cases: Sequence[SupersessionCase]) -> SupersessionReport:
    """Compute the full supersession report (rate plus the chance baseline).

    Parameters
    ----------
    cases
        The supersession cases to score.

    Returns
    -------
    SupersessionReport
        Totals, the recency-correct ``rate``, and the documented uniform-guess
        ``chance_baseline``. On empty input all fields are zero.
    """
    if not cases:
        return SupersessionReport(total=0, recency_correct=0, rate=0.0, chance_baseline=0.0)
    correct = sum(
        1 for case in cases if case.applied is not None and case.applied == current_decision(case)
    )
    total = len(cases)
    baseline = sum(1.0 / len(case.chain) for case in cases) / total
    return SupersessionReport(
        total=total,
        recency_correct=correct,
        rate=correct / total,
        chance_baseline=baseline,
    )
