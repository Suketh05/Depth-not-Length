r"""Offline protocol model of Oiya as evaluated in the paper (tab:stdbench).

Oiya appears in the paper only as a competitor *product name* in the unified-harness field,
where it posts the weakest ranking quality of the field (tab:stdbench HotpotQA MRR 0.515, the
lowest column; also the smallest session-token win-rate gap, tab:tok_context_by_competitor
68.6%). The paper states no internal mechanism for it. Consistent with that profile we model
it as a *recency* store -- a raw chat/event log that returns the most recently written items
first, independent of the query (the "Chat log" organisation the org sweep places at high
scatter, tab:org). This is an offline, deterministic stand-in, **not** a reproduction of the
vendor's internals nor of the paper's measured tab:stdbench numbers (those come from the full
public-benchmark harness).

A recency order carries no query signal, so it has no similarity ceiling to speak of -- it
sits at chance on ranking and, by the compliance factorization P_comply = P_ret * kappa
(Equation eq:factor), floors P_ret whenever the governing item is not among the most recent
writes. The arm is runnable with no network or API key.
"""

from __future__ import annotations

from collections.abc import Iterable

from membench.retrieval.base import ArmFamily, MemorySystem, pack_to_budget, register_arm
from membench.types import MemoryItem, RetrievedContext

__all__ = ["OiyaMemory"]


@register_arm(
    "oiya",
    family=ArmFamily.EXTERNAL,
    follows_links=False,
    uses_embeddings=False,
    requires_network=False,
    deterministic=True,
)
class OiyaMemory(MemorySystem):
    """Recency-ordered chat-log store (offline protocol model of Oiya).

    ``write`` appends items to an event log; ``retrieve`` returns them most-recent-first,
    ignoring the query entirely, and packs the result to budget. Two different queries
    therefore receive the same ordering -- the defining behaviour of a raw recency log.
    """

    def __init__(self) -> None:
        self._items: list[MemoryItem] = []

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Append items to the event log in write order."""
        self._items.extend(items)

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Return items most-recent-first (query-independent), packed to budget."""
        del query  # a recency log carries no query signal by construction
        if not self._items:
            return RetrievedContext.empty()
        ranked = [(item.item_id, item.text) for item in reversed(self._items)]
        return pack_to_budget(ranked, budget_tokens)
