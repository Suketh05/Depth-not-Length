"""The three control arms: ``none``, ``full_context``, and ``random_context``.

These bracket every other arm. ``none`` is the floor (what the model gets right
from the query alone). ``full_context`` is the "just buy tokens" control that dumps
the whole corpus in corpus order. ``random_context`` is the negative control that
fills the *same* budget with randomly chosen items — if it scores as well as a
real retriever, the win was buying tokens, not structure. All three share the
budget-packing primitive, so they differ only in how (or whether) they rank.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

from membench.retrieval.base import ArmFamily, MemorySystem, pack_to_budget, register_arm
from membench.types import MemoryItem, RetrievedContext

__all__ = ["FullContextMemory", "NoneMemory", "RandomContextMemory"]


@register_arm("none", family=ArmFamily.CONTROL, follows_links=False, uses_embeddings=False)
class NoneMemory(MemorySystem):
    """No memory at all: ingests nothing, retrieves nothing (the floor control)."""

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Discard the corpus; this arm has no memory."""
        del items

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Return empty context regardless of query or budget."""
        del query, budget_tokens
        return RetrievedContext.empty()


@register_arm("full_context", family=ArmFamily.CONTROL, follows_links=False, uses_embeddings=False)
class FullContextMemory(MemorySystem):
    """Dump the whole corpus in corpus order, truncated to budget (no retrieval)."""

    def __init__(self) -> None:
        self._items: list[MemoryItem] = []

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Store the corpus to be replayed in order at retrieval time."""
        self._items.extend(items)

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Pack the corpus in its original order up to the budget."""
        del query
        return pack_to_budget(((i.item_id, i.text) for i in self._items), budget_tokens)


@register_arm(
    "random_context", family=ArmFamily.CONTROL, follows_links=False, uses_embeddings=False
)
class RandomContextMemory(MemorySystem):
    """Fill the budget with randomly ordered items (the negative control).

    The query is deliberately ignored: selection must not depend on relevance, or
    this would no longer isolate "budget, not structure". A seed makes the shuffle
    reproducible across runs.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._items: list[MemoryItem] = []
        self._rng = random.Random(seed)

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Store the corpus to be shuffled at retrieval time."""
        self._items.extend(items)

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Pack a random permutation of the corpus up to the budget."""
        del query
        shuffled = self._items[:]
        self._rng.shuffle(shuffled)
        return pack_to_budget(((i.item_id, i.text) for i in shuffled), budget_tokens)
