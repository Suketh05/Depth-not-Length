"""random_context arm: the negative control. Random nodes from the corpus,
filled to the same budget_tokens every other arm receives -- including
Brief's. If random selection scores as well as Brief, the win was buying
tokens, not structure; if it doesn't (the expected result), that's the
evidence that what wins is *which* nodes get walked, not how many fit.

Per CLAUDE.md, random_context's budget must equal Brief's exactly. Nothing
in this file enforces that itself -- budget_tokens is the caller-supplied
invariant every arm shares (adapters/base.py), so equality is automatic as
long as the caller (run.py / configs) passes the same budget_tokens to every
arm in a run. That's the whole point of budget_tokens being an input rather
than something an arm picks for itself.
"""

from __future__ import annotations

import random

from adapters.base import MemorySystem, RetrievedContext
from benchmarks.task import MemoryItem

_CHARS_PER_TOKEN = 4  # same heuristic as the other arms


def _approx_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


class RandomContextMemorySystem(MemorySystem):
    def __init__(self, seed: int | None = None):
        self._items: list[MemoryItem] = []
        self._rng = random.Random(seed)

    def write(self, items: list[MemoryItem]) -> None:
        self._items.extend(items)

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        # query is deliberately unused -- selection must not depend on
        # relevance, or this isn't the control it's supposed to be.
        shuffled = self._items[:]
        self._rng.shuffle(shuffled)

        included_text: list[str] = []
        included_ids: list[str] = []
        used_tokens = 0
        for item in shuffled:
            item_tokens = _approx_tokens(item.text)
            if used_tokens + item_tokens > budget_tokens:
                break
            included_text.append(item.text)
            included_ids.append(item.item_id)
            used_tokens += item_tokens

        return RetrievedContext(text="\n\n".join(included_text), ids=included_ids)
