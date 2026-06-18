"""mem0 arm: similarity top-k retrieval filled to budget_tokens.

Stands in for a real Mem0-style similarity memory system without adding an
embeddings API dependency: write() and retrieve() use plain term-frequency
cosine similarity (stdlib only). The mechanism under test in the ablation
is "rank candidates by resemblance to the query," not the specific embedding
model behind it, so a lexical similarity is a faithful, dependency-free
stand-in for this arm.

Items are ranked by similarity to the query, then included greedily in rank
order until the next item would exceed budget_tokens -- the same
truncate-at-first-overflow rule adapters/fullcontext.py uses, just applied
to a similarity-sorted list instead of corpus order. "k" isn't fixed; it
falls out of how many top-ranked items fit the budget.
"""

import math
import re
from collections import Counter

from adapters.base import MemorySystem, RetrievedContext
from benchmarks.task import MemoryItem

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
_CHARS_PER_TOKEN = 4  # same heuristic as adapters/fullcontext.py


def _term_vector(text: str) -> Counter:
    return Counter(t.lower() for t in _TOKEN_PATTERN.findall(text))


def _cosine(a: Counter, b: Counter) -> float:
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    dot = sum(a[t] * b[t] for t in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    return dot / (norm_a * norm_b)


def _approx_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


class Mem0MemorySystem(MemorySystem):
    def __init__(self):
        self._items: list[MemoryItem] = []

    def write(self, items: list[MemoryItem]) -> None:
        self._items.extend(items)

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        query_vector = _term_vector(query)
        ranked = sorted(
            self._items,
            key=lambda item: _cosine(query_vector, _term_vector(item.text)),
            reverse=True,
        )

        included_text: list[str] = []
        included_ids: list[str] = []
        used_tokens = 0
        for item in ranked:
            item_tokens = _approx_tokens(item.text)
            if used_tokens + item_tokens > budget_tokens:
                break
            included_text.append(item.text)
            included_ids.append(item.item_id)
            used_tokens += item_tokens

        return RetrievedContext(text="\n\n".join(included_text), ids=included_ids)
