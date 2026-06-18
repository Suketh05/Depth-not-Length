"""The "just buy tokens" control: dump the whole corpus into context, truncated
to budget_tokens. No retrieval logic at all -- this is the brute-force baseline
that the structured arms have to beat at equal or lower cost, per the 4-arm
ablation and the depth-ceiling thesis in docs/handoff.md.
"""

from adapters.base import MemorySystem, RetrievedContext
from benchmarks.task import MemoryItem

# Rough chars-per-token heuristic. Good enough for a budget-truncation arm
# whose whole point is "no real retrieval logic"; real per-call token counts
# for scoring come from the LLM API response in agent/llm_client.py, not here.
_CHARS_PER_TOKEN = 4


def _approx_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


class FullContextMemorySystem(MemorySystem):
    def __init__(self):
        self._items: list[MemoryItem] = []

    def write(self, items: list[MemoryItem]) -> None:
        self._items.extend(items)

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        included_text: list[str] = []
        included_ids: list[str] = []
        used_tokens = 0

        for item in self._items:
            item_tokens = _approx_tokens(item.text)
            if used_tokens + item_tokens > budget_tokens:
                break
            included_text.append(item.text)
            included_ids.append(item.item_id)
            used_tokens += item_tokens

        return RetrievedContext(text="\n\n".join(included_text), ids=included_ids)
