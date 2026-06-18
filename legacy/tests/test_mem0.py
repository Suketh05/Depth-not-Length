from adapters.mem0 import Mem0MemorySystem
from benchmarks.task import MemoryItem


def test_mem0_ranks_by_similarity_not_corpus_order():
    mem = Mem0MemorySystem()
    mem.write(
        [
            MemoryItem(item_id="unrelated", text="zebra giraffe lion savanna"),
            MemoryItem(item_id="relevant", text="cursor pagination api endpoint"),
        ]
    )

    result = mem.retrieve("add cursor pagination to the api endpoint", budget_tokens=1000)

    assert result.ids[0] == "relevant"


def test_mem0_stops_at_budget_even_with_relevant_items_remaining():
    mem = Mem0MemorySystem()
    mem.write(
        [
            MemoryItem(item_id="a", text="x" * 40),  # ~10 tokens
            MemoryItem(item_id="b", text="x" * 40),
        ]
    )

    result = mem.retrieve("x", budget_tokens=10)

    assert result.ids == ["a"]
