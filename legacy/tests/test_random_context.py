from adapters.random_context import RandomContextMemorySystem
from benchmarks.task import MemoryItem


def _items(n):
    return [MemoryItem(item_id=f"d{i}", text="x" * 40) for i in range(n)]  # ~10 tokens each


def test_random_context_stops_at_budget():
    mem = RandomContextMemorySystem(seed=0)
    mem.write(_items(10))

    result = mem.retrieve("anything", budget_tokens=25)

    assert len(result.ids) <= 2


def test_random_context_is_reproducible_with_a_fixed_seed():
    items = _items(10)
    mem_a = RandomContextMemorySystem(seed=42)
    mem_a.write(items)
    mem_b = RandomContextMemorySystem(seed=42)
    mem_b.write(items)

    assert mem_a.retrieve("q", budget_tokens=1000).ids == mem_b.retrieve("q", budget_tokens=1000).ids


def test_random_context_ignores_the_query():
    items = _items(5)
    mem_a = RandomContextMemorySystem(seed=1)
    mem_a.write(items)
    mem_b = RandomContextMemorySystem(seed=1)
    mem_b.write(items)

    assert (
        mem_a.retrieve("query one", 1000).ids
        == mem_b.retrieve("a totally different query", 1000).ids
    )
