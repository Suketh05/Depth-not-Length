"""Interface tests for the two frozen contracts. These don't test any arm's
retrieval quality -- just that the shapes hold: IDs survive retrieval, and
budget is accepted as an input rather than chosen internally."""

import pytest

from adapters.base import MemorySystem, RetrievedContext
from benchmarks.task import MemoryItem, Task


class DummyMemorySystem(MemorySystem):
    """Minimal concrete arm: stores everything, echoes it back regardless of
    budget. Enough to prove the abstract contract is satisfiable."""

    def __init__(self):
        self._items: list[MemoryItem] = []

    def write(self, items: list[MemoryItem]) -> None:
        self._items.extend(items)

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        return RetrievedContext(
            text="\n".join(item.text for item in self._items),
            ids=[item.item_id for item in self._items],
        )


def test_dummy_memory_system_satisfies_interface():
    mem = DummyMemorySystem()
    items = [
        MemoryItem(item_id="d1", text="decision one"),
        MemoryItem(item_id="d2", text="decision two"),
    ]
    mem.write(items)

    result = mem.retrieve(query="what did we decide?", budget_tokens=100)

    assert isinstance(result, RetrievedContext)
    assert result.ids == ["d1", "d2"]
    assert "decision one" in result.text


def test_retrieved_context_ids_survive_even_with_tiny_budget():
    # Budget is an input the caller controls; a real arm may truncate text,
    # but ids must still name what was retrieved so precision/recall can be
    # scored against governing_decisions.
    mem = DummyMemorySystem()
    mem.write([MemoryItem(item_id="d1", text="x" * 1000)])

    result = mem.retrieve(query="q", budget_tokens=1)

    assert result.ids == ["d1"]


def test_memory_system_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        MemorySystem()


def test_task_dataclass_shape():
    item = MemoryItem(item_id="d1", text="some decision")
    task = Task(
        task_id="t1",
        dataset="dcbench",
        query="implement the pricing rule",
        repo_ref="repo@main",
        memory_corpus=[item],
        governing_decisions=["d1"],
        depth=2,
        spec_variant="stripped",
        scorer="compliance",
    )

    assert task.repo_ref == "repo@main"
    assert task.memory_corpus[0].item_id == "d1"


def test_task_repo_ref_none_for_longmemeval():
    task = Task(
        task_id="lme-1",
        dataset="longmemeval",
        query="what did I say my deadline was?",
        repo_ref=None,
        memory_corpus=[],
        governing_decisions=[],
        depth=1,
        spec_variant="full",
        scorer="correctness",
    )

    assert task.repo_ref is None
