"""The MemorySystem contract: every arm (none, fullcontext, mem0, brief,
random_context) implements exactly these two methods and nothing else. That
uniformity is what lets run.py treat the memory system as a swappable variable
while everything else about a task stays fixed."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from benchmarks.task import MemoryItem


@dataclass
class RetrievedContext:
    """What retrieve() hands back to the agent.

    text is the assembled context string injected into the prompt. ids is the
    list of MemoryItem.item_id values that contributed to text. ids must never
    be dropped: scoring computes retrieval precision/recall by comparing ids
    against Task.governing_decisions, and that comparison is impossible if an
    arm only returns prose.
    """

    text: str
    ids: list[str]


class MemorySystem(ABC):
    """Base class every arm subclasses.

    budget_tokens is a parameter of retrieve(), not a property of the arm: the
    caller (run.py) passes the same budget to every arm for a given task, so
    that memory architecture is the only variable being compared. An arm must
    never read its own budget from config or pick one internally.
    """

    @abstractmethod
    def write(self, items: list[MemoryItem]) -> None:
        """Ingest a task's memory_corpus before retrieval happens."""
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Return assembled context for query, truncated to budget_tokens."""
        raise NotImplementedError
