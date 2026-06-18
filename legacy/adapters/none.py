"""The floor arm: no memory at all. Ingests nothing, retrieves nothing.

Exists so the headline table has a baseline -- whatever compliance shows up
here is what the model gets right from the query alone, with zero help.
"""

from adapters.base import MemorySystem, RetrievedContext
from benchmarks.task import MemoryItem


class NoneMemorySystem(MemorySystem):
    def write(self, items: list[MemoryItem]) -> None:
        pass

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        return RetrievedContext(text="", ids=[])
