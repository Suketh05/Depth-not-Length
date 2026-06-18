"""Shared base for third-party memory adapters exposed through add/search.

Most external memory systems share one shape: ingest text items, then search by
query. This base captures that shape and the bookkeeping every adapter needs —
mapping the external system's identifiers back to our stable ``item_id`` so
retrieval precision/recall is well defined — leaving each concrete adapter to
implement just two backend hooks. Ranking is whatever order the backend returns;
the shared budget primitive then truncates it identically to every other arm.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable

from membench.retrieval.base import MemorySystem, pack_to_budget
from membench.types import MemoryItem, RetrievedContext

__all__ = ["KeyedExternalAdapter"]


class KeyedExternalAdapter(MemorySystem):
    """A MemorySystem backed by an external add/search store.

    Subclasses implement :meth:`_backend_add` (ingest one item, return the external
    id) and :meth:`_backend_search` (return external ids best-first). This base maps
    external ids back to our ``item_id`` values and packs the result to budget.
    """

    def __init__(self) -> None:
        self._text_by_id: dict[str, str] = {}
        self._ext_to_item: dict[str, str] = {}

    def _search_k(self) -> int:
        """Return the number of candidates to request from the backend (override if needed)."""
        return 50

    @abstractmethod
    def _backend_add(self, item_id: str, text: str) -> str:
        """Ingest one item into the backend; return its external identifier."""
        raise NotImplementedError

    @abstractmethod
    def _backend_search(self, query: str, k: int) -> list[str]:
        """Return up to ``k`` external identifiers for ``query``, best first."""
        raise NotImplementedError

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Ingest the corpus into the backend, recording the id mapping."""
        for item in items:
            self._text_by_id[item.item_id] = item.text
            ext_id = self._backend_add(item.item_id, item.text)
            self._ext_to_item[str(ext_id)] = item.item_id

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Search the backend, map results to our ids, and pack to budget."""
        if not self._text_by_id:
            return RetrievedContext.empty()
        ranked: list[tuple[str, str]] = []
        seen: set[str] = set()
        for ext_id in self._backend_search(query, self._search_k()):
            item_id = self._ext_to_item.get(str(ext_id))
            if item_id is None or item_id in seen:
                continue
            seen.add(item_id)
            ranked.append((item_id, self._text_by_id[item_id]))
        return pack_to_budget(ranked, budget_tokens)
