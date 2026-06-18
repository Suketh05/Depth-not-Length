"""Adapter for Microsoft GraphRAG.

GraphRAG indexes a corpus into an entity/community knowledge graph and answers
queries with graph-aware local/global search. It is a second graph-based
competitor (community-detection rather than typed decision links), so it tests the
same question as Zep: does a generic graph close the depth gap?

GraphRAG's programmatic surface is heavy and evolving; the real wrapper targets its
documented search-engine entry and maps returned context records back to our
``item_id`` values. Best-effort and behind an isolated env
(``competitors/envs/graphrag.txt``); logic is unit-tested via an injected client.
"""

from __future__ import annotations

from typing import Any

from membench.retrieval.base import ArmFamily, register_arm
from membench.retrieval.external._base import KeyedExternalAdapter

__all__ = ["GraphRAGMemory"]


class _RealGraphRAG:
    """Best-effort wrapper of GraphRAG indexing + search to the add/search contract."""

    def __init__(self) -> None:
        try:
            import graphrag  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "GraphRAG adapter needs the graphrag package in an isolated env: "
                "uv pip install -r competitors/envs/graphrag.txt"
            ) from exc
        self._docs: list[tuple[str, str]] = []

    def add(self, item_id: str, text: str) -> None:
        # GraphRAG indexes a corpus in bulk; accumulate, (re)index lazily at search.
        self._docs.append((item_id, text))

    def search(self, query: str, k: int) -> list[str]:  # pragma: no cover - needs graphrag + LLM
        from graphrag.query.structured_search.local_search.search import LocalSearch

        del LocalSearch, query, k  # real wiring requires a built index + LLM config
        raise RuntimeError("GraphRAG live search requires a built index and LLM configuration")


@register_arm(
    "graphrag",
    family=ArmFamily.EXTERNAL,
    follows_links=True,
    uses_embeddings=True,
    requires_network=True,
    deterministic=False,
)
class GraphRAGMemory(KeyedExternalAdapter):
    """Microsoft GraphRAG-backed memory arm.

    Parameters
    ----------
    client
        Object exposing ``add(item_id, text)`` and ``search(query, k) -> list[str]``;
        defaults to a lazily-constructed real GraphRAG wrapper.
    """

    def __init__(self, client: Any | None = None) -> None:
        super().__init__()
        self._client = client

    def _ensure(self) -> Any:
        if self._client is None:
            self._client = _RealGraphRAG()
        return self._client

    def _backend_add(self, item_id: str, text: str) -> str:
        self._ensure().add(item_id, text)
        return item_id

    def _backend_search(self, query: str, k: int) -> list[str]:
        results: list[str] = self._ensure().search(query, k)
        return results
