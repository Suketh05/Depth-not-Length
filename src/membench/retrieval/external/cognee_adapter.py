"""Adapter for Cognee, an LLM "memory engine" that builds a knowledge graph.

Cognee ingests text, cognifies it into a graph + vector store, and answers
queries with graph-aware search. Another graph-flavoured competitor; its module-
level API is async, so the real wrapper drives it via ``asyncio.run`` and recovers
``item_id`` values from the search payload.

Requires the isolated ``cognee`` env (and an LLM key); the injected-client path is
unit-tested without either.
"""

from __future__ import annotations

import asyncio
from typing import Any

from membench.retrieval.base import ArmFamily, register_arm
from membench.retrieval.external._base import KeyedExternalAdapter

__all__ = ["CogneeMemory"]


class _RealCognee:
    """Wrap Cognee's async add/cognify/search API to the internal add/search contract."""

    def __init__(self) -> None:
        try:
            import cognee  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Cognee adapter needs the cognee package in an isolated env (+ LLM key): "
                "uv pip install -r competitors/envs/cognee.txt"
            ) from exc

    def add(self, item_id: str, text: str) -> None:
        import cognee

        asyncio.run(cognee.add(f"[{item_id}] {text}"))

    def search(self, query: str, k: int) -> list[str]:  # pragma: no cover - needs cognee + LLM
        import cognee

        asyncio.run(cognee.cognify())
        results = asyncio.run(cognee.search(query_text=query))
        ids: list[str] = []
        for result in results[:k]:
            text = str(result)
            if text.startswith("[") and "]" in text:
                ids.append(text[1 : text.index("]")])
        return ids


@register_arm(
    "cognee",
    family=ArmFamily.EXTERNAL,
    follows_links=True,
    uses_embeddings=True,
    requires_network=True,
    deterministic=False,
)
class CogneeMemory(KeyedExternalAdapter):
    """Cognee-backed knowledge-graph memory arm.

    Parameters
    ----------
    client
        Object exposing ``add(item_id, text)`` and ``search(query, k) -> list[str]``;
        defaults to a lazily-constructed real Cognee wrapper.
    """

    def __init__(self, client: Any | None = None) -> None:
        super().__init__()
        self._client = client

    def _ensure(self) -> Any:
        if self._client is None:
            self._client = _RealCognee()
        return self._client

    def _backend_add(self, item_id: str, text: str) -> str:
        self._ensure().add(item_id, text)
        return item_id

    def _backend_search(self, query: str, k: int) -> list[str]:
        results: list[str] = self._ensure().search(query, k)
        return results
