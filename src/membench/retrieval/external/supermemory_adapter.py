"""Adapter for Supermemory, a hosted "universal memory" API.

Supermemory ingests content and serves it back to agents over an API by semantic
search — positioned, like Brief, as a context layer for AI applications, but
similarity-based. We tag each item with its ``item_id`` so results map back to our
gold ids.

Requires the optional ``supermemory`` SDK (``competitors/envs/supermemory.txt``)
and an API key; the adapter accepts an injected client so its logic is unit-tested
without the network.
"""

from __future__ import annotations

import os
from typing import Any

from membench.retrieval.base import ArmFamily, register_arm
from membench.retrieval.external._base import KeyedExternalAdapter

__all__ = ["SupermemoryMemory"]


class _RealSupermemory:
    """Wraps the Supermemory SDK to the internal add/search-by-item-id contract."""

    def __init__(self, api_key: str | None) -> None:
        try:
            from supermemory import Supermemory
        except ImportError as exc:
            raise RuntimeError(
                "Supermemory adapter needs the supermemory SDK in an isolated env: "
                "uv pip install -r competitors/envs/supermemory.txt"
            ) from exc
        self._client = Supermemory(api_key=api_key or os.environ.get("SUPERMEMORY_API_KEY"))

    def add(self, item_id: str, text: str) -> None:
        self._client.memories.add(content=text, metadata={"item_id": item_id})

    def search(self, query: str, k: int) -> list[str]:
        response = self._client.search.execute(q=query, limit=k)
        ids: list[str] = []
        for result in getattr(response, "results", []) or []:
            metadata = getattr(result, "metadata", {}) or {}
            item_id = metadata.get("item_id")
            if item_id is not None:
                ids.append(str(item_id))
        return ids


@register_arm(
    "supermemory",
    family=ArmFamily.EXTERNAL,
    follows_links=False,
    uses_embeddings=True,
    requires_network=True,
    deterministic=False,
)
class SupermemoryMemory(KeyedExternalAdapter):
    """Supermemory-backed memory arm.

    Parameters
    ----------
    client
        Object exposing ``add(item_id, text)`` and ``search(query, k) -> list[str]``.
        Defaults to a lazily-constructed real Supermemory client wrapper.
    api_key
        API key (falls back to ``SUPERMEMORY_API_KEY``).
    """

    def __init__(self, client: Any | None = None, api_key: str | None = None) -> None:
        super().__init__()
        self._client = client
        self._api_key = api_key

    def _ensure(self) -> Any:
        if self._client is None:
            self._client = _RealSupermemory(self._api_key)
        return self._client

    def _backend_add(self, item_id: str, text: str) -> str:
        self._ensure().add(item_id, text)
        return item_id

    def _backend_search(self, query: str, k: int) -> list[str]:
        results: list[str] = self._ensure().search(query, k)
        return results
