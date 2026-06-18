"""Adapter for Mem0 (mem0ai), a popular LLM-extraction memory system.

Mem0 extracts and consolidates salient facts from ingested text with an LLM and
retrieves them by semantic search. It is a flagship "AI memory" product and a
direct competitor to Brief's context layer on the similarity axis. We key each
ingested item with its ``item_id`` in Mem0 metadata so search results map straight
back to our gold ids.

Requires the optional, isolated ``mem0ai`` package (``competitors/envs/mem0.txt``);
the adapter accepts an injected client so its logic is unit-tested without it.
"""

from __future__ import annotations

from typing import Any

from membench.retrieval.base import ArmFamily, register_arm
from membench.retrieval.external._base import KeyedExternalAdapter

__all__ = ["Mem0Memory"]


@register_arm(
    "mem0",
    family=ArmFamily.EXTERNAL,
    follows_links=False,
    uses_embeddings=True,
    requires_network=True,
    deterministic=False,
)
class Mem0Memory(KeyedExternalAdapter):
    """Mem0-backed memory arm (semantic extraction + search).

    Parameters
    ----------
    client
        An object exposing ``add(text, user_id, metadata)`` and
        ``search(query, user_id, limit)``. Defaults to a real ``mem0.Memory``,
        constructed lazily on first use.
    user_id
        Mem0 partition key for this benchmark run.
    """

    def __init__(self, client: Any | None = None, user_id: str = "membench") -> None:
        super().__init__()
        self._client = client
        self._user_id = user_id

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from mem0 import Memory
            except ImportError as exc:
                raise RuntimeError(
                    "Mem0 adapter needs the mem0ai package in an isolated env: "
                    "uv pip install -r competitors/envs/mem0.txt"
                ) from exc
            self._client = Memory()
        return self._client

    def _backend_add(self, item_id: str, text: str) -> str:
        self._ensure_client().add(text, user_id=self._user_id, metadata={"item_id": item_id})
        return item_id  # we key by our own id via Mem0 metadata

    def _backend_search(self, query: str, k: int) -> list[str]:
        result = self._ensure_client().search(query, user_id=self._user_id, limit=k)
        rows = result.get("results", result) if isinstance(result, dict) else result
        ids: list[str] = []
        for row in rows:
            metadata = row.get("metadata") or {}
            item_id = metadata.get("item_id")
            if item_id is not None:
                ids.append(str(item_id))
        return ids
