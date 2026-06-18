"""Adapter for Letta (formerly MemGPT), an agent-memory framework.

Letta gives an agent self-editing core memory plus a searchable *archival* memory.
We use the archival store as the benchmark memory: insert each item (tagged with
its ``item_id``) and retrieve via archival search. It represents the
"agent-managed memory" school rather than a retrieval index.

Requires the isolated ``letta-client`` env (and a Letta server); the injected
client path is unit-tested without either.
"""

from __future__ import annotations

from typing import Any

from membench.retrieval.base import ArmFamily, register_arm
from membench.retrieval.external._base import KeyedExternalAdapter

__all__ = ["LettaMemory"]


class _RealLetta:
    """Wrap Letta's archival-memory insert/search to the internal add/search contract."""

    def __init__(self) -> None:
        try:
            from letta_client import Letta
        except ImportError as exc:
            raise RuntimeError(
                "Letta adapter needs letta-client in an isolated env (+ a Letta server): "
                "uv pip install -r competitors/envs/letta.txt"
            ) from exc
        self._client = Letta()
        self._agent = self._client.agents.create(name="membench")

    def add(self, item_id: str, text: str) -> None:
        self._client.agents.passages.create(agent_id=self._agent.id, text=f"[{item_id}] {text}")

    def search(self, query: str, k: int) -> list[str]:
        passages = self._client.agents.passages.search(
            agent_id=self._agent.id, query=query, limit=k
        )
        ids: list[str] = []
        for passage in passages:
            text = getattr(passage, "text", "")
            if text.startswith("[") and "]" in text:
                ids.append(text[1 : text.index("]")])
        return ids


@register_arm(
    "letta",
    family=ArmFamily.EXTERNAL,
    follows_links=False,
    uses_embeddings=True,
    requires_network=True,
    deterministic=False,
)
class LettaMemory(KeyedExternalAdapter):
    """Letta/MemGPT archival-memory-backed arm.

    Parameters
    ----------
    client
        Object exposing ``add(item_id, text)`` and ``search(query, k) -> list[str]``;
        defaults to a lazily-constructed real Letta wrapper.
    """

    def __init__(self, client: Any | None = None) -> None:
        super().__init__()
        self._client = client

    def _ensure(self) -> Any:
        if self._client is None:
            self._client = _RealLetta()
        return self._client

    def _backend_add(self, item_id: str, text: str) -> str:
        self._ensure().add(item_id, text)
        return item_id

    def _backend_search(self, query: str, k: int) -> list[str]:
        results: list[str] = self._ensure().search(query, k)
        return results
