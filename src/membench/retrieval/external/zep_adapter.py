"""Adapter for Zep / Graphiti, a temporal knowledge-graph memory.

Graphiti (the engine behind Zep's memory) builds a temporally-aware knowledge
graph from ingested episodes and retrieves via hybrid graph + semantic search. It
is the closest *graph-based* competitor to Brief, which makes it an especially
important comparison: it tests whether a generic knowledge graph closes the depth
gap, or whether Brief's *typed, decision-centric* graph is what matters.

Graphiti is async and backed by a graph database; the real wrapper drives it via
``asyncio.run`` and keys each episode by ``item_id``. Requires the isolated
``graphiti-core`` env (and a Neo4j backend); the injected-client path is unit
tested without either.
"""

from __future__ import annotations

import asyncio
from typing import Any

from membench.retrieval.base import ArmFamily, register_arm
from membench.retrieval.external._base import KeyedExternalAdapter

__all__ = ["ZepGraphitiMemory"]


class _RealGraphiti:
    """Wrap Graphiti's async episode/search API to the internal add/search contract."""

    def __init__(self) -> None:
        try:
            from graphiti_core import Graphiti
        except ImportError as exc:
            raise RuntimeError(
                "Zep/Graphiti adapter needs graphiti-core in an isolated env (+ Neo4j): "
                "uv pip install -r competitors/envs/zep.txt"
            ) from exc
        self._graphiti = Graphiti()  # reads NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD

    def add(self, item_id: str, text: str) -> None:
        asyncio.run(
            self._graphiti.add_episode(
                name=item_id, episode_body=text, source_description="membench"
            )
        )

    def search(self, query: str, k: int) -> list[str]:
        results = asyncio.run(self._graphiti.search(query, num_results=k))
        ids: list[str] = []
        for result in results:
            name = getattr(result, "name", None) or getattr(result, "source_node_uuid", None)
            if name is not None:
                ids.append(str(name))
        return ids


@register_arm(
    "zep",
    family=ArmFamily.EXTERNAL,
    follows_links=True,  # Graphiti is graph-based, so it does follow edges
    uses_embeddings=True,
    requires_network=True,
    deterministic=False,
)
class ZepGraphitiMemory(KeyedExternalAdapter):
    """Zep/Graphiti-backed temporal-knowledge-graph memory arm.

    Parameters
    ----------
    client
        Object exposing ``add(item_id, text)`` and ``search(query, k) -> list[str]``;
        defaults to a lazily-constructed real Graphiti wrapper.
    """

    def __init__(self, client: Any | None = None) -> None:
        super().__init__()
        self._client = client

    def _ensure(self) -> Any:
        if self._client is None:
            self._client = _RealGraphiti()
        return self._client

    def _backend_add(self, item_id: str, text: str) -> str:
        self._ensure().add(item_id, text)
        return item_id

    def _backend_search(self, query: str, k: int) -> list[str]:
        results: list[str] = self._ensure().search(query, k)
        return results
