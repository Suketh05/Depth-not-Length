"""Optional live-Brief arm: query a real Brief workspace over MCP.

The local graph arm (:mod:`membench.retrieval.graph.arm`) is the reproducible,
load-bearing structured arm and produces the reported numbers. This arm exists for
*ecological validity*: it confirms the same mechanism against the real Brief
product over the Model Context Protocol, using a pre-seeded workspace.

It is deliberately kept off the reproducible path. ``write`` is a no-op — Brief's
ingestion is interactive and approval-gated, so the workspace must be pre-seeded
out of band (the real dcbench harness seeds once via a separate step). ``retrieve``
calls the live ``brief_search`` tool over Streamable HTTP, authenticating with a
token read from the environment, never hardcoded. The arm requires the optional
``mcp`` extra and network access, and is registered as non-deterministic so the
analysis layer keeps it separate from the measured arms.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Iterable
from typing import Any

from membench.retrieval.base import ArmFamily, MemorySystem, pack_to_budget, register_arm
from membench.types import MemoryItem, RetrievedContext

__all__ = ["BriefLiveMemory"]

_DEFAULT_ENDPOINT = "https://app.briefhq.ai/mcp"
_SEARCH_LIMIT = 50


def _extract_matches(result: Any) -> list[dict[str, Any]]:
    """Best-effort parse of a brief_search CallToolResult into a list of records."""
    structured = getattr(result, "structuredContent", None)
    if structured:
        if isinstance(structured, dict):
            got = structured.get("results", structured)
            return got if isinstance(got, list) else []
        return structured if isinstance(structured, list) else []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            got = parsed.get("results", parsed)
            return got if isinstance(got, list) else []
        return parsed if isinstance(parsed, list) else []
    return []


async def _search(endpoint: str, token: str, query: str, limit: int) -> list[dict[str, Any]]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with (
        streamablehttp_client(endpoint, headers={"Authorization": f"Bearer {token}"}) as (
            read,
            write,
            _,
        ),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        result = await session.call_tool(
            "brief_search", {"query": query, "types": "decision", "limit": limit}
        )
        return _extract_matches(result)


@register_arm(
    "brief_live",
    family=ArmFamily.GRAPH,
    follows_links=True,
    uses_embeddings=True,
    requires_network=True,
    deterministic=False,
)
class BriefLiveMemory(MemorySystem):
    """Retrieve from a live, pre-seeded Brief workspace via the MCP ``brief_search`` tool.

    Parameters
    ----------
    token
        Brief OAuth token; falls back to the ``BRIEF_OAUTH_TOKEN`` environment
        variable. Raised if absent (this arm reads its credential, never hardcodes).
    endpoint
        MCP Streamable-HTTP endpoint.
    """

    def __init__(self, token: str | None = None, endpoint: str = _DEFAULT_ENDPOINT) -> None:
        self._token = token or os.environ.get("BRIEF_OAUTH_TOKEN")
        if not self._token:
            raise RuntimeError(
                "brief_live needs a Brief OAuth token (arg or BRIEF_OAUTH_TOKEN); "
                "it is never hardcoded."
            )
        self._endpoint = endpoint

    def write(self, items: Iterable[MemoryItem]) -> None:
        """No-op: the Brief workspace is pre-seeded out of band (see module docstring)."""
        del items

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Query live Brief and pack the returned records into the budget."""
        assert self._token is not None
        matches = asyncio.run(_search(self._endpoint, self._token, query, _SEARCH_LIMIT))
        ranked = [
            (str(m.get("id", "")), str(m.get("snippet") or m.get("title") or ""))
            for m in matches
            if m.get("id")
        ]
        return pack_to_budget(ranked, budget_tokens)
