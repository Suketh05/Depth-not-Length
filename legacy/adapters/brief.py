"""brief arm: structured-graph memory, walking Brief's typed decision graph
via the real Brief product (not a local stand-in, unlike mem0.py).

write() is a documented no-op. Brief's only ingestion tool,
brief_record_decision, renders an interactive approval card and explicitly
instructs the caller to stop and wait for a human to approve -- it cannot be
called unattended from inside a benchmark loop that writes per task. The real
dcbench repo has the same shape: it seeds Brief once via a separate
`benchmark/seed.ts --api-key ...` step before any task runs, not per task.
So here too, the workspace is assumed pre-seeded with task.memory_corpus
out-of-band before a benchmark run starts; write() has nothing to do.

retrieve() calls the real brief_search MCP tool over Streamable HTTP,
authenticating with an OAuth token read from BRIEF_OAUTH_TOKEN -- never
hardcoded, per the fairness/reproducibility rule that this is the one file a
teammate edits to point the arm at a real workspace. brief_search takes a
result-count `limit`, not a token budget, so this walks the graph by asking
for the max limit, then truncates the ranked results to budget_tokens with
the same truncate-at-first-overflow rule the other arms use, and returns the
IDs of exactly the nodes that made the cut -- the nodes actually walked.

Verification gap, stated plainly: the `mcp` package requires Python >=3.10.
This environment runs Python 3.9, so the streamable-HTTP client import below
could not be pip-installed or exercised against a live server, and the
result-parsing in _extract_matches is best-effort against the documented
CallToolResult shape, not confirmed against a real brief_search response.
Confirm both against a live call with a real BRIEF_OAUTH_TOKEN before
trusting this arm's numbers.
"""

import asyncio
import json
import os

from adapters.base import MemorySystem, RetrievedContext
from benchmarks.task import MemoryItem

_ENDPOINT = "https://app.briefhq.ai/mcp"
_CHARS_PER_TOKEN = 4  # same heuristic as adapters/fullcontext.py and mem0.py
_SEARCH_LIMIT = 50  # brief_search's documented max


def _approx_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


def _extract_matches(result) -> list[dict]:
    """Best-effort parse of a brief_search CallToolResult. Tries the
    structured-output field first, then falls back to parsing JSON out of
    the first text content block. Unverified against a live response."""
    structured = getattr(result, "structuredContent", None)
    if structured:
        return structured.get("results", structured) if isinstance(structured, dict) else structured

    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        return parsed.get("results", parsed) if isinstance(parsed, dict) else parsed
    return []


async def _search_via_mcp(token: str, query: str, limit: int) -> list[dict]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(
        _ENDPOINT, headers={"Authorization": f"Bearer {token}"}
    ) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "brief_search", {"query": query, "types": "decision", "limit": limit}
            )
            return _extract_matches(result)


class BriefMemorySystem(MemorySystem):
    def __init__(self):
        self._token = os.environ.get("BRIEF_OAUTH_TOKEN")
        if not self._token:
            raise RuntimeError(
                "BRIEF_OAUTH_TOKEN is not set. The brief arm reads its "
                "credential from the environment and never hardcodes it."
            )

    def write(self, items: list[MemoryItem]) -> None:
        # Intentional no-op -- see module docstring. Brief's only ingestion
        # path is interactive and approval-gated; the workspace is assumed
        # pre-seeded before the benchmark runs.
        pass

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        matches = asyncio.run(_search_via_mcp(self._token, query, _SEARCH_LIMIT))

        included_text: list[str] = []
        included_ids: list[str] = []
        used_tokens = 0
        for match in matches:
            text = match.get("snippet") or match.get("title") or ""
            item_tokens = _approx_tokens(text)
            if used_tokens + item_tokens > budget_tokens:
                break
            included_text.append(text)
            included_ids.append(match.get("id"))
            used_tokens += item_tokens

        return RetrievedContext(text="\n\n".join(included_text), ids=included_ids)
