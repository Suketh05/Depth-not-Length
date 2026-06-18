"""HyDE: Hypothetical Document Embeddings retrieval arm.

HyDE (Gao et al., 2022) addresses the query-document vocabulary gap by first
generating a *hypothetical* answer document for the query, then retrieving by the
embedding of that hypothetical document rather than of the raw query. It is a
genuinely stronger dense variant on query/answer mismatch — and a fair test of
whether "write a better query" closes the depth gap (it does not: a hypothetical
answer still resembles the *visible* task, not a decision stripped into a node
several hops away).

The query expander is pluggable. The default is a deterministic template expander
so the arm runs offline and reproducibly; a real LLM generator (which writes an
actual hypothetical answer) implements the same ``QueryExpander`` protocol.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol, runtime_checkable

from membench.retrieval.base import ArmFamily, MemorySystem, register_arm
from membench.retrieval.dense import DenseMemory
from membench.retrieval.embeddings import EmbeddingProvider
from membench.types import MemoryItem, RetrievedContext

__all__ = ["CallableExpander", "HyDEMemory", "QueryExpander", "TemplateExpander"]


@runtime_checkable
class QueryExpander(Protocol):
    """Turns a query into a hypothetical document to embed in its place."""

    def expand(self, query: str) -> str:
        """Return a hypothetical answer document for ``query``."""
        ...


class TemplateExpander:
    """Deterministic offline expander: scaffold the query into a pseudo-answer."""

    def expand(self, query: str) -> str:
        """Wrap the query in an answer-shaped template (deterministic, no model)."""
        return (
            f"{query}\n\n"
            "A correct implementation honours the relevant team decisions and the "
            f"constraints they impose. {query}"
        )


class CallableExpander:
    """Adapt any ``str -> str`` function (e.g. an LLM call) to the expander protocol."""

    def __init__(self, fn: Callable[[str], str]) -> None:
        self._fn = fn

    def expand(self, query: str) -> str:
        """Return the wrapped function's hypothetical document for ``query``."""
        return self._fn(query)


@register_arm("hyde", family=ArmFamily.DENSE, follows_links=False, uses_embeddings=True)
class HyDEMemory(MemorySystem):
    """Embed a hypothetical document for the query, then rank densely.

    Parameters
    ----------
    expander
        Query-to-hypothetical-document generator (defaults to a deterministic
        template expander).
    provider
        Embedding backend for the underlying dense retriever.
    """

    def __init__(
        self, expander: QueryExpander | None = None, provider: EmbeddingProvider | None = None
    ) -> None:
        self._expander: QueryExpander = expander or TemplateExpander()
        self._dense = DenseMemory(provider=provider)

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Ingest the corpus into the underlying dense retriever."""
        self._dense.write(items)

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Expand the query to a hypothetical document and rank densely by it."""
        hypothetical = self._expander.expand(query)
        return self._dense.retrieve(hypothetical, budget_tokens)
