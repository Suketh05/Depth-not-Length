"""Importing this module registers every built-in memory arm.

Arms self-register on import; this is the single place that imports all of them so
the runner, CLI, and config layer can rely on a fully-populated registry without
each caller importing arm modules individually.
"""

from __future__ import annotations

from membench.retrieval import (  # noqa: F401  (imported for registration side effects)
    bm25,
    controls,
    dense,
    hybrid,
    hyde,
    raptor,
    rerank,
    tfidf,
)
from membench.retrieval.base import available_arms
from membench.retrieval.external import (  # noqa: F401
    cognee_adapter,
    graphrag_adapter,
    langchain_adapter,
    letta_adapter,
    mem0_adapter,
    supermemory_adapter,
    zep_adapter,
)
from membench.retrieval.graph import arm, live  # noqa: F401

__all__ = ["available_arms"]
