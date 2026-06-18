"""The two core contracts every dataset and memory arm conforms to.

This module defines the small, immutable value objects that flow through the
entire pipeline:

``MemoryItem``
    One unit of memory a memory system ingests. Carries a stable ``item_id`` that
    is the join key between what was written, what retrieval returns, and the gold
    governing decisions — without it, retrieval precision/recall is undefined.

``RetrievedContext``
    What ``retrieve`` hands back: the assembled context string *and* the ordered
    ids of the items that composed it. The ids are never discarded; retrieval
    quality is scored against them.

``Task``
    One evaluation unit, identical in shape across every dataset. ``run.py``-style
    loops only ever see ``Task`` objects, which is what lets the harness be a flat
    loop over tasks x arms x models.

Keeping these as frozen dataclasses (rather than mutable records or dicts) means a
corpus written for one task can never be mutated by an arm and leak into another,
and equality/﻿hashing are structural — useful for deduplicating run cells and
caching.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

__all__ = [
    "MemoryItem",
    "RetrievedContext",
    "Scorer",
    "SpecVariant",
    "Task",
]


class SpecVariant(str, Enum):
    """Whether a task's governing constraint is stated in the query or stripped out.

    ``FULL`` keeps the constraint in the query (the vanilla control where no memory
    system should be needed). ``STRIPPED`` removes it, making recovery entirely the
    memory system's job. The string-valued enum serialises transparently into the
    result rows.
    """

    FULL = "full"
    STRIPPED = "stripped"


class Scorer(str, Enum):
    """The scoring contract a task is graded under.

    ``COMPLIANCE`` grades whether the response honours the governing decisions
    (used by the coding datasets). ``QA_MATCH`` grades a natural-language answer
    against a gold answer (used by LongMemEval), where code-identifier compliance
    has no meaningful signal.
    """

    COMPLIANCE = "compliance"
    QA_MATCH = "qa_match"


_EMPTY_METADATA: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class MemoryItem:
    """A single ingestible unit of memory.

    Parameters
    ----------
    item_id
        Stable identifier, unique within a task's corpus. Join key for scoring.
    text
        The natural-language or code content of this memory.
    metadata
        Free-form attributes. Convention-bearing keys used elsewhere in the
        benchmark: ``node_type`` (``"constraint"`` | ``"justification"``) and
        ``constrains`` (the ``item_id`` a justification node points back to),
        which let the structured-graph arm follow an explicit link rather than
        guess by resemblance.
    """

    item_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=lambda: _EMPTY_METADATA)

    def __post_init__(self) -> None:
        if not self.item_id:
            raise ValueError("MemoryItem.item_id must be a non-empty string")
        # Freeze metadata behind a read-only view so a frozen item is frozen all
        # the way down (a mutable dict on a 'frozen' dataclass is a footgun).
        if not isinstance(self.metadata, MappingProxyType):
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def node_type(self) -> str | None:
        """The graph node role, if this item was emitted as a typed graph node."""
        value = self.metadata.get("node_type")
        return str(value) if value is not None else None

    @property
    def constrains(self) -> str | None:
        """The ``item_id`` this justification node links back to, if any."""
        value = self.metadata.get("constrains")
        return str(value) if value is not None else None

    def approx_tokens(self, chars_per_token: int = 4) -> int:
        """Rough token count under a fixed chars-per-token heuristic.

        The benchmark scores *real* per-call tokens from the model API; this
        cheap estimate is only used by arms to pack a fixed retrieval budget, and
        it must be identical across arms so the budget is a fair constraint.
        """
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        return len(self.text) // chars_per_token


@dataclass(frozen=True, slots=True)
class RetrievedContext:
    """The output of ``MemorySystem.retrieve``.

    Parameters
    ----------
    text
        The assembled context string injected into the prompt.
    ids
        The ordered ``item_id`` values that composed ``text``. Must be preserved:
        retrieval precision/recall compares these against the gold ids.
    """

    text: str
    ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Normalise any iterable of ids to a tuple for immutability (idempotent
        # on a tuple, so no isinstance branch is needed).
        object.__setattr__(self, "ids", tuple(self.ids))

    @classmethod
    def empty(cls) -> RetrievedContext:
        """Return a retrieval that produced nothing (the ``none`` arm's output)."""
        return cls(text="", ids=())


@dataclass(frozen=True, slots=True)
class Task:
    """One evaluation unit, uniform across all datasets.

    Parameters
    ----------
    task_id
        Globally unique task identifier (dataset-prefixed by convention).
    dataset
        Source dataset name (e.g. ``"dcbench"``).
    query
        The prompt given to the agent.
    repo_ref
        Pinned repository reference for coding tasks, or ``None`` for
        conversational tasks (LongMemEval) where there is no repo to honour a
        decision in.
    memory_corpus
        Everything a memory system ingests for this task, including any stripped
        constraint content. Always covers the full seeded set, not only the
        governing items, so an arm cannot infer relevance from corpus size.
    governing_decisions
        The gold ``item_id`` values the output must honour. Used for both
        decision compliance and retrieval precision/recall.
    depth
        Causal-chain depth ``d`` (>= 1). For the coding datasets this equals the
        strip-count; for LongMemEval it is read off the dataset's own multi-hop
        structure.
    spec_variant
        Whether the governing constraint is in the query (``FULL``) or stripped
        (``STRIPPED``).
    scorer
        Which scoring contract grades this task.
    """

    task_id: str
    dataset: str
    query: str
    repo_ref: str | None
    memory_corpus: tuple[MemoryItem, ...]
    governing_decisions: tuple[str, ...]
    depth: int
    spec_variant: SpecVariant
    scorer: Scorer

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("Task.task_id must be non-empty")
        if self.depth < 1:
            raise ValueError(f"Task.depth must be >= 1, got {self.depth}")
        # Normalise collections to tuples (immutability) and coerce string enum
        # inputs that loaders may pass. All calls are idempotent on already-correct
        # values, so no isinstance guards are needed.
        object.__setattr__(self, "memory_corpus", tuple(self.memory_corpus))
        object.__setattr__(self, "governing_decisions", tuple(self.governing_decisions))
        object.__setattr__(self, "spec_variant", SpecVariant(self.spec_variant))
        object.__setattr__(self, "scorer", Scorer(self.scorer))

        ids = [item.item_id for item in self.memory_corpus]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Task {self.task_id}: memory_corpus item_ids are not unique")

    @property
    def is_coding(self) -> bool:
        """True when there is a repository to merge a change into."""
        return self.repo_ref is not None

    @property
    def corpus_by_id(self) -> Mapping[str, MemoryItem]:
        """Index from ``item_id`` to the memory item (ground truth for scoring)."""
        return MappingProxyType({item.item_id: item for item in self.memory_corpus})
