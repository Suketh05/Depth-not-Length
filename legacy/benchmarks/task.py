"""The Task contract: every benchmark loader (dcbench, swebench, longmemeval)
emits Task objects in this shape and nothing else. run.py only ever sees Tasks,
which is what lets it be a flat loop over tasks x arms x models."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryItem:
    """One unit of memory that a MemorySystem.write() ingests.

    item_id must be stable across a run: it is the join key between what was
    written, what retrieve() returns in RetrievedContext.ids, and the gold IDs
    in Task.governing_decisions. Scoring can't compute precision/recall without it.
    """

    item_id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Task:
    """One evaluation unit, identical in shape across all three datasets.

    repo_ref is None for LongMemEval (no repo, conversational transcript only).
    memory_corpus is everything write() ingests for this task, including any
    stripped spec content for dcbench/swebench. governing_decisions holds the
    gold item_ids the output must honor, used to score retrieval precision/recall
    and decision compliance.
    """

    task_id: str
    dataset: str
    query: str
    repo_ref: Optional[str]
    memory_corpus: list[MemoryItem]
    governing_decisions: list[str]
    depth: int
    spec_variant: str  # "full" | "stripped"
    scorer: str
