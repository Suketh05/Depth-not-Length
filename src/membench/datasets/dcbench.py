"""dcbench loader: decisions invisible in code, with the spec-stripping harness.

Source data is a vendored snapshot of github.com/brief-hq/dcbench (15 decisions and
per-task gotchas). Depth equals strip-count, one dial:

* ``d=1`` (full): the governing constraints are stated in the query. No memory is
  needed -- the vanilla control.
* ``d=2`` (stripped): constraints removed from the query; each decision is one
  combined corpus node (constraint + rationale). One retrieval hop recovers it.
* ``d=3`` (stripped): same query stripping, but each decision splits into a
  constraint node and a justification node, the justification carrying a
  ``constrains`` link back to the constraint. Recovering the whole picture takes a
  second hop -- which only a link-following arm can make.

The ``constrains`` metadata is exactly what the structured-graph arm turns into a
``CONSTRAINS`` edge, so depth-3 dcbench is where the structured arm's advantage is
designed to appear.
"""

from __future__ import annotations

import json
from typing import Any

from membench.datasets.base import DATA_DIR, max_depth_by_task_id
from membench.types import MemoryItem, Scorer, SpecVariant, Task

__all__ = ["load_dcbench_tasks"]

_DCBENCH_DIR = DATA_DIR / "dcbench"
_SOURCE_COMMIT = "9f507ce0a6ef61124aca90cc4e17fb80592a5e4e"
_REPO_REF = f"github.com/brief-hq/dcbench@{_SOURCE_COMMIT}"


def _load(name: str) -> dict[str, Any]:
    with open(_DCBENCH_DIR / name) as handle:
        data: dict[str, Any] = json.load(handle)
        return data


def _combined_corpus(decisions_by_id: dict[str, dict[str, Any]]) -> tuple[MemoryItem, ...]:
    """One node per decision (constraint + rationale together) -- the d<=2 shape."""
    return tuple(
        MemoryItem(
            item_id=decision_id,
            text=f"{d['decision']}\n\nRationale: {d['rationale']}",
            metadata={"topic": d["topic"], "category": d["category"], "node_type": "decision"},
        )
        for decision_id, d in decisions_by_id.items()
    )


def _split_corpus(decisions_by_id: dict[str, dict[str, Any]]) -> tuple[MemoryItem, ...]:
    """Two linked nodes per decision (constraint + justification) -- the d=3 shape."""
    items: list[MemoryItem] = []
    for decision_id, d in decisions_by_id.items():
        items.append(
            MemoryItem(
                item_id=decision_id,
                text=d["decision"],
                metadata={"topic": d["topic"], "node_type": "constraint"},
            )
        )
        items.append(
            MemoryItem(
                item_id=f"{decision_id}-rationale",
                text=d["rationale"],
                metadata={
                    "topic": d["topic"],
                    "node_type": "justification",
                    "constrains": decision_id,
                },
            )
        )
    return tuple(items)


def load_dcbench_tasks(
    spec_variant: SpecVariant | str = SpecVariant.FULL,
    depth: int = 1,
    task_ids: list[str] | None = None,
) -> list[Task]:
    """Return dcbench tasks for the given spec variant and depth.

    ``full`` is the unstripped d=1 control; ``stripped`` supports depth 2 or 3.
    Tasks not certified to the requested depth in depth_labels.jsonl are dropped.
    """
    variant = SpecVariant(spec_variant)
    if variant is SpecVariant.FULL and depth != 1:
        raise ValueError("spec_variant=full is the d=1 control; depth must be 1")
    if variant is SpecVariant.STRIPPED and depth not in (2, 3):
        raise ValueError("spec_variant=stripped supports depth 2 or 3 only")

    decisions_by_id = {d["id"]: d for d in _load("decisions.json")["decisions"]}
    raw_tasks = _load("tasks.json")["tasks"]
    if task_ids is not None:
        wanted = set(task_ids)
        raw_tasks = [t for t in raw_tasks if t["task_id"] in wanted]

    certified = max_depth_by_task_id("dcbench")
    raw_tasks = [t for t in raw_tasks if certified.get(t["task_id"], 0) >= depth]

    corpus = _split_corpus(decisions_by_id) if depth == 3 else _combined_corpus(decisions_by_id)

    tasks: list[Task] = []
    for raw in raw_tasks:
        governing = tuple(dict.fromkeys(g["decision_id"] for g in raw["gotchas"]))
        if variant is SpecVariant.FULL:
            spec_block = "\n\n".join(f"- {decisions_by_id[d]['decision']}" for d in governing)
            query = f"{raw['prompt']}\n\nRelevant team decisions you must honour:\n{spec_block}"
        else:
            query = raw["prompt"]
        tasks.append(
            Task(
                task_id=f"dcbench-{raw['task_id']}",
                dataset="dcbench",
                query=query,
                repo_ref=_REPO_REF,
                memory_corpus=corpus,
                governing_decisions=governing,
                depth=depth,
                spec_variant=variant,
                scorer=Scorer.COMPLIANCE,
            )
        )
    return tasks
