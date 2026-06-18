"""SWE-bench loader: a cross-file constraint in a remote module.

Source data is a vendored, mined snapshot of SWE-bench_Lite (django/sphinx/pytest).
Each instance's gold patch depends on a function/class defined in a *different* file
than the one being edited; that remote definition is the governing "decision". Depth
equals strip-count, mirroring dcbench:

* ``d=1`` (full): the remote definition is folded into the query (control).
* ``d=2`` (stripped): removed from the query; one combined node per remote definition.
* ``d=3`` (stripped): the node splits into a constraint sub-node (code) and a
  justification sub-node (its docstring), linked by ``constrains``. Only instances
  whose remote definition has a docstring are certified for d=3.

memory_corpus is pooled per repo (a django task's haystack is all mined django
definitions, governing + distractor), never mixed across the unrelated codebases.
"""

from __future__ import annotations

import json
from typing import Any

from membench.datasets.base import DATA_DIR, max_depth_by_task_id
from membench.types import MemoryItem, Scorer, SpecVariant, Task

__all__ = ["load_swebench_tasks"]

_PATH = DATA_DIR / "swebench" / "instances.json"


def _load_instances() -> list[dict[str, Any]]:
    data: dict[str, Any] = json.loads(_PATH.read_text())
    instances: list[dict[str, Any]] = data["instances"]
    return instances


def _combined_corpus(repo_instances: list[dict[str, Any]]) -> tuple[MemoryItem, ...]:
    items, seen = [], set()
    for inst in repo_instances:
        qualname = inst["remote_qualname"]
        if qualname in seen:
            continue
        seen.add(qualname)
        items.append(
            MemoryItem(
                item_id=qualname,
                text=inst["snippet_full"],
                metadata={"repo": inst["repo"], "node_type": "definition"},
            )
        )
    return tuple(items)


def _split_corpus(repo_instances: list[dict[str, Any]]) -> tuple[MemoryItem, ...]:
    items, seen = [], set()
    for inst in repo_instances:
        qualname = inst["remote_qualname"]
        if qualname in seen:
            continue
        seen.add(qualname)
        items.append(
            MemoryItem(
                item_id=qualname,
                text=inst["snippet_constraint"],
                metadata={"repo": inst["repo"], "node_type": "constraint"},
            )
        )
        if inst["snippet_justification"]:
            items.append(
                MemoryItem(
                    item_id=f"{qualname}-rationale",
                    text=inst["snippet_justification"],
                    metadata={
                        "repo": inst["repo"],
                        "node_type": "justification",
                        "constrains": qualname,
                    },
                )
            )
    return tuple(items)


def load_swebench_tasks(
    spec_variant: SpecVariant | str = SpecVariant.FULL,
    depth: int = 1,
    task_ids: list[str] | None = None,
) -> list[Task]:
    """Return SWE-bench tasks for the given spec variant and depth.

    ``full`` is the unstripped d=1 control; ``stripped`` supports depth 2 or 3 (d=3
    only for instances whose remote definition has a docstring, per depth labels).
    """
    variant = SpecVariant(spec_variant)
    if variant is SpecVariant.FULL and depth != 1:
        raise ValueError("spec_variant=full is the d=1 control; depth must be 1")
    if variant is SpecVariant.STRIPPED and depth not in (2, 3):
        raise ValueError("spec_variant=stripped supports depth 2 or 3 only")

    certified = max_depth_by_task_id("swebench")
    instances = [i for i in _load_instances() if certified.get(i["instance_id"], 0) >= depth]

    by_repo: dict[str, list[dict[str, Any]]] = {}
    for inst in instances:
        by_repo.setdefault(inst["repo"], []).append(inst)
    corpus_by_repo = {
        repo: (_split_corpus(insts) if depth == 3 else _combined_corpus(insts))
        for repo, insts in by_repo.items()
    }

    selected = instances
    if task_ids is not None:
        wanted = set(task_ids)
        selected = [i for i in selected if i["instance_id"] in wanted]

    tasks: list[Task] = []
    for inst in selected:
        if variant is SpecVariant.FULL:
            query = (
                f"{inst['problem_statement']}\n\n"
                f"Relevant code in this repo you must honour:\n{inst['snippet_full']}"
            )
        else:
            query = inst["problem_statement"]
        tasks.append(
            Task(
                task_id=inst["instance_id"],
                dataset="swebench",
                query=query,
                repo_ref=f"github.com/{inst['repo']}@{inst['base_commit']}",
                memory_corpus=corpus_by_repo[inst["repo"]],
                governing_decisions=(inst["remote_qualname"],),
                depth=depth,
                spec_variant=variant,
                scorer=Scorer.COMPLIANCE,
            )
        )
    return tasks
