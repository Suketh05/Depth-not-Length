"""SWE-bench loader -- spec-stripping harness, d=1/2/3, mirroring dcbench.py.

Source data is a vendored, mined snapshot of huggingface.co/datasets/
princeton-nlp/SWE-bench_Lite (pinned in benchmarks/data/swebench/
instances.json's _provenance block), not a live clone or dependency.

Per CLAUDE.md/handoff.md, SWE-bench's failure mode is "depth d>=2 constraint
in a remote module": the gold patch's correctness depends on a function/class
defined in a DIFFERENT file than the one being edited. Unlike dcbench, this
repo had no existing decision records to vendor -- each instance's
"constraint" was mined by reconstructing the gold patch's post-patch file and
statically resolving an identifier referenced in the patch to its real
in-repo definition (see instances.json's _provenance.mining_method). Scope is
restricted to django/django, sphinx-doc/sphinx, pytest-dev/pytest by explicit
instruction; instances with no resolvable cross-file dependency were dropped,
not forced -- 21 kept out of the candidates reviewed (yield logged in
_provenance).

Depth == strip-count, the same dial as dcbench:
  d=1 (spec_variant="full")     -- the remote constraint's full source is
                                    folded directly into the query (vanilla
                                    control; no memory system needed).
  d=2 (spec_variant="stripped") -- the constraint is removed from the query.
                                    memory_corpus holds one combined node per
                                    instance (code + docstring together).
  d=3 (spec_variant="stripped") -- same query-side stripping; the node splits
                                    into a constraint sub-node (code with its
                                    docstring statement removed) and a
                                    justification sub-node (the docstring).
                                    Only instances whose remote definition has
                                    a docstring support this (depth_labels.jsonl's
                                    max_depth); the rest cap at depth=2.

memory_corpus is pooled per repo, not globally: a django task's corpus is all
mined django instances' constraints (governing + distractor), never mixed
with pytest/sphinx instances, since they're unrelated codebases. This mirrors
dcbench's "memory_corpus = every seeded decision, not just the governing
ones" principle, using real, repo-local definitions as the distractors
instead of fabricated ones.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.task import MemoryItem, Task

_DATA_PATH = Path(__file__).parent / "data" / "swebench" / "instances.json"
_DEPTH_LABELS_PATH = Path(__file__).parent / "depth_labels.jsonl"


def _load_instances() -> list[dict]:
    return json.loads(_DATA_PATH.read_text())["instances"]


def _max_depth_by_task_id() -> dict:
    labels = {}
    with open(_DEPTH_LABELS_PATH) as f:
        for line in f:
            row = json.loads(line)
            if row["dataset"] == "swebench":
                labels[row["task_id"]] = row["max_depth"]
    return labels


def _combined_memory_corpus(repo_instances: list[dict]) -> list[MemoryItem]:
    """One node per unique remote definition -- the d<=2 shape. Two
    instances in the same repo sometimes mine the same dependency (e.g. two
    different pytest bugs both touching _pytest.nodes.Item); de-duplicated
    by item_id rather than creating redundant entries."""
    items, seen = [], set()
    for inst in repo_instances:
        item_id = inst["remote_qualname"]
        if item_id in seen:
            continue
        seen.add(item_id)
        items.append(
            MemoryItem(
                item_id=item_id,
                text=inst["snippet_full"],
                metadata={"repo": inst["repo"], "remote_module": inst["remote_module"]},
            )
        )
    return items


def _split_memory_corpus(repo_instances: list[dict]) -> list[MemoryItem]:
    """Two linked nodes per unique remote definition -- the d=3 shape, only
    for instances whose definition has a docstring (depth_labels.jsonl
    already filtered the instance list to max_depth>=3 before this is
    called)."""
    items, seen = [], set()
    for inst in repo_instances:
        item_id = inst["remote_qualname"]
        if item_id in seen:
            continue
        seen.add(item_id)
        items.append(
            MemoryItem(
                item_id=item_id,
                text=inst["snippet_constraint"],
                metadata={
                    "repo": inst["repo"],
                    "remote_module": inst["remote_module"],
                    "node_type": "constraint",
                },
            )
        )
        if inst["snippet_justification"]:
            items.append(
                MemoryItem(
                    item_id=f"{item_id}-rationale",
                    text=inst["snippet_justification"],
                    metadata={
                        "repo": inst["repo"],
                        "node_type": "justification",
                        "constrains": item_id,
                    },
                )
            )
    return items


def load_swebench_tasks(
    spec_variant: str = "full", depth: int = 1, task_ids: list[str] | None = None
) -> list[Task]:
    """Return SWE-bench Tasks for the given spec_variant/depth.

    spec_variant="full" is the unstripped d=1 control (depth must be 1).
    spec_variant="stripped" supports depth 2 or 3 (see module docstring);
    depth=3 only applies to instances certified for it in depth_labels.jsonl
    (those whose mined remote definition has a docstring) -- others are
    dropped from the depth=3 set rather than stretched past what was mined.
    task_ids, if given, filters to specific SWE-bench instance_ids.
    """
    if spec_variant == "full":
        if depth != 1:
            raise ValueError("spec_variant='full' is the d=1 control; depth must be 1")
    elif spec_variant == "stripped":
        if depth not in (2, 3):
            raise ValueError("spec_variant='stripped' supports depth 2 or 3 only")
    else:
        raise ValueError(f"spec_variant={spec_variant!r} must be 'full' or 'stripped'")

    max_depth_by_id = _max_depth_by_task_id()
    all_instances = [
        inst for inst in _load_instances() if max_depth_by_id.get(inst["instance_id"], 0) >= depth
    ]

    by_repo: dict[str, list[dict]] = {}
    for inst in all_instances:
        by_repo.setdefault(inst["repo"], []).append(inst)
    corpus_by_repo = {
        repo: (_split_memory_corpus(insts) if depth == 3 else _combined_memory_corpus(insts))
        for repo, insts in by_repo.items()
    }

    selected = all_instances
    if task_ids is not None:
        wanted = set(task_ids)
        selected = [inst for inst in selected if inst["instance_id"] in wanted]

    tasks = []
    for inst in selected:
        if spec_variant == "full":
            query = (
                f"{inst['problem_statement']}\n\n"
                "Relevant code in this repo you must honor:\n"
                f"{inst['snippet_full']}"
            )
        else:
            # Stripped: the constraint's source is gone from the query.
            # Recovering it is entirely the memory system's job.
            query = inst["problem_statement"]

        tasks.append(
            Task(
                task_id=inst["instance_id"],
                dataset="swebench",
                query=query,
                repo_ref=f"github.com/{inst['repo']}@{inst['base_commit']}",
                memory_corpus=corpus_by_repo[inst["repo"]],
                governing_decisions=[inst["remote_qualname"]],
                depth=depth,
                spec_variant=spec_variant,
                scorer="compliance",
            )
        )
    return tasks
