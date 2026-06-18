"""dcbench loader -- spec-stripping harness, d=1/2/3.

Source data is a vendored snapshot of github.com/brief-hq/dcbench (pinned commit
in benchmarks/data/dcbench/*.json's _provenance block), not a live clone: we only
need the 15 decisions and their per-task gotcha weights, not the Next.js app the
real benchmark scores diffs against.

Per CLAUDE.md, depth == strip-count -- one dial, not two operations:
  d=1 (spec_variant="full")     -- the governing decisions' constraint sentences
                                    are stated directly in the query. No memory
                                    system should be needed to comply; this is
                                    the vanilla control.
  d=2 (spec_variant="stripped") -- the constraint sentences are removed from the
                                    query entirely. memory_corpus still holds one
                                    combined node per decision (constraint +
                                    rationale together) -- one retrieval hop
                                    recovers the whole thing.
  d=3 (spec_variant="stripped") -- same query-side stripping as d=2, but each
                                    decision's memory_corpus node is split into
                                    two linked nodes: a constraint node and a
                                    justification node (the justification's
                                    metadata carries "constrains": <decision_id>
                                    back to its constraint). Recovering the full
                                    picture takes a second hop.
spec_variant="stripped" always strips the same thing (the constraint sentence
out of the query); depth only changes how memory_corpus represents what was
stripped. CLAUDE.md's "3+" bucket doesn't define a depth=4 split for dcbench's
flat decision/rationale data model, so depth must be exactly 1, 2, or 3 -- there
is no silent reuse of the depth=3 shape under a higher label.

Per task, benchmarks/depth_labels.jsonl (data, not code) records the max depth
that task is certified for. Right now every dcbench task is certified to depth 3
by a single-pass automated check (see that file's annotation_method field) --
NOT CLAUDE.md's two-independent-annotator protocol, which this repo has not run.
A task whose certified max_depth is below the requested depth is dropped from
the returned set rather than stripped past what's been checked.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.task import MemoryItem, Task

_DATA_DIR = Path(__file__).parent / "data" / "dcbench"
_DEPTH_LABELS_PATH = Path(__file__).parent / "depth_labels.jsonl"
_SOURCE_COMMIT = "9f507ce0a6ef61124aca90cc4e17fb80592a5e4e"
_REPO_REF = f"github.com/brief-hq/dcbench@{_SOURCE_COMMIT}"


def _load_json(name: str) -> dict:
    with open(_DATA_DIR / name) as f:
        return json.load(f)


def _decision_text(decision: dict) -> str:
    return f"{decision['decision']}\n\nRationale: {decision['rationale']}"


def _max_depth_by_task_id() -> dict:
    labels = {}
    with open(_DEPTH_LABELS_PATH) as f:
        for line in f:
            row = json.loads(line)
            if row["dataset"] == "dcbench":
                labels[row["task_id"]] = row["max_depth"]
    return labels


def _combined_memory_corpus(decisions_by_id: dict) -> list[MemoryItem]:
    """One node per decision, constraint and rationale together -- the d<=2
    shape. Used for both d=1 and d=2: nothing is stripped for d=1's control,
    and d=2 only strips the query, not the corpus structure."""
    return [
        MemoryItem(
            item_id=decision_id,
            text=_decision_text(decision),
            metadata={"topic": decision["topic"], "category": decision["category"]},
        )
        for decision_id, decision in decisions_by_id.items()
    ]


def _split_memory_corpus(decisions_by_id: dict) -> list[MemoryItem]:
    """Two linked nodes per decision -- the d=3 shape. A real graph walk
    (adapters/brief.py) can follow the justification node's "constrains" link
    back to its constraint; a flat retriever just sees one more item in the
    corpus, with no signal that the two are related."""
    items = []
    for decision_id, decision in decisions_by_id.items():
        items.append(
            MemoryItem(
                item_id=decision_id,
                text=decision["decision"],
                metadata={
                    "topic": decision["topic"],
                    "category": decision["category"],
                    "node_type": "constraint",
                },
            )
        )
        items.append(
            MemoryItem(
                item_id=f"{decision_id}-rationale",
                text=decision["rationale"],
                metadata={
                    "topic": decision["topic"],
                    "category": decision["category"],
                    "node_type": "justification",
                    "constrains": decision_id,
                },
            )
        )
    return items


def load_dcbench_tasks(
    spec_variant: str = "full", depth: int = 1, task_ids: list[str] | None = None
) -> list[Task]:
    """Return dcbench Tasks for the given spec_variant/depth.

    spec_variant="full" is the unstripped d=1 control (depth must be 1).
    spec_variant="stripped" supports depth 2 or 3 (see module docstring).
    task_ids, if given, filters to those dcbench task IDs (e.g. ["TASK-001"]).
    Tasks not certified to the requested depth in depth_labels.jsonl are
    silently dropped from the returned list (none are, as of this data file).
    """
    if spec_variant == "full":
        if depth != 1:
            raise ValueError("spec_variant='full' is the d=1 control; depth must be 1")
    elif spec_variant == "stripped":
        if depth not in (2, 3):
            raise ValueError("spec_variant='stripped' supports depth 2 or 3 only")
    else:
        raise ValueError(f"spec_variant={spec_variant!r} must be 'full' or 'stripped'")

    decisions_by_id = {d["id"]: d for d in _load_json("decisions.json")["decisions"]}
    raw_tasks = _load_json("tasks.json")["tasks"]
    if task_ids is not None:
        wanted = set(task_ids)
        raw_tasks = [t for t in raw_tasks if t["task_id"] in wanted]

    max_depth_by_task_id = _max_depth_by_task_id()
    raw_tasks = [
        t for t in raw_tasks if max_depth_by_task_id.get(t["task_id"], 0) >= depth
    ]

    # memory_corpus always covers every seeded decision, not just the
    # governing ones for a given task -- so an arm can't infer which
    # decisions matter just by noticing the corpus is suspiciously small.
    if depth == 3:
        memory_corpus = _split_memory_corpus(decisions_by_id)
    else:
        memory_corpus = _combined_memory_corpus(decisions_by_id)

    tasks = []
    for raw in raw_tasks:
        governing_ids = list(dict.fromkeys(g["decision_id"] for g in raw["gotchas"]))

        if spec_variant == "full":
            spec_block = "\n\n".join(
                f"- {decisions_by_id[d_id]['decision']}" for d_id in governing_ids
            )
            query = (
                f"{raw['prompt']}\n\n"
                "Relevant team decisions you must honor:\n"
                f"{spec_block}"
            )
        else:
            # Stripped: the constraint sentences are gone from the query.
            # Recovering them is entirely the memory system's job.
            query = raw["prompt"]

        tasks.append(
            Task(
                task_id=f"dcbench-{raw['task_id']}",
                dataset="dcbench",
                query=query,
                repo_ref=_REPO_REF,
                memory_corpus=memory_corpus,
                governing_decisions=governing_ids,
                depth=depth,
                spec_variant=spec_variant,
                scorer="compliance",
            )
        )
    return tasks
