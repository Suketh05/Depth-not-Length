"""LongMemEval loader -- temporal-reasoning + knowledge-update splits ONLY.

Source data is a vendored, capped snapshot of huggingface.co/datasets/
xiaowu0162/longmemeval-cleaned (longmemeval_s_cleaned split, pinned HF
revision in benchmarks/data/longmemeval/snapshot.json's _provenance block),
not a live dependency -- same vendoring pattern as dcbench/swebench.

Per CLAUDE.md, LongMemEval is NOT spec-stripped: depth is read directly off
the dataset's own existing multi-hop structure (how many distinct
answer_session_ids ground the gold answer) rather than dialed by us the way
dcbench/swebench's strip-count is. depth = min(len(answer_session_ids), 3),
bucketed to match the d=1/d=2/d=3+ convention used elsewhere, computed once
when the snapshot was vendored (see snapshot.json's per-question "depth"
field) -- this loader does not recompute or alter it.

repo_ref is None: these are conversational-memory questions, not code edits,
so there is no repo to honor a decision in.

Known gap, stated plainly: scorer="qa_match" because scoring/compliance.py's
code-identifier extraction (PascalCase/camelCase/function-call patterns) has
no meaningful signal against natural-language conversation transcripts -- it
would silently produce near-zero, meaningless compliance numbers if run as-is.
A real QA-answer-match scorer (comparing the model's response against the
snapshot's "answer" field, keyed by task_id) does not exist yet, and
agent/runner.py currently calls score_compliance() unconditionally regardless
of task.scorer. Both are out of scope for this loader; task.scorer is the
contract's intended dispatch point for that future scorer, just not wired up.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.task import MemoryItem, Task

_DATA_PATH = Path(__file__).parent / "data" / "longmemeval" / "snapshot.json"

_ALLOWED_TYPES = {"temporal-reasoning", "knowledge-update"}


def _session_text(session: dict) -> str:
    lines = [f"{turn['role']}: {turn['content']}" for turn in session["turns"]]
    return "\n".join(lines)


def load_longmemeval_tasks(question_types: list[str] | None = None) -> list[Task]:
    """Return LongMemEval Tasks. question_types, if given, must be a subset
    of {"temporal-reasoning", "knowledge-update"} -- the only two types
    vendored, per CLAUDE.md's scope lock. Defaults to both."""
    wanted_types = set(question_types) if question_types is not None else _ALLOWED_TYPES
    if not wanted_types <= _ALLOWED_TYPES:
        raise ValueError(
            f"question_types must be a subset of {_ALLOWED_TYPES}; "
            "LongMemEval is locked to the temporal + knowledge-update splits only"
        )

    snapshot = json.loads(_DATA_PATH.read_text())

    tasks = []
    for q in snapshot["questions"]:
        if q["question_type"] not in wanted_types:
            continue

        memory_corpus = [
            MemoryItem(
                item_id=session["session_id"],
                text=_session_text(session),
                metadata={"date": session["date"], "is_gold": session["is_gold"]},
            )
            for session in q["sessions"]
        ]

        tasks.append(
            Task(
                task_id=f"longmemeval-{q['question_id']}",
                dataset="longmemeval",
                query=q["question"],
                repo_ref=None,
                memory_corpus=memory_corpus,
                governing_decisions=q["answer_session_ids"],
                depth=q["depth"],
                spec_variant="full",  # not stripped -- see module docstring
                scorer="qa_match",
            )
        )
    return tasks
