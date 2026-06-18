"""LongMemEval loader: temporal-reasoning + knowledge-update splits only.

Source data is a vendored, capped snapshot of LongMemEval (the two splits that test
temporal supersession). Unlike the coding datasets, LongMemEval is NOT spec-stripped:
depth is read directly off the dataset's own multi-hop structure (how many distinct
gold sessions ground the answer), bucketed to the d=1/2/3 convention. ``repo_ref`` is
``None`` -- these are conversational-memory questions, not code edits.

The gold answer travels on the task (in metadata) so the QA-match scorer can grade
the model's response against it; ``scorer = QA_MATCH`` because code-identifier
compliance has no signal on natural-language transcripts (the runner is scorer-
agnostic, and the metrics layer dispatches on ``Task.scorer``).
"""

from __future__ import annotations

import json
from typing import Any

from membench.datasets.base import DATA_DIR
from membench.types import MemoryItem, Scorer, SpecVariant, Task

__all__ = ["ALLOWED_QUESTION_TYPES", "gold_answers", "load_longmemeval_tasks"]

_PATH = DATA_DIR / "longmemeval" / "snapshot.json"
ALLOWED_QUESTION_TYPES = frozenset({"temporal-reasoning", "knowledge-update"})


def _session_text(session: dict[str, Any]) -> str:
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in session["turns"])


def load_longmemeval_tasks(question_types: list[str] | None = None) -> list[Task]:
    """Return LongMemEval tasks for the temporal + knowledge-update splits.

    ``question_types`` must be a subset of the two allowed splits (defaults to both).
    Each question becomes a task whose corpus is its candidate sessions; the gold
    answer-session ids are the governing decisions (for retrieval scoring) and the
    gold answer string rides in ``query``-paired metadata for the QA-match scorer.
    """
    wanted = frozenset(question_types) if question_types is not None else ALLOWED_QUESTION_TYPES
    if not wanted <= ALLOWED_QUESTION_TYPES:
        raise ValueError(
            f"question_types must be a subset of {sorted(ALLOWED_QUESTION_TYPES)}; "
            "LongMemEval is locked to the temporal + knowledge-update splits only"
        )

    snapshot: dict[str, Any] = json.loads(_PATH.read_text())
    tasks: list[Task] = []
    for q in snapshot["questions"]:
        if q["question_type"] not in wanted:
            continue
        corpus = tuple(
            MemoryItem(
                item_id=session["session_id"],
                text=_session_text(session),
                metadata={"date": session["date"], "is_gold": session["is_gold"]},
            )
            for session in q["sessions"]
        )
        tasks.append(
            Task(
                task_id=f"longmemeval-{q['question_id']}",
                dataset="longmemeval",
                query=q["question"],
                repo_ref=None,
                memory_corpus=corpus,
                governing_decisions=tuple(q["answer_session_ids"]),
                depth=int(q["depth"]),
                spec_variant=SpecVariant.FULL,  # not stripped; see module docstring
                scorer=Scorer.QA_MATCH,
            )
        )
    return tasks


def gold_answers() -> dict[str, str]:
    """Return ``{task_id: gold_answer}`` for the QA-match scorer."""
    snapshot: dict[str, Any] = json.loads(_PATH.read_text())
    return {
        f"longmemeval-{q['question_id']}": str(q["answer"])  # some gold answers are numeric
        for q in snapshot["questions"]
        if q["question_type"] in ALLOWED_QUESTION_TYPES
    }
