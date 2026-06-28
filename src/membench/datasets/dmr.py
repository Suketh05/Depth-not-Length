"""DMR (Deep Memory Retrieval) loader: recall a fact buried many sessions earlier.

Deep Memory Retrieval (DMR) is the conversational-memory benchmark introduced with
MemGPT [1]_ and built on top of the Multi-Session Chat (MSC) corpus [2]_. Each
example is a chronologically ordered multi-session dialogue followed by a question
whose answer was established in *one specific, distant* session. The agent succeeds
only if its memory surfaces that gold session rather than the more recent (and
usually more salient) chatter -- so DMR isolates *deep recall*: the further back the
fact lives, the harder it is to retrieve under a fixed context budget.

Like LongMemEval (and unlike the spec-stripped coding datasets) DMR is **not**
spec-stripped. ``repo_ref`` is ``None`` (these are conversational questions, not code
edits) and ``scorer = QA_MATCH`` (a natural-language answer is graded against gold;
code-identifier compliance has no signal on transcripts). The gold answer rides
alongside via :func:`gold_answers` for that scorer.

Depth = recall distance
-----------------------
Sessions are ordered oldest-to-newest with 0-based index ``g`` for the gold session
and ``S`` total sessions; the question is posed *after* the most recent session. The
recall distance (mapped directly to :attr:`~membench.types.Task.depth`) is

.. math::

    d = S - g

i.e. the number of sessions spanned from the gold session through the most recent
one, inclusive. A fact in the latest session is ``d = 1`` (shallow); a fact in the
oldest session is ``d = S`` (deepest). This makes the gold session the depth-defining
target, exactly the quantity the similarity-decay model ``s_i = s_0 rho^i`` predicts
gets harder to recover as ``d`` grows.

References
----------
.. [1] Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., &
   Gonzalez, J. E. (2023). MemGPT: Towards LLMs as Operating Systems.
   arXiv:2310.08560. (Introduces the Deep Memory Retrieval task.)
.. [2] Xu, J., Szlam, A., & Weston, J. (2022). Beyond Goldfish Memory: Long-Term
   Open-Domain Conversation. ACL 2022. (The Multi-Session Chat corpus DMR is
   derived from.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from membench.types import MemoryItem, Scorer, SpecVariant, Task

__all__ = ["gold_answers", "load_dmr_tasks", "recall_distance"]


def _session_text(session: dict[str, Any]) -> str:
    """Flatten a session's turns into a single ``role: content`` transcript string."""
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in session["turns"])


def recall_distance(gold_index: int, n_sessions: int) -> int:
    """Return the DMR recall distance ``d = S - g`` for a gold session.

    Parameters
    ----------
    gold_index
        0-based index of the gold session in chronological (oldest-first) order.
    n_sessions
        Total number of sessions ``S`` in the example.

    Returns
    -------
    int
        Recall distance ``d = n_sessions - gold_index`` (``>= 1``); ``1`` when the
        gold fact is in the most recent session, ``n_sessions`` when it is in the
        oldest.

    Raises
    ------
    ValueError
        If ``gold_index`` is outside ``[0, n_sessions)``.
    """
    if not 0 <= gold_index < n_sessions:
        raise ValueError(f"gold_index {gold_index} out of range for {n_sessions} sessions")
    return n_sessions - gold_index


def load_dmr_tasks(path: str | Path) -> list[Task]:
    """Load Deep Memory Retrieval tasks from a JSON snapshot.

    The snapshot is a tiny, self-contained file (no external download); its schema
    is ``{"examples": [{"question_id", "question", "answer", "gold_session_id",
    "sessions": [{"session_id", "turns": [{"role", "content"}], "date"?}]}]}``.
    Sessions are taken in the order given (oldest first). Each session becomes one
    :class:`~membench.types.MemoryItem` (the join key is its ``session_id``); the
    gold session id is the sole governing decision, and the task depth is the recall
    distance :func:`recall_distance` of that gold session.

    Parameters
    ----------
    path
        Filesystem path to the DMR JSON snapshot.

    Returns
    -------
    list of Task
        One conversational task per example, with ``scorer = QA_MATCH``,
        ``repo_ref = None`` and ``spec_variant = FULL``.

    Raises
    ------
    ValueError
        If an example's ``gold_session_id`` is not among its sessions.
    """
    snapshot: dict[str, Any] = json.loads(Path(path).read_text())
    tasks: list[Task] = []
    for example in snapshot["examples"]:
        sessions: list[dict[str, Any]] = example["sessions"]
        gold_id = example["gold_session_id"]
        gold_index = next(
            (i for i, s in enumerate(sessions) if s["session_id"] == gold_id),
            None,
        )
        if gold_index is None:
            raise ValueError(
                f"gold_session_id {gold_id!r} not found in example "
                f"{example['question_id']!r} sessions"
            )
        depth = recall_distance(gold_index, len(sessions))
        corpus = tuple(
            MemoryItem(
                item_id=session["session_id"],
                text=_session_text(session),
                metadata={
                    "session_index": index,
                    "is_gold": session["session_id"] == gold_id,
                    **({"date": session["date"]} if "date" in session else {}),
                },
            )
            for index, session in enumerate(sessions)
        )
        tasks.append(
            Task(
                task_id=f"dmr-{example['question_id']}",
                dataset="dmr",
                query=example["question"],
                repo_ref=None,
                memory_corpus=corpus,
                governing_decisions=(gold_id,),
                depth=depth,
                spec_variant=SpecVariant.FULL,  # conversational; not spec-stripped
                scorer=Scorer.QA_MATCH,
            )
        )
    return tasks


def gold_answers(path: str | Path) -> dict[str, str]:
    """Return ``{task_id: gold_answer}`` for the QA-match scorer.

    Parameters
    ----------
    path
        Filesystem path to the DMR JSON snapshot.

    Returns
    -------
    dict of str to str
        Maps each ``dmr-<question_id>`` task id to its gold answer string.
    """
    snapshot: dict[str, Any] = json.loads(Path(path).read_text())
    return {
        f"dmr-{example['question_id']}": str(example["answer"]) for example in snapshot["examples"]
    }
