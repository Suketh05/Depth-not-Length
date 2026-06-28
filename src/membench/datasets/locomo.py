r"""LoCoMo loader: very long-term, multi-session conversational memory.

LoCoMo (Maharana et al., 2024) is a benchmark of very long human-like dialogues:
each *sample* is a conversation between two speakers spread over many dated
*sessions* (tens of sessions, hundreds of turns), paired with QA whose answer is
grounded in specific dialogue turns (the ``evidence``) that may sit far back in the
conversation. Each turn carries a stable ``dia_id`` of the form ``"D{session}:{turn}"``
(e.g. ``"D3:2"`` is the 2nd turn of session 3).

This loader maps the schema onto the uniform :class:`~membench.types.Task`:

* every dialogue turn becomes a :class:`~membench.types.MemoryItem` keyed by its
  ``dia_id`` (the join key for retrieval scoring), text ``"{speaker}: {text}"``;
* every answerable QA becomes one task whose ``memory_corpus`` is the whole
  conversation, whose ``governing_decisions`` are the gold ``evidence`` turn ids,
  and whose ``scorer`` is :attr:`~membench.types.Scorer.QA_MATCH` (these are
  natural-language transcripts, so code-identifier compliance has no signal);
* there is no repository, so ``repo_ref`` is ``None`` and the spec is never
  stripped (``spec_variant = FULL``) -- as with LongMemEval, depth is read off the
  conversation's own structure rather than dialled by stripping.

Depth -- the "session distance" of a question
---------------------------------------------
This is the *depth-not-length* construct applied to a transcript. The supporting
turns of a question span sessions ``s_lo .. s_hi``; the answer therefore requires
bridging a gap of ``s_hi - s_lo`` sessions. Depth is defined as that span plus one,

.. math::

    d = \bigl(\max_e \operatorname{sess}(e)\bigr)
        - \bigl(\min_e \operatorname{sess}(e)\bigr) + 1,

over the evidence turns ``e`` of the question, where :math:`\operatorname{sess}`
reads the session index off the ``dia_id``. A single-session (single-hop) question
has ``d = 1``; a question whose evidence straddles session 1 and session 3 has
``d = 3``. Depth thus grows with the *session gap*, instantiating the
similarity-decay model ``s_i = s_0 rho^i`` over session distance that the benchmark
theory assumes -- the farther apart the linked turns, the deeper the recall.

Adversarial questions (LoCoMo category 5) carry no evidence turns -- they are
unanswerable by construction -- so they are skipped: with no gold turns there is
nothing to score retrieval against.

References
----------
Maharana, A., Lee, D.-H., Tulyakov, S., Bansal, M., Barbieri, F., & Fang, Y.
(2024). *Evaluating Very Long-Term Conversational Memory of LLM Agents.* ACL 2024.
arXiv:2402.17753. Dataset: https://github.com/snap-research/locomo
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection
from pathlib import Path
from typing import Any

from membench.types import MemoryItem, Scorer, SpecVariant, Task

__all__ = ["ADVERSARIAL_CATEGORY", "gold_answers", "load_locomo_tasks"]

# LoCoMo question categories: 1 single-hop, 2 multi-hop, 3 temporal,
# 4 open-domain (commonsense), 5 adversarial (unanswerable -- no evidence turns).
ADVERSARIAL_CATEGORY = 5

_SESSION_KEY = re.compile(r"session_(\d+)")
_DIA_ID = re.compile(r"D(\d+):\d+")


def _session_of_dia_id(dia_id: str) -> int:
    """Return the 1-based session index encoded in a ``dia_id`` (``"D3:2"`` -> 3)."""
    match = _DIA_ID.fullmatch(dia_id)
    if match is None:
        raise ValueError(f"malformed LoCoMo dia_id: {dia_id!r} (expected 'D<session>:<turn>')")
    return int(match.group(1))


def _corpus_from_conversation(conversation: dict[str, Any]) -> tuple[MemoryItem, ...]:
    """Flatten every dialogue turn of a conversation into ordered ``MemoryItem``s.

    The ``conversation`` dict mixes ``session_<n>`` turn lists with sibling metadata
    keys (``speaker_a``/``speaker_b``, ``session_<n>_date_time``); only the
    ``session_<n>`` lists are turned into corpus items, ordered by session index.
    """
    items: list[tuple[int, MemoryItem]] = []
    for key, value in conversation.items():
        match = _SESSION_KEY.fullmatch(key)
        if match is None:
            continue
        session_idx = int(match.group(1))
        date = str(conversation.get(f"{key}_date_time", ""))
        for turn in value:
            items.append(
                (
                    session_idx,
                    MemoryItem(
                        item_id=str(turn["dia_id"]),
                        text=f"{turn['speaker']}: {turn['text']}",
                        metadata={
                            "session": session_idx,
                            "date": date,
                            "speaker": str(turn["speaker"]),
                            "node_type": "turn",
                        },
                    ),
                )
            )
    items.sort(key=lambda pair: pair[0])
    return tuple(item for _, item in items)


def load_locomo_tasks(
    path: str | Path,
    *,
    categories: Collection[int] | None = None,
) -> list[Task]:
    """Return LoCoMo QA tasks from a vendored LoCoMo JSON file.

    Parameters
    ----------
    path
        Path to a LoCoMo JSON file: a list of samples, each with ``sample_id``, a
        ``conversation`` (sessions + speaker/date metadata), and a ``qa`` list.
    categories
        Optional LoCoMo category filter (1 single-hop, 2 multi-hop, 3 temporal,
        4 open-domain, 5 adversarial). Defaults to all *answerable* questions.
        Adversarial questions carry no evidence and are always dropped regardless.

    Returns
    -------
    list[Task]
        One task per answerable QA. ``governing_decisions`` are the gold evidence
        turn ids; ``depth`` is the evidence session span (see module docstring);
        ``scorer`` is ``QA_MATCH`` and ``repo_ref`` is ``None``.

    Raises
    ------
    ValueError
        If a question's evidence references a ``dia_id`` absent from its
        conversation (a corrupt vendored sample).
    """
    wanted = frozenset(categories) if categories is not None else None
    samples: list[dict[str, Any]] = json.loads(Path(path).read_text())

    tasks: list[Task] = []
    for sample in samples:
        conversation: dict[str, Any] = sample["conversation"]
        corpus = _corpus_from_conversation(conversation)
        corpus_ids = {item.item_id for item in corpus}
        sample_id = str(sample["sample_id"])

        for q_index, qa in enumerate(sample["qa"]):
            category = int(qa.get("category", 0))
            evidence = [str(e) for e in qa.get("evidence", [])]
            if category == ADVERSARIAL_CATEGORY or not evidence:
                continue  # unanswerable: no gold turns to ground or score against
            if wanted is not None and category not in wanted:
                continue

            missing = [e for e in evidence if e not in corpus_ids]
            if missing:
                raise ValueError(
                    f"LoCoMo sample {sample_id!r} question {q_index}: "
                    f"evidence not in conversation: {missing}"
                )

            sessions = [_session_of_dia_id(e) for e in evidence]
            depth = max(sessions) - min(sessions) + 1
            governing = tuple(dict.fromkeys(evidence))  # de-dup, preserve order
            tasks.append(
                Task(
                    task_id=f"locomo-{sample_id}-q{q_index}",
                    dataset="locomo",
                    query=str(qa["question"]),
                    repo_ref=None,
                    memory_corpus=corpus,
                    governing_decisions=governing,
                    depth=depth,
                    spec_variant=SpecVariant.FULL,  # not stripped; see module docstring
                    scorer=Scorer.QA_MATCH,
                )
            )
    return tasks


def gold_answers(path: str | Path) -> dict[str, str]:
    """Return ``{task_id: gold_answer}`` for the QA-match scorer.

    Mirrors :func:`load_locomo_tasks`' answerable-question selection so the keys
    line up exactly with the loaded tasks.
    """
    samples: list[dict[str, Any]] = json.loads(Path(path).read_text())
    answers: dict[str, str] = {}
    for sample in samples:
        sample_id = str(sample["sample_id"])
        for q_index, qa in enumerate(sample["qa"]):
            category = int(qa.get("category", 0))
            if category == ADVERSARIAL_CATEGORY or not qa.get("evidence"):
                continue
            answers[f"locomo-{sample_id}-q{q_index}"] = str(qa["answer"])
    return answers
