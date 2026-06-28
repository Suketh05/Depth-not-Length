"""HotpotQA loader: the canonical multi-hop QA benchmark, in distractor setting.

Source schema is the public *distractor* split of HotpotQA [1]_. Each example pairs
a question with a ``context`` of ten paragraphs -- the two *gold* paragraphs that
together support the answer plus eight TF-IDF-retrieved *distractor* paragraphs --
and labels the exact gold sentences in ``supporting_facts``. HotpotQA is a *2-hop*
benchmark by construction: every answerable question requires composing a fact from
one gold paragraph with a fact from a second (a "bridge" entity links the two, or
two entities are "compared"), so reasoning crosses exactly two documents [1]_.

Mapping the schema onto :class:`~membench.types.Task`
-----------------------------------------------------
A record has the documented fields::

    {
        "_id": str,
        "question": str,
        "answer": str,
        "type": "bridge" | "comparison",
        "level": "easy" | "medium" | "hard",
        "supporting_facts": [[title, sent_id], ...],
        "context": [[title, [sentence, ...]], ...],
    }

* **Corpus.** Every sentence of every context paragraph becomes one
  :class:`~membench.types.MemoryItem`, so retrieval is scored at the same
  sentence granularity HotpotQA labels its evidence at. The ``item_id`` is the
  ``"{title} :: s{sent_id}"`` join key; ``metadata`` carries ``title``,
  ``sent_id``, ``is_supporting`` and the question ``type`` (so the structured arm
  can tell gold evidence from distractors when present).
* **Gold chain.** ``governing_decisions`` is the ordered, de-duplicated tuple of
  the supporting-sentence ``item_id``s -- the depth-labeled evidence chain the
  answer must rest on; the supporting-facts order is the documented hop order.
* **Depth.** ``depth`` is the number of *distinct gold paragraphs* (titles) the
  supporting facts span -- i.e. the hop count. For a well-formed HotpotQA example
  this is exactly ``2`` (the defining 2-hop property [1]_); a degenerate
  single-paragraph example clamps to ``depth = 1`` to satisfy the ``Task`` invariant.
* **Scorer.** ``scorer = QA_MATCH`` (a free-text answer, like LongMemEval) and
  ``repo_ref = None`` -- there is no repository and code-identifier compliance has
  no signal on encyclopedic prose. The gold answer rides alongside via
  :func:`hotpotqa_gold_answers`, mirroring the LongMemEval loader.

References
----------
.. [1] Z. Yang, P. Qi, S. Zhang, Y. Bengio, W. W. Cohen, R. Salakhutdinov, and
   C. D. Manning. "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question
   Answering." EMNLP 2018. arXiv:1809.09600. The distractor setting and the
   ``supporting_facts`` sentence-level evidence labels are defined in Sections 1-3.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from membench.types import MemoryItem, Scorer, SpecVariant, Task

__all__ = ["hotpotqa_gold_answers", "load_hotpotqa_tasks", "sentence_item_id"]

_HotpotRecord = Mapping[str, Any]


def sentence_item_id(title: str, sent_id: int) -> str:
    """Return the stable corpus ``item_id`` for one context sentence.

    Parameters
    ----------
    title
        The paragraph title the sentence belongs to.
    sent_id
        The zero-based index of the sentence within that paragraph.

    Returns
    -------
    str
        The ``"{title} :: s{sent_id}"`` join key used for both the corpus item
        and the gold supporting-fact references, so the two always agree.
    """
    return f"{title} :: s{sent_id}"


def _read_records(source: str | Path | Sequence[_HotpotRecord]) -> list[_HotpotRecord]:
    """Load records from a JSON file path or accept an in-memory record sequence."""
    if isinstance(source, (str, Path)):
        loaded: Any = json.loads(Path(source).read_text())
        if not isinstance(loaded, list):
            raise ValueError("HotpotQA file must contain a JSON list of records")
        return list(loaded)
    return list(source)


def _build_corpus(
    context: Sequence[Sequence[Any]], supporting: set[tuple[str, int]]
) -> tuple[MemoryItem, ...]:
    """Flatten ``context`` paragraphs into one sentence-level item per sentence."""
    items: list[MemoryItem] = []
    for title, sentences in context:
        for sent_id, sentence in enumerate(sentences):
            items.append(
                MemoryItem(
                    item_id=sentence_item_id(title, sent_id),
                    text=sentence,
                    metadata={
                        "title": title,
                        "sent_id": sent_id,
                        "is_supporting": (title, sent_id) in supporting,
                    },
                )
            )
    return tuple(items)


def load_hotpotqa_tasks(
    source: str | Path | Sequence[_HotpotRecord],
    *,
    levels: Sequence[str] | None = None,
) -> list[Task]:
    """Parse HotpotQA distractor records into :class:`~membench.types.Task` objects.

    Parameters
    ----------
    source
        Either a path to a HotpotQA distractor JSON file (a list of records) or an
        already-parsed sequence of record mappings. No download is performed.
    levels
        Optional difficulty filter; a subset of ``{"easy", "medium", "hard"}``.
        ``None`` keeps every record.

    Returns
    -------
    list[membench.types.Task]
        One task per record. ``governing_decisions`` is the ordered gold
        supporting-sentence chain, ``depth`` is the number of distinct gold
        paragraphs (the hop count, ``2`` for a well-formed example), the corpus is
        every context sentence, and ``scorer`` is :attr:`~membench.types.Scorer.QA_MATCH`.

    Raises
    ------
    ValueError
        If a supporting fact references a ``(title, sent_id)`` absent from the
        record's context (a corrupt example), or ``levels`` is not a subset of the
        three documented difficulty levels.
    """
    wanted_levels = frozenset(levels) if levels is not None else None
    if wanted_levels is not None and not wanted_levels <= {"easy", "medium", "hard"}:
        raise ValueError("levels must be a subset of {'easy', 'medium', 'hard'}")

    tasks: list[Task] = []
    for record in _read_records(source):
        if wanted_levels is not None and record.get("level") not in wanted_levels:
            continue

        context = record["context"]
        present = {
            sentence_item_id(title, sent_id)
            for title, sentences in context
            for sent_id in range(len(sentences))
        }
        supporting_pairs = [(title, int(sent_id)) for title, sent_id in record["supporting_facts"]]
        supporting_set = set(supporting_pairs)

        # Ordered, de-duplicated gold chain in the documented supporting-facts order.
        chain: list[str] = list(
            dict.fromkeys(sentence_item_id(title, sent_id) for title, sent_id in supporting_pairs)
        )
        missing = [gold_id for gold_id in chain if gold_id not in present]
        if missing:
            raise ValueError(
                f"HotpotQA record {record['_id']}: supporting facts reference "
                f"sentences absent from context: {missing}"
            )

        # Depth = number of distinct gold paragraphs (titles) = the hop count.
        gold_titles = dict.fromkeys(title for title, _ in supporting_pairs)
        depth = max(len(gold_titles), 1)

        tasks.append(
            Task(
                task_id=f"hotpotqa-{record['_id']}",
                dataset="hotpotqa",
                query=record["question"],
                repo_ref=None,
                memory_corpus=_build_corpus(context, supporting_set),
                governing_decisions=tuple(chain),
                depth=depth,
                spec_variant=SpecVariant.FULL,  # never spec-stripped; see module docstring
                scorer=Scorer.QA_MATCH,
            )
        )
    return tasks


def hotpotqa_gold_answers(source: str | Path | Sequence[_HotpotRecord]) -> dict[str, str]:
    """Return ``{task_id: gold_answer}`` for the QA-match scorer.

    Parameters
    ----------
    source
        The same path or record sequence accepted by :func:`load_hotpotqa_tasks`.

    Returns
    -------
    dict[str, str]
        Maps each ``"hotpotqa-{_id}"`` task id to its gold answer string.
    """
    return {f"hotpotqa-{record['_id']}": str(record["answer"]) for record in _read_records(source)}
