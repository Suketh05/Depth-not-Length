"""Tests for the HotpotQA distractor-split loader.

The fixture is the canonical bridge example from Figure 1 of the HotpotQA paper
(Yang et al., EMNLP 2018, arXiv:1809.09600): the question

    "What is the middle name of the character that Allie Goertz wrote a song about?"

is answered by composing one fact from the *Allie Goertz* paragraph (she wrote a
song about Milhouse) with one fact from the *Milhouse* paragraph (his full name is
"Milhouse Mussolini Van Houten") -- a textbook 2-hop / bridge question whose answer
is "Mussolini". A topically-unrelated *Rick and Morty* paragraph is added as a
distractor, matching the distractor-setting protocol.

GOLDEN VALUES (hand-derived from the schema, NOT from running the loader):
* governing_decisions == ("Allie Goertz :: s1", "Milhouse :: s0")
      -- the two supporting_facts [["Allie Goertz", 1], ["Milhouse", 0]] mapped
         through the documented "{title} :: s{sent_id}" join key, in order.
* depth == 2  -- the supporting facts span exactly two distinct paragraphs
      (titles), which IS the defining 2-hop property of HotpotQA.
* corpus size == 6  -- 2 + 2 + 2 sentences across the three context paragraphs.
* gold answer == "Mussolini".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from membench.datasets.hotpotqa import (
    hotpotqa_gold_answers,
    load_hotpotqa_tasks,
    sentence_item_id,
)
from membench.retrieval.bm25 import BM25Memory
from membench.types import Scorer, SpecVariant

_RECORD: dict[str, Any] = {
    "_id": "5a7a06935542990198eaf050",
    "type": "bridge",
    "level": "hard",
    "question": "What is the middle name of the character that Allie Goertz wrote a song about?",
    "answer": "Mussolini",
    "supporting_facts": [["Allie Goertz", 1], ["Milhouse", 0]],
    "context": [
        [
            "Allie Goertz",
            [
                "Allie Goertz is an American musician and writer.",
                "She wrote a song about the Simpsons character Milhouse.",
            ],
        ],
        [
            "Milhouse",
            [
                "Milhouse Mussolini Van Houten is a character in the series The Simpsons.",
                "Milhouse was named after President Richard Nixon.",
            ],
        ],
        [
            "Rick and Morty",
            [
                "Rick and Morty is an adult animated science fiction sitcom.",
                "The series follows a scientist and his grandson on interdimensional adventures.",
            ],
        ],
    ],
}


def _records() -> list[dict[str, Any]]:
    return [_RECORD]


def test_loader_produces_hand_computed_golden_task() -> None:
    (task,) = load_hotpotqa_tasks(_records())

    # Identity / shape.
    assert task.task_id == "hotpotqa-5a7a06935542990198eaf050"
    assert task.dataset == "hotpotqa"
    assert task.query == _RECORD["question"]
    assert task.repo_ref is None
    assert task.spec_variant is SpecVariant.FULL
    assert task.scorer is Scorer.QA_MATCH

    # GOLDEN: the gold chain == the two supporting facts, in order, via the join key.
    assert task.governing_decisions == ("Allie Goertz :: s1", "Milhouse :: s0")

    # GOLDEN: depth == 2 distinct gold paragraphs == the 2-hop property.
    assert task.depth == 2

    # GOLDEN: corpus is every context sentence (2 + 2 + 2).
    assert len(task.memory_corpus) == 6
    assert all(g in task.corpus_by_id for g in task.governing_decisions)

    # The supporting flag marks exactly the two gold sentences.
    supporting = {i.item_id for i in task.memory_corpus if i.metadata["is_supporting"]}
    assert supporting == {"Allie Goertz :: s1", "Milhouse :: s0"}


def test_bm25_retrieves_a_supporting_sentence_over_the_distractor() -> None:
    (task,) = load_hotpotqa_tasks(_records())

    arm = BM25Memory()
    arm.write(task.memory_corpus)
    retrieved = arm.retrieve(task.query, budget_tokens=150)

    gold = set(task.governing_decisions)
    # An existing arm run over the produced corpus recovers gold evidence.
    assert gold & set(retrieved.ids)

    # BM25 must rank a gold supporting sentence above the topically-unrelated
    # distractor paragraph (this is a behavioural invariant of lexical overlap,
    # not a value copied back from the loader).
    gold_ranks = [retrieved.ids.index(g) for g in task.governing_decisions]
    distractor_ranks = [
        retrieved.ids.index(i.item_id)
        for i in task.memory_corpus
        if i.metadata["title"] == "Rick and Morty"
    ]
    assert min(gold_ranks) < min(distractor_ranks)


def test_gold_answer_lookup() -> None:
    answers = hotpotqa_gold_answers(_records())
    # GOLDEN: hand-read from the fixture.
    assert answers == {"hotpotqa-5a7a06935542990198eaf050": "Mussolini"}


def test_loads_from_json_file_path(tmp_path: Path) -> None:
    path = tmp_path / "hotpot_distractor.json"
    path.write_text(json.dumps(_records()))

    from_path = load_hotpotqa_tasks(path)
    from_records = load_hotpotqa_tasks(_records())

    assert from_path == from_records
    assert hotpotqa_gold_answers(path) == hotpotqa_gold_answers(_records())


def test_sentence_item_id_is_the_documented_join_key() -> None:
    assert sentence_item_id("Allie Goertz", 1) == "Allie Goertz :: s1"


def test_levels_filter_drops_non_matching_records() -> None:
    assert load_hotpotqa_tasks(_records(), levels=["easy"]) == []
    assert len(load_hotpotqa_tasks(_records(), levels=["hard"])) == 1
    with pytest.raises(ValueError, match="levels must be a subset"):
        load_hotpotqa_tasks(_records(), levels=["trivial"])


def test_supporting_fact_absent_from_context_is_rejected() -> None:
    bad = json.loads(json.dumps(_RECORD))  # deep copy
    bad["supporting_facts"] = [["Allie Goertz", 9]]  # sent_id out of range
    with pytest.raises(ValueError, match="absent from context"):
        load_hotpotqa_tasks([bad])
