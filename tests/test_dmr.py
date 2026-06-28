"""Tests for the DMR (Deep Memory Retrieval) loader.

The golden values are derived independently from the loader: DMR recall distance is
``d = S - g`` (``S`` sessions, gold at 0-based chronological index ``g``; see MemGPT,
Packer et al. 2023). For the fixture below:

* ``q1`` -- 3 sessions, gold ``s0`` at index 0 -> d = 3 - 0 = 3 (deepest)
* ``q2`` -- 4 sessions, gold ``s2`` at index 2 -> d = 4 - 2 = 2
* ``q3`` -- 2 sessions, gold ``s1`` at index 1 -> d = 2 - 1 = 1 (shallow)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from membench.datasets.dmr import gold_answers, load_dmr_tasks, recall_distance
from membench.retrieval.bm25 import BM25Memory
from membench.types import Scorer, SpecVariant


def _snapshot() -> dict[str, Any]:
    """A tiny, self-contained DMR snapshot (no external download)."""
    return {
        "examples": [
            {
                "question_id": "q1",
                "question": "What breed is my dog Rex?",
                "answer": "beagle",
                "gold_session_id": "s0",
                "sessions": [
                    {
                        "session_id": "s0",
                        "date": "2024-01-01",
                        "turns": [
                            {"role": "user", "content": "I just adopted a beagle named Rex."},
                            {"role": "assistant", "content": "Congrats on adopting Rex!"},
                        ],
                    },
                    {
                        "session_id": "s1",
                        "turns": [{"role": "user", "content": "I started learning the cello."}],
                    },
                    {
                        "session_id": "s2",
                        "turns": [{"role": "user", "content": "We booked a trip to Lisbon."}],
                    },
                ],
            },
            {
                "question_id": "q2",
                "question": "Which city did I move to?",
                "answer": "Porto",
                "gold_session_id": "s2",
                "sessions": [
                    {
                        "session_id": "s0",
                        "turns": [{"role": "user", "content": "I work as a cartographer."}],
                    },
                    {
                        "session_id": "s1",
                        "turns": [{"role": "user", "content": "My favorite color is teal."}],
                    },
                    {
                        "session_id": "s2",
                        "turns": [{"role": "user", "content": "I moved to a flat in Porto."}],
                    },
                    {
                        "session_id": "s3",
                        "turns": [{"role": "user", "content": "I'm trying a new pasta recipe."}],
                    },
                ],
            },
            {
                "question_id": "q3",
                "question": "What is my cat's name?",
                "answer": "Mochi",
                "gold_session_id": "s1",
                "sessions": [
                    {
                        "session_id": "s0",
                        "turns": [{"role": "user", "content": "I used to play the trumpet."}],
                    },
                    {
                        "session_id": "s1",
                        "turns": [{"role": "user", "content": "I got a tabby cat called Mochi."}],
                    },
                ],
            },
        ]
    }


def _write_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "dmr_snapshot.json"
    path.write_text(json.dumps(_snapshot()))
    return path


class TestRecallDistance:
    def test_golden_distances(self) -> None:
        # Hand-computed d = S - g, independent of the loader.
        assert recall_distance(0, 3) == 3  # gold oldest of 3 -> deepest
        assert recall_distance(2, 4) == 2
        assert recall_distance(1, 2) == 1  # gold most-recent of 2 -> shallow

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            recall_distance(3, 3)
        with pytest.raises(ValueError, match="out of range"):
            recall_distance(-1, 3)


class TestDepthIsRecallDistance:
    def test_depths_match_hand_computed_golden(self, tmp_path: Path) -> None:
        tasks = {t.task_id: t for t in load_dmr_tasks(_write_fixture(tmp_path))}
        # GOLDEN (see module docstring): d = S - g.
        assert tasks["dmr-q1"].depth == 3
        assert tasks["dmr-q2"].depth == 2
        assert tasks["dmr-q3"].depth == 1

    def test_deeper_fact_has_greater_depth(self, tmp_path: Path) -> None:
        tasks = {t.task_id: t for t in load_dmr_tasks(_write_fixture(tmp_path))}
        assert tasks["dmr-q1"].depth > tasks["dmr-q2"].depth > tasks["dmr-q3"].depth


class TestTaskStructure:
    def test_conversational_task_invariants(self, tmp_path: Path) -> None:
        for t in load_dmr_tasks(_write_fixture(tmp_path)):
            assert t.dataset == "dmr"
            assert t.task_id.startswith("dmr-")
            assert t.scorer is Scorer.QA_MATCH
            assert t.spec_variant is SpecVariant.FULL
            assert t.repo_ref is None
            assert not t.is_coding
            assert t.depth >= 1

    def test_gold_session_is_the_target_and_present(self, tmp_path: Path) -> None:
        for t in load_dmr_tasks(_write_fixture(tmp_path)):
            # exactly one governing decision: the deep gold session
            assert len(t.governing_decisions) == 1
            (gold_id,) = t.governing_decisions
            assert gold_id in t.corpus_by_id
            assert t.corpus_by_id[gold_id].metadata["is_gold"] is True
            # only the gold session is flagged gold
            golds = [i for i in t.memory_corpus if i.metadata["is_gold"]]
            assert [i.item_id for i in golds] == [gold_id]

    def test_corpus_covers_all_sessions(self, tmp_path: Path) -> None:
        tasks = {t.task_id: t for t in load_dmr_tasks(_write_fixture(tmp_path))}
        assert len(tasks["dmr-q2"].memory_corpus) == 4  # all 4 sessions seeded
        assert tasks["dmr-q2"].corpus_by_id["s2"].metadata["session_index"] == 2

    def test_missing_gold_session_raises(self, tmp_path: Path) -> None:
        snap = _snapshot()
        snap["examples"][0]["gold_session_id"] = "does-not-exist"
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(snap))
        with pytest.raises(ValueError, match="not found"):
            load_dmr_tasks(path)


class TestGoldAnswers:
    def test_gold_answers_align_with_tasks(self, tmp_path: Path) -> None:
        path = _write_fixture(tmp_path)
        gold = gold_answers(path)
        task_ids = {t.task_id for t in load_dmr_tasks(path)}
        assert set(gold) == task_ids
        assert gold["dmr-q1"] == "beagle"
        assert all(isinstance(v, str) and v for v in gold.values())


class TestArmRetrieval:
    def test_bm25_recovers_the_gold_session(self, tmp_path: Path) -> None:
        # A standard lexical arm should surface the gold session on the fixture:
        # the distinctive token "Rex" appears only in s0.
        tasks = {t.task_id: t for t in load_dmr_tasks(_write_fixture(tmp_path))}
        task = tasks["dmr-q1"]
        arm = BM25Memory()
        arm.write(task.memory_corpus)
        result = arm.retrieve(task.query, budget_tokens=200)
        (gold_id,) = task.governing_decisions
        assert gold_id in result.ids
        assert result.ids[0] == gold_id  # gold ranks first under BM25
