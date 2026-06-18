"""Tests for the dcbench loader and its spec-stripping harness."""

from __future__ import annotations

import pytest

from membench.datasets.dcbench import load_dcbench_tasks
from membench.types import SpecVariant


class TestDepthDial:
    def test_full_is_depth_one_only(self) -> None:
        with pytest.raises(ValueError, match="depth must be 1"):
            load_dcbench_tasks(SpecVariant.FULL, depth=2)

    def test_stripped_depth_must_be_2_or_3(self) -> None:
        with pytest.raises(ValueError, match="depth 2 or 3"):
            load_dcbench_tasks(SpecVariant.STRIPPED, depth=1)

    def test_full_keeps_constraints_in_query(self) -> None:
        tasks = load_dcbench_tasks(SpecVariant.FULL, depth=1)
        assert tasks
        assert all("you must honour" in t.query.lower() for t in tasks)

    def test_stripped_removes_constraints_from_query(self) -> None:
        tasks = load_dcbench_tasks(SpecVariant.STRIPPED, depth=2)
        assert all("you must honour" not in t.query.lower() for t in tasks)


class TestCorpusShape:
    def test_combined_corpus_has_one_node_per_decision(self) -> None:
        tasks = load_dcbench_tasks(SpecVariant.STRIPPED, depth=2)
        corpus = tasks[0].memory_corpus
        # 15 seeded decisions, one node each; no justification split at d=2
        assert all(item.node_type == "decision" for item in corpus)
        assert not any(item.constrains for item in corpus)

    def test_split_corpus_links_justification_to_constraint(self) -> None:
        tasks = load_dcbench_tasks(SpecVariant.STRIPPED, depth=3)
        corpus = tasks[0].memory_corpus
        justifications = [i for i in corpus if i.node_type == "justification"]
        assert justifications
        for just in justifications:
            assert just.constrains is not None
            # the constraint it points at is present in the corpus
            assert any(c.item_id == just.constrains for c in corpus)

    def test_corpus_covers_all_decisions_not_just_governing(self) -> None:
        tasks = load_dcbench_tasks(SpecVariant.STRIPPED, depth=2)
        # corpus is larger than any single task's governing set (can't infer relevance)
        assert len(tasks[0].memory_corpus) > len(tasks[0].governing_decisions)


class TestTasks:
    def test_task_ids_prefixed_and_governing_populated(self) -> None:
        tasks = load_dcbench_tasks(SpecVariant.FULL, depth=1)
        assert all(t.task_id.startswith("dcbench-") for t in tasks)
        assert all(t.governing_decisions for t in tasks)
        assert all(t.is_coding for t in tasks)

    def test_task_id_filter(self) -> None:
        tasks = load_dcbench_tasks(SpecVariant.FULL, depth=1, task_ids=["TASK-001"])
        assert len(tasks) == 1 and tasks[0].task_id == "dcbench-TASK-001"
