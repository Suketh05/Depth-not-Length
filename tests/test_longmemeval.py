"""Tests for the LongMemEval loader."""

from __future__ import annotations

import pytest

from membench.datasets.longmemeval import (
    ALLOWED_QUESTION_TYPES,
    gold_answers,
    load_longmemeval_tasks,
)
from membench.types import Scorer, SpecVariant


class TestLoader:
    def test_loads_both_splits_by_default(self) -> None:
        tasks = load_longmemeval_tasks()
        assert tasks
        assert all(t.dataset == "longmemeval" for t in tasks)

    def test_scorer_is_qa_match_and_no_repo(self) -> None:
        for t in load_longmemeval_tasks():
            assert t.scorer is Scorer.QA_MATCH
            assert t.repo_ref is None
            assert not t.is_coding
            assert t.spec_variant is SpecVariant.FULL

    def test_depth_read_from_structure(self) -> None:
        depths = {t.depth for t in load_longmemeval_tasks()}
        assert depths <= {1, 2, 3} and depths  # native multi-hop structure

    def test_governing_decisions_are_answer_sessions_present_in_corpus(self) -> None:
        for t in load_longmemeval_tasks():
            assert t.governing_decisions
            for sid in t.governing_decisions:
                assert sid in t.corpus_by_id

    def test_question_type_filter(self) -> None:
        only = load_longmemeval_tasks(question_types=["knowledge-update"])
        assert only and len(only) < len(load_longmemeval_tasks())

    def test_rejects_out_of_scope_split(self) -> None:
        with pytest.raises(ValueError, match="subset"):
            load_longmemeval_tasks(question_types=["single-session-user"])

    def test_allowed_types_locked(self) -> None:
        assert {"temporal-reasoning", "knowledge-update"} == ALLOWED_QUESTION_TYPES


class TestGoldAnswers:
    def test_gold_answers_align_with_tasks(self) -> None:
        gold = gold_answers()
        task_ids = {t.task_id for t in load_longmemeval_tasks()}
        assert set(gold) == task_ids
        assert all(isinstance(v, str) and v for v in gold.values())
