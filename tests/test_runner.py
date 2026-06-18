"""Tests for the task runner and the builtin arm registry."""

from __future__ import annotations

import pytest

from membench.agents.combos import build_system
from membench.agents.runner import AttemptRecord, RunConfig, run_task
from membench.retrieval import builtin
from membench.types import MemoryItem, Scorer, SpecVariant, Task


def _task() -> Task:
    corpus = (
        MemoryItem("D-1", "primary buttons must use the Button component", {"node_type": "c"}),
        MemoryItem(
            "D-1-rationale",
            "the export dialog call to action should be accessible",
            {"node_type": "j", "constrains": "D-1"},
        ),
        MemoryItem("D-2", "use the DateRangePicker widget"),
    )
    return Task(
        task_id="t1",
        dataset="dcbench",
        query="make the export dialog call to action accessible",
        repo_ref="github.com/x/y@a",
        memory_corpus=corpus,
        governing_decisions=("D-1",),
        depth=3,
        spec_variant=SpecVariant.STRIPPED,
        scorer=Scorer.COMPLIANCE,
    )


class TestBuiltin:
    def test_registers_all_arms(self) -> None:
        arms = builtin.available_arms()
        for expected in ("none", "bm25", "dense", "brief_graph", "mem0", "raptor", "zep"):
            assert expected in arms
        assert len(arms) >= 20  # 20+ systems


class TestRunConfig:
    def test_rejects_bad_policy(self) -> None:
        with pytest.raises(ValueError, match="retry_policy"):
            RunConfig(budget_tokens=100, retry_policy="bogus")


class TestRunTask:
    def test_single_shot_one_record(self) -> None:
        llm, memory = build_system_offline("none")
        records = run_task(_task(), memory, llm, RunConfig(budget_tokens=250), arm_name="none")
        assert len(records) == 1
        rec = records[0]
        assert isinstance(rec, AttemptRecord)
        assert rec.arm == "none" and rec.depth == 3 and rec.spec_variant == "stripped"

    def test_graph_arm_recovers_governing_via_link(self) -> None:
        # The stub honours identifiers in context; the graph arm puts D-1 there.
        llm, memory = build_system_offline("brief_graph")
        rec = run_task(
            _task(), memory, llm, RunConfig(budget_tokens=10_000), arm_name="brief_graph"
        )[0]
        assert "D-1" in rec.retrieved_ids  # recovered through the CONSTRAINS link

    def test_none_arm_retrieves_nothing(self) -> None:
        llm, memory = build_system_offline("none")
        rec = run_task(_task(), memory, llm, RunConfig(budget_tokens=250), arm_name="none")[0]
        assert rec.retrieved_ids == ()

    def test_as_dict_is_json_friendly(self) -> None:
        llm, memory = build_system_offline("dense")
        rec = run_task(_task(), memory, llm, RunConfig(budget_tokens=250), arm_name="dense")[0]
        row = rec.as_dict()
        assert isinstance(row["retrieved_ids"], list)
        assert row["model"] == "claude-sonnet-4-6"

    def test_retry_until_correct_stops_when_correct(self) -> None:
        llm, memory = build_system_offline("none")
        calls = {"n": 0}

        def always_after_two(_text: str, _task: Task) -> bool:
            calls["n"] += 1
            return calls["n"] >= 2

        records = run_task(
            _task(),
            memory,
            llm,
            RunConfig(budget_tokens=250, retry_policy="retry_until_correct", max_attempts=5),
            arm_name="none",
            is_correct=always_after_two,
        )
        assert len(records) == 2


def build_system_offline(arm_name: str) -> tuple[object, object]:
    from membench.agents.combos import Combo

    return build_system(Combo(arm_name, "claude", arm_name), offline=True)
