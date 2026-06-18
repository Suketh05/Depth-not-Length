"""Tests for the synthetic depth-controlled dataset.

These assert *dataset properties* (structure, similarity decay, link reachability)
and the qualitative mechanism, not specific recall numbers (those belong to the
analysis layer).
"""

from __future__ import annotations

import pytest

from membench.agents.combos import Combo, build_system
from membench.agents.runner import RunConfig, run_task
from membench.datasets.synthetic import AREAS, load_synthetic_tasks
from membench.retrieval import builtin  # noqa: F401
from membench.retrieval._text import tokenize


def _recall(rec: object) -> float:
    gold = set(rec.governing_decisions)  # type: ignore[attr-defined]
    got = set(rec.retrieved_ids)  # type: ignore[attr-defined]
    return len(gold & got) / len(gold)


class TestStructure:
    def test_depth_controls_chain_length(self) -> None:
        for depth in (1, 2, 3):
            task = load_synthetic_tasks(depths=(depth,), per_depth=1, seed=0)[0]
            assert task.depth == depth
            # governing node is the terminal of a length-(depth+1) target chain
            assert task.governing_decisions[0].endswith("-rule")

    def test_governing_node_carries_a_guard_identifier(self) -> None:
        task = load_synthetic_tasks(depths=(2,), per_depth=1, seed=0)[0]
        rule = task.corpus_by_id[task.governing_decisions[0]]
        assert "()" in rule.text  # the code identifier the rule mandates

    def test_corpus_has_distractors_and_filler(self) -> None:
        task = load_synthetic_tasks(depths=(2,), per_depth=1, seed=0, distractor_areas=4)[0]
        ids = set(task.corpus_by_id)
        assert any(i.startswith("filler-") for i in ids)
        # nodes from more than one area (target + distractors)
        slugs = {i.split("-")[0] for i in ids if not i.startswith("filler")}
        assert len(slugs) >= 3

    def test_deterministic_with_seed(self) -> None:
        a = load_synthetic_tasks(depths=(2,), per_depth=5, seed=7)
        b = load_synthetic_tasks(depths=(2,), per_depth=5, seed=7)
        assert [t.task_id for t in a] == [t.task_id for t in b]
        assert [tuple(i.item_id for i in t.memory_corpus) for t in a] == [
            tuple(i.item_id for i in t.memory_corpus) for t in b
        ]


class TestSimilarityDecay:
    def test_governing_overlaps_query_less_than_entry(self) -> None:
        # the decay property: entry resembles the query; the governing rule does not
        task = load_synthetic_tasks(depths=(3,), per_depth=1, seed=0)[0]
        q = set(tokenize(task.query))
        by_id = task.corpus_by_id
        entry = next(i for i in by_id.values() if i.item_id.endswith("-entry"))
        rule = by_id[task.governing_decisions[0]]
        entry_overlap = len(q & set(tokenize(entry.text)))
        rule_overlap = len(q & set(tokenize(rule.text)))
        assert entry_overlap > rule_overlap


class TestMechanism:
    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_graph_recovers_governing_within_hop_budget(self, depth: int) -> None:
        # The structured arm follows links to the governing rule at depth <= 3.
        task = load_synthetic_tasks(depths=(depth,), per_depth=1, seed=0)[0]
        llm, memory = build_system(Combo("g", "claude", "brief_graph_3hop"), offline=True)
        rec = run_task(
            task, memory, llm, RunConfig(budget_tokens=150), arm_name="brief_graph_3hop"
        )[0]
        assert _recall(rec) == 1.0

    def test_areas_cover_realistic_domains(self) -> None:
        slugs = {a.slug for a in AREAS}
        assert {"export", "auth", "billing"} <= slugs
        assert len(AREAS) >= 8
