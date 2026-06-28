"""Tests for the distractor-scatter depth generator.

These assert *dataset properties*: determinism, valid depth-labelled Tasks, the
scatter=0 single-area baseline, and the hand-computed cross-area spread golden
A(d, s) = min(s + 1, d + 1).
"""

from __future__ import annotations

import pytest

from membench.datasets.scatter_synth import generate_scatter_tasks
from membench.datasets.synthetic import AREAS
from membench.types import Scorer, SpecVariant, Task


def _chain_areas(task: Task) -> list[str]:
    """Return the area slug of each chain (``-hop``) node, in hop order."""
    by_id = task.corpus_by_id
    hop = 0
    slugs: list[str] = []
    while True:
        item = by_id.get(f"{task.task_id}-hop{hop}")
        if item is None:
            break
        area = item.metadata["area"]
        slugs.append(str(area))
        hop += 1
    return slugs


def _distinct_areas(task: Task) -> int:
    """Number of distinct areas the chain spans."""
    return len(set(_chain_areas(task)))


class TestStructure:
    def test_valid_depth_labeled_tasks(self) -> None:
        for depth in (1, 2, 3):
            task = generate_scatter_tasks(depths=(depth,), scatter=2, per_depth=1, seed=0)[0]
            assert task.depth == depth
            assert task.dataset == "synthetic-scatter"
            assert task.spec_variant is SpecVariant.STRIPPED
            assert task.scorer is Scorer.COMPLIANCE
            # the chain has depth + 1 hop nodes; the gold is its terminal rule
            assert len(_chain_areas(task)) == depth + 1
            assert task.governing_decisions == (f"{task.task_id}-hop{depth}",)

    def test_governing_node_carries_a_guard_identifier(self) -> None:
        task = generate_scatter_tasks(depths=(2,), scatter=1, per_depth=1, seed=0)[0]
        rule = task.corpus_by_id[task.governing_decisions[0]]
        assert "()" in rule.text  # the code identifier compliance scoring targets

    def test_corpus_item_ids_unique(self) -> None:
        # Task.__post_init__ enforces this, so construction succeeding is the check;
        # assert explicitly across a richer parameterisation too.
        for task in generate_scatter_tasks(depths=(1, 2, 3), scatter=3, per_depth=4, seed=1):
            ids = [i.item_id for i in task.memory_corpus]
            assert len(ids) == len(set(ids))


class TestDeterminism:
    def test_deterministic_with_seed(self) -> None:
        a = generate_scatter_tasks(depths=(2,), scatter=2, per_depth=5, seed=7)
        b = generate_scatter_tasks(depths=(2,), scatter=2, per_depth=5, seed=7)
        assert [t.task_id for t in a] == [t.task_id for t in b]
        # full corpus identity (ids + texts), since the corpus is RNG-shuffled
        assert [[(i.item_id, i.text) for i in t.memory_corpus] for t in a] == [
            [(i.item_id, i.text) for i in t.memory_corpus] for t in b
        ]


class TestScatter:
    def test_scatter_zero_is_single_area(self) -> None:
        # scatter=0 -> the whole chain lives in one area (the baseline).
        for depth in (1, 2, 3):
            task = generate_scatter_tasks(depths=(depth,), scatter=0, per_depth=1, seed=0)[0]
            assert _distinct_areas(task) == 1

    def test_cross_area_spread_matches_closed_form(self) -> None:
        # GOLDEN (hand-computed, independent of the implementation):
        # a depth-d chain has d + 1 nodes assigned round-robin to a pool of
        # (scatter + 1) distinct areas, so distinct areas A(d, s) = min(s+1, d+1).
        # For d = 3 (4 nodes): hop->pool index is
        #   s=0: 0,0,0,0 -> {0}           -> 1
        #   s=1: 0,1,0,1 -> {0,1}         -> 2
        #   s=2: 0,1,2,0 -> {0,1,2}       -> 3
        #   s=3: 0,1,2,3 -> {0,1,2,3}     -> 4
        #   s=5: 0,1,2,3 -> {0,1,2,3}     -> 4 (saturated by chain length)
        depth = 3
        expected = {0: 1, 1: 2, 2: 3, 3: 4, 5: 4}
        for scatter, gold in expected.items():
            task = generate_scatter_tasks(depths=(depth,), scatter=scatter, per_depth=1, seed=0)[0]
            assert _distinct_areas(task) == gold == min(scatter + 1, depth + 1)

    def test_spread_is_monotonic_nondecreasing(self) -> None:
        depth = 3
        spreads = [
            _distinct_areas(
                generate_scatter_tasks(depths=(depth,), scatter=s, per_depth=1, seed=0)[0]
            )
            for s in range(len(AREAS))
        ]
        assert spreads == sorted(spreads)
        assert spreads[0] == 1
        assert max(spreads) == depth + 1  # saturates at one area per hop

    def test_entry_resembles_query_more_than_a_scattered_hop(self) -> None:
        # The decay property survives scatter: the target-area entry overlaps the
        # query; a hop dropped into a foreign area does not.
        task = generate_scatter_tasks(depths=(3,), scatter=3, per_depth=1, seed=0)[0]
        query_words = set(task.query.lower().split())
        by_id = task.corpus_by_id
        entry_words = set(by_id[f"{task.task_id}-hop0"].text.lower().split())
        rule_words = set(by_id[task.governing_decisions[0]].text.lower().split())
        assert len(query_words & entry_words) > len(query_words & rule_words)


class TestValidation:
    def test_rejects_negative_scatter(self) -> None:
        with pytest.raises(ValueError, match="scatter must be >= 0"):
            generate_scatter_tasks(depths=(2,), scatter=-1, per_depth=1, seed=0)

    def test_rejects_oversized_scatter(self) -> None:
        with pytest.raises(ValueError, match="scatter must be <="):
            generate_scatter_tasks(depths=(2,), scatter=len(AREAS), per_depth=1, seed=0)

    def test_rejects_zero_depth(self) -> None:
        with pytest.raises(ValueError, match="depth must be >= 1"):
            generate_scatter_tasks(depths=(0,), scatter=0, per_depth=1, seed=0)
