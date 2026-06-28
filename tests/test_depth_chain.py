"""Tests for the parametric N-hop reasoning-chain generator.

These assert *dataset properties* -- the depth invariant (gold node exactly ``N``
typed-link hops from the query), determinism, count, valid Tasks, and that a real
arm runs over the tasks -- not specific recall numbers, which belong to the
analysis layer.

The headline golden is a textbook graph-theory fact, derived independently of the
generator's own code: a depth-``d`` reasoning chain is a path graph ``P_{d+1}``,
which has exactly ``d + 1`` vertices, ``d`` edges, and endpoint distance ``d``
(Bondy & Murty, *Graph Theory*, 2008, ch. 1). So for ``d = 4`` the target chain
must have 5 nodes, 4 typed edges, and the governing node must be reachable from
the entry in exactly 4 hops.
"""

from __future__ import annotations

import pytest

from membench.agents.combos import Combo, build_system
from membench.agents.runner import RunConfig, run_task
from membench.datasets.depth_chain import (
    DEFAULT_RELATIONS,
    TOPICS,
    generate_depth_chain_tasks,
)
from membench.retrieval import builtin  # noqa: F401  populate the arm REGISTRY
from membench.types import Scorer, SpecVariant, Task


def _target_chain_ids(task: Task) -> list[str]:
    """The target chain's node ids (the ``-tgt-`` keyed nodes), entry-first."""
    ids = [i for i in task.corpus_by_id if "-tgt-h" in i]
    return sorted(ids, key=lambda i: int(i.rsplit("-h", 1)[1]))


def _walk_typed_hops(task: Task) -> tuple[int, str]:
    """Follow the single outgoing typed edge from the entry; return (hops, final_id).

    This walks the ``edges`` metadata the generator emitted -- it does NOT call any
    distance routine of the generator -- so the hop count is an independent check
    of the constructed artefact against the path-graph formula.
    """
    by_id = task.corpus_by_id
    entry = next(i for i in by_id if i.endswith("-tgt-h0"))
    current = entry
    hops = 0
    while True:
        edges = list(by_id[current].metadata.get("edges", ()))
        if not edges:
            return hops, current
        current = edges[0][0]
        hops += 1


class TestDepthInvariant:
    @pytest.mark.parametrize("depth", [1, 2, 3, 4])
    def test_gold_is_exactly_depth_typed_hops_from_query(self, depth: int) -> None:
        # GOLDEN (path graph P_{d+1}): nodes = d + 1, edges = d, endpoint dist = d.
        # e.g. d = 4 -> 5 nodes, 4 edges, governing reached in exactly 4 hops.
        task = generate_depth_chain_tasks(depths=(depth,), per_depth=1, seed=0)[0]

        chain = _target_chain_ids(task)
        assert len(chain) == depth + 1  # nodes = d + 1

        governing_id = task.governing_decisions[0]
        assert governing_id == chain[-1]
        assert governing_id.endswith(f"-h{depth}")

        hops, reached = _walk_typed_hops(task)
        assert hops == depth  # endpoint distance = d (the depth-ladder invariant)
        assert reached == governing_id

    def test_final_hop_is_a_constrains_edge(self) -> None:
        # The last typed link must be CONSTRAINS so the graph arm recovers the
        # governing node; earlier links cycle through the intermediate relations.
        task = generate_depth_chain_tasks(depths=(3,), per_depth=1, seed=0)[0]
        by_id = task.corpus_by_id
        penultimate = next(i for i in by_id if i.endswith("-tgt-h2"))
        rels = [rel for _, rel in by_id[penultimate].metadata["edges"]]
        assert rels == ["constrains"]
        entry = next(i for i in by_id if i.endswith("-tgt-h0"))
        assert by_id[entry].metadata["edges"][0][1] in DEFAULT_RELATIONS

    def test_governing_node_carries_code_identifier(self) -> None:
        task = generate_depth_chain_tasks(depths=(2,), per_depth=1, seed=0)[0]
        rule = task.corpus_by_id[task.governing_decisions[0]]
        assert "()" in rule.text  # the compliance target


class TestCorpusShape:
    def test_corpus_size_matches_hand_count(self) -> None:
        # GOLDEN corpus census for depth d with D distractor chains and F filler:
        #   chain nodes = (D + 1) * (d + 1);  filler nodes = F (= 4).
        # depth=2, D=5  ->  6 * 3 = 18 chain nodes + 4 filler = 22 corpus items.
        task = generate_depth_chain_tasks(depths=(2,), per_depth=1, seed=0, distractor_chains=5)[0]
        ids = set(task.corpus_by_id)
        chain_nodes = [i for i in ids if "-h" in i and "-fill" not in i]
        filler_nodes = [i for i in ids if "-fill" in i]
        assert len(filler_nodes) == 4
        assert len(chain_nodes) == (5 + 1) * (2 + 1)  # 18
        assert len(ids) == 22

    def test_corpus_has_distractor_chains(self) -> None:
        task = generate_depth_chain_tasks(depths=(2,), per_depth=1, seed=0, distractor_chains=4)[0]
        keys = {i.split("-h")[0] for i in task.corpus_by_id if "-dis" in i}
        assert len(keys) == 4  # four distinct distractor chains


class TestCount:
    def test_per_depth_count_honored(self) -> None:
        tasks = generate_depth_chain_tasks(depths=(1, 2, 3), per_depth=7, seed=0)
        assert len(tasks) == 3 * 7
        for depth in (1, 2, 3):
            assert sum(t.depth == depth for t in tasks) == 7


class TestDeterminism:
    def test_deterministic_by_seed(self) -> None:
        a = generate_depth_chain_tasks(depths=(2, 3), per_depth=5, seed=7)
        b = generate_depth_chain_tasks(depths=(2, 3), per_depth=5, seed=7)
        assert [t.task_id for t in a] == [t.task_id for t in b]
        assert [tuple(i.item_id for i in t.memory_corpus) for t in a] == [
            tuple(i.item_id for i in t.memory_corpus) for t in b
        ]

    def test_seed_changes_corpus_order(self) -> None:
        a = generate_depth_chain_tasks(depths=(3,), per_depth=3, seed=0)
        b = generate_depth_chain_tasks(depths=(3,), per_depth=3, seed=1)
        # Same task ids (deterministic schema) but the shuffled order differs.
        assert [t.task_id for t in a] == [t.task_id for t in b]
        assert [tuple(i.item_id for i in t.memory_corpus) for t in a] != [
            tuple(i.item_id for i in t.memory_corpus) for t in b
        ]


class TestValidTasks:
    def test_tasks_are_well_formed(self) -> None:
        tasks = generate_depth_chain_tasks(depths=(1, 2, 3, 4), per_depth=2, seed=0)
        for task in tasks:
            assert isinstance(task, Task)
            assert task.dataset == "depth_chain"
            assert task.spec_variant is SpecVariant.STRIPPED
            assert task.scorer is Scorer.COMPLIANCE
            assert task.is_coding  # has a repo_ref
            # gold is present in the corpus and uniquely identified
            assert task.governing_decisions[0] in task.corpus_by_id
            assert len(task.governing_decisions) == 1

    def test_invalid_arguments_rejected(self) -> None:
        with pytest.raises(ValueError):
            generate_depth_chain_tasks(depths=(0,), per_depth=1)
        with pytest.raises(ValueError):
            generate_depth_chain_tasks(depths=(2,), per_depth=0)
        with pytest.raises(ValueError):
            generate_depth_chain_tasks(depths=(2,), per_depth=1, relations=())

    def test_topic_pool_is_large_enough_for_distractors(self) -> None:
        # Distinct vocabulary per chain requires more topics than chains per task.
        assert len(TOPICS) >= 8
        assert len({t.slug for t in TOPICS}) == len(TOPICS)


class TestArmRunsOverThem:
    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_graph_arm_recovers_gold_within_hop_budget(self, depth: int) -> None:
        # A 3-hop link-following arm reaches the governing node for depth <= 3.
        task = generate_depth_chain_tasks(depths=(depth,), per_depth=1, seed=0)[0]
        llm, memory = build_system(Combo("g", "claude", "brief_graph_3hop"), offline=True)
        rec = run_task(
            task, memory, llm, RunConfig(budget_tokens=150), arm_name="brief_graph_3hop"
        )[0]
        gold = set(rec.governing_decisions)
        got = set(rec.retrieved_ids)
        assert gold & got == gold  # full recall of the deep governing node

    def test_arm_runs_over_a_deep_chain(self) -> None:
        # Beyond the hop budget the arm still runs and produces a record (the
        # generator emits valid tasks at any depth, recovery aside).
        task = generate_depth_chain_tasks(depths=(5,), per_depth=1, seed=0)[0]
        llm, memory = build_system(Combo("g", "claude", "brief_graph_3hop"), offline=True)
        records = run_task(
            task, memory, llm, RunConfig(budget_tokens=150), arm_name="brief_graph_3hop"
        )
        assert len(records) >= 1
        assert records[0].task_id == task.task_id
