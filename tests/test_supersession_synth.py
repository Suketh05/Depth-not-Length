"""Tests for the supersession-chain synthetic dataset.

These assert dataset *properties* -- the head/latest invariant, the typed
``supersedes`` edge chain, determinism, depth labelling -- plus a hand-computed
golden gold id and an end-to-end arm run (run_benchmark-style usage), not any
specific recall numbers (those belong to the analysis layer).

GOLDEN (derived by hand from the construction rules, not by running the code):
``len(SLOTS) == 6``. For ``generate_supersession_tasks(chain_len=3, n=1, seed=0)``
the single task ``i == 0`` targets ``SLOTS[(0 + 3) % 6] == SLOTS[3]`` and gets
``task_id == "supersede-c3-00"``. Its chain has revisions v0 -> v1 -> v2 where
v2 supersedes v1 supersedes v0, so the head (the latest revision, the unique node
with in-degree 0 under ``supersedes``) is ``v2``. Therefore::

    governing_decisions == ("supersede-c3-00-v2",)
    depth == chain_len == 3
"""

from __future__ import annotations

import pytest

from membench.agents.combos import Combo, build_system
from membench.agents.runner import RunConfig, run_task
from membench.datasets.supersession_synth import SLOTS, generate_supersession_tasks
from membench.retrieval import builtin  # noqa: F401  populate the arm REGISTRY
from membench.types import Task


def _superseded_targets(task: Task) -> dict[str, str]:
    """Map ``older_id -> newer_id`` for every ``supersedes`` edge in the target chain."""
    prefix = f"{task.task_id}-v"
    superseded: dict[str, str] = {}
    for item in task.memory_corpus:
        if not item.item_id.startswith(prefix):
            continue  # skip distractor-chain nodes (they carry a slot slug segment)
        for target, rel in item.metadata.get("edges", ()):
            if rel == "supersedes":
                superseded[target] = item.item_id
    return superseded


class TestGolden:
    def test_head_is_latest_and_matches_hand_computed_gold(self) -> None:
        task = generate_supersession_tasks(chain_len=3, n=1, seed=0)[0]
        # GOLDEN: see module docstring -- derived from construction, not from output.
        assert task.task_id == "supersede-c3-00"
        assert task.governing_decisions == ("supersede-c3-00-v2",)
        assert task.depth == 3

    def test_slots_pool_size_underpinning_the_golden(self) -> None:
        # The golden gold id depends on SLOTS rotation; pin the pool size.
        assert len(SLOTS) == 6

    def test_head_node_carries_a_concrete_value_identifier(self) -> None:
        task = generate_supersession_tasks(chain_len=3, n=1, seed=0)[0]
        head = task.corpus_by_id[task.governing_decisions[0]]
        # SLOTS[3] is the "queue" slot; rev 2 value is its third value (KAFKA).
        assert "KAFKA" in head.text


class TestChainStructure:
    @pytest.mark.parametrize("chain_len", [1, 2, 3, 5])
    def test_depth_equals_chain_len_and_chain_is_linear(self, chain_len: int) -> None:
        task = generate_supersession_tasks(chain_len=chain_len, n=1, seed=0, distractor_chains=0)[0]
        assert task.depth == chain_len

        target_ids = [i.item_id for i in task.memory_corpus if i.item_id.startswith(task.task_id)]
        assert len(target_ids) == chain_len  # exactly chain_len decision nodes

        head = task.governing_decisions[0]
        superseded = _superseded_targets(task)
        # Every node except the head is superseded exactly once -> the head is the
        # unique in-degree-0 (latest) node.
        assert head not in superseded
        assert set(superseded) == set(target_ids) - {head}
        assert len(superseded) == chain_len - 1

    def test_each_revision_supersedes_its_immediate_predecessor(self) -> None:
        task = generate_supersession_tasks(chain_len=4, n=1, seed=0, distractor_chains=0)[0]
        # v_k must supersede exactly v_{k-1}.
        for k in range(1, 4):
            node = task.corpus_by_id[f"{task.task_id}-v{k}"]
            assert list(node.metadata["edges"]) == [(f"{task.task_id}-v{k - 1}", "supersedes")]
        # The oldest revision has no outgoing supersedes edge.
        assert list(task.corpus_by_id[f"{task.task_id}-v0"].metadata["edges"]) == []

    def test_single_revision_chain_has_no_edges(self) -> None:
        task = generate_supersession_tasks(chain_len=1, n=1, seed=0, distractor_chains=0)[0]
        assert task.governing_decisions == (f"{task.task_id}-v0",)
        assert list(task.corpus_by_id[task.task_id + "-v0"].metadata["edges"]) == []


class TestDistractorsAndIntegrity:
    def test_distractor_chains_add_other_slot_noise_with_unique_ids(self) -> None:
        task = generate_supersession_tasks(chain_len=3, n=1, seed=1, distractor_chains=2)[0]
        ids = [i.item_id for i in task.memory_corpus]
        assert len(ids) == len(set(ids))  # Task.__post_init__ would also reject dups
        # 1 target chain + 2 distractor chains, each of length 3.
        assert len(ids) == 3 * (1 + 2)
        # The head is still the target chain's latest revision, not a distractor's.
        assert task.governing_decisions[0].startswith(task.task_id + "-v")

    def test_distractor_count_clamped_to_available_slots(self) -> None:
        task = generate_supersession_tasks(chain_len=2, n=1, seed=0, distractor_chains=999)[0]
        # target + (len(SLOTS) - 1) distractor chains at most, each length 2.
        assert len(task.memory_corpus) == 2 * len(SLOTS)

    def test_rejects_bad_arguments(self) -> None:
        with pytest.raises(ValueError, match="chain_len"):
            generate_supersession_tasks(chain_len=0, n=1, seed=0)
        with pytest.raises(ValueError, match="n must be"):
            generate_supersession_tasks(chain_len=2, n=-1, seed=0)


class TestDeterminism:
    def test_same_seed_identical_tasks(self) -> None:
        a = generate_supersession_tasks(chain_len=3, n=5, seed=7)
        b = generate_supersession_tasks(chain_len=3, n=5, seed=7)
        assert [t.task_id for t in a] == [t.task_id for t in b]
        assert [t.governing_decisions for t in a] == [t.governing_decisions for t in b]
        assert [tuple(i.item_id for i in t.memory_corpus) for t in a] == [
            tuple(i.item_id for i in t.memory_corpus) for t in b
        ]

    def test_gold_is_order_independent_of_shuffle(self) -> None:
        shuffled = generate_supersession_tasks(chain_len=3, n=3, seed=2, shuffle_corpus=True)
        ordered = generate_supersession_tasks(chain_len=3, n=3, seed=2, shuffle_corpus=False)
        assert [t.governing_decisions for t in shuffled] == [t.governing_decisions for t in ordered]


class TestArmRun:
    def test_arm_runs_over_a_supersession_task(self) -> None:
        # run_benchmark-style usage: resolve an arm by name, run_task over a Task.
        task = generate_supersession_tasks(chain_len=3, n=1, seed=0)[0]
        llm, memory = build_system(Combo("s", "claude", "full_context"), offline=True)
        records = run_task(
            task, memory, llm, RunConfig(budget_tokens=100_000), arm_name="full_context"
        )
        assert len(records) == 1
        rec = records[0]
        assert rec.arm == "full_context"
        assert rec.depth == 3
        # full_context packs the whole corpus, so the gold head is recovered.
        assert task.governing_decisions[0] in rec.retrieved_ids
