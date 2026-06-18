import pytest

from benchmarks.dcbench import load_dcbench_tasks


def test_full_spec_states_decision_in_the_query():
    tasks = load_dcbench_tasks(spec_variant="full", depth=1, task_ids=["TASK-001"])

    task = tasks[0]
    assert task.depth == 1
    assert task.spec_variant == "full"
    decision_sentence = next(
        item.text.split("\n\n")[0]
        for item in task.memory_corpus
        if item.item_id == task.governing_decisions[0]
    )
    assert decision_sentence in task.query


def test_stripped_d2_removes_decision_from_query_keeps_combined_corpus_node():
    tasks = load_dcbench_tasks(spec_variant="stripped", depth=2, task_ids=["TASK-001"])

    task = tasks[0]
    assert task.depth == 2
    assert task.spec_variant == "stripped"
    for decision_id in task.governing_decisions:
        item = next(i for i in task.memory_corpus if i.item_id == decision_id)
        assert "Rationale:" in item.text  # constraint + rationale combined
        assert item.text.split("\n\n")[0] not in task.query


def test_stripped_d3_splits_constraint_and_rationale_into_linked_nodes():
    tasks = load_dcbench_tasks(spec_variant="stripped", depth=3, task_ids=["TASK-001"])

    task = tasks[0]
    assert task.depth == 3
    decision_id = task.governing_decisions[0]
    constraint = next(i for i in task.memory_corpus if i.item_id == decision_id)
    rationale = next(
        i for i in task.memory_corpus if i.item_id == f"{decision_id}-rationale"
    )
    assert constraint.metadata["node_type"] == "constraint"
    assert rationale.metadata["node_type"] == "justification"
    assert rationale.metadata["constrains"] == decision_id
    assert constraint.text not in task.query


def test_full_spec_rejects_nonzero_depth():
    with pytest.raises(ValueError):
        load_dcbench_tasks(spec_variant="full", depth=2)


def test_stripped_rejects_depth_outside_2_or_3():
    with pytest.raises(ValueError):
        load_dcbench_tasks(spec_variant="stripped", depth=1)
    with pytest.raises(ValueError):
        load_dcbench_tasks(spec_variant="stripped", depth=4)


def test_unknown_spec_variant_rejected():
    with pytest.raises(ValueError):
        load_dcbench_tasks(spec_variant="bogus")


def test_memory_corpus_covers_every_seeded_decision_not_just_governing_ones():
    tasks = load_dcbench_tasks(spec_variant="stripped", depth=2, task_ids=["TASK-002"])

    task = tasks[0]
    assert len(task.governing_decisions) < 15
    assert len({i.item_id for i in task.memory_corpus}) == 15
