import pytest

from benchmarks.swebench import load_swebench_tasks


def test_full_spec_states_constraint_in_the_query():
    tasks = load_swebench_tasks(spec_variant="full", depth=1, task_ids=["django__django-11019"])

    task = tasks[0]
    assert task.depth == 1
    assert task.spec_variant == "full"
    item = next(i for i in task.memory_corpus if i.item_id == task.governing_decisions[0])
    assert item.text in task.query


def test_stripped_d2_removes_constraint_from_query_keeps_combined_node():
    tasks = load_swebench_tasks(spec_variant="stripped", depth=2, task_ids=["django__django-11019"])

    task = tasks[0]
    assert task.depth == 2
    item = next(i for i in task.memory_corpus if i.item_id == task.governing_decisions[0])
    assert item.text not in task.query
    assert "class OrderedSet" in item.text or "def" in item.text


def test_stripped_d3_splits_constraint_and_justification_when_docstring_exists():
    tasks = load_swebench_tasks(spec_variant="stripped", depth=3, task_ids=["django__django-11019"])

    assert len(tasks) == 1  # this instance is certified for depth 3 (has a docstring)
    task = tasks[0]
    decision_id = task.governing_decisions[0]
    constraint = next(i for i in task.memory_corpus if i.item_id == decision_id)
    rationale = next(i for i in task.memory_corpus if i.item_id == f"{decision_id}-rationale")
    assert constraint.metadata["node_type"] == "constraint"
    assert rationale.metadata["node_type"] == "justification"
    assert rationale.metadata["constrains"] == decision_id


def test_instances_without_docstring_are_dropped_at_depth_3():
    # django__django-11283 (IntegrityError) has no docstring -- max_depth=2.
    tasks = load_swebench_tasks(spec_variant="stripped", depth=3, task_ids=["django__django-11283"])
    assert tasks == []

    tasks_d2 = load_swebench_tasks(spec_variant="stripped", depth=2, task_ids=["django__django-11283"])
    assert len(tasks_d2) == 1


def test_memory_corpus_is_pooled_per_repo_not_globally():
    tasks = load_swebench_tasks(spec_variant="stripped", depth=2, task_ids=["pytest-dev__pytest-5221"])

    task = tasks[0]
    assert all(i.metadata["repo"] == "pytest-dev/pytest" for i in task.memory_corpus)


def test_repo_ref_carries_repo_and_pinned_commit():
    tasks = load_swebench_tasks(spec_variant="full", depth=1, task_ids=["sphinx-doc__sphinx-7686"])
    assert tasks[0].repo_ref.startswith("github.com/sphinx-doc/sphinx@")


def test_full_spec_rejects_nonzero_depth():
    with pytest.raises(ValueError):
        load_swebench_tasks(spec_variant="full", depth=2)


def test_stripped_rejects_depth_outside_2_or_3():
    with pytest.raises(ValueError):
        load_swebench_tasks(spec_variant="stripped", depth=1)


def test_loads_all_21_instances_at_depth_2():
    tasks = load_swebench_tasks(spec_variant="stripped", depth=2)
    assert len(tasks) == 21
