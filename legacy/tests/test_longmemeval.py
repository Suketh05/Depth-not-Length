import pytest

from benchmarks.longmemeval import load_longmemeval_tasks


def test_loads_only_locked_question_types():
    tasks = load_longmemeval_tasks()
    assert len(tasks) == 12  # the vendored snapshot's full set
    assert all(t.dataset == "longmemeval" for t in tasks)


def test_repo_ref_is_none_no_code_to_honor_a_decision_in():
    tasks = load_longmemeval_tasks()
    assert all(t.repo_ref is None for t in tasks)


def test_depth_is_read_off_answer_session_count_not_invented():
    tasks = load_longmemeval_tasks()
    for task in tasks:
        assert task.depth == min(len(task.governing_decisions), 3)
        assert 1 <= task.depth <= 3


def test_memory_corpus_includes_gold_and_filler_sessions():
    tasks = load_longmemeval_tasks()
    task = next(t for t in tasks if t.depth >= 1)
    gold_ids = set(task.governing_decisions)
    corpus_ids = {i.item_id for i in task.memory_corpus}
    assert gold_ids <= corpus_ids
    assert len(corpus_ids) > len(gold_ids)  # real distractor sessions present


def test_question_types_filter_rejects_unlocked_types():
    with pytest.raises(ValueError):
        load_longmemeval_tasks(question_types=["multi-session"])


def test_question_types_filter_narrows_to_requested_subset():
    tasks = load_longmemeval_tasks(question_types=["knowledge-update"])
    all_tasks = load_longmemeval_tasks()
    assert 0 < len(tasks) < len(all_tasks)
