"""Tests for the SWE-bench loader."""

from __future__ import annotations

import pytest

from membench.datasets.swebench import load_swebench_tasks
from membench.types import SpecVariant


class TestDepthDial:
    def test_full_requires_depth_one(self) -> None:
        with pytest.raises(ValueError, match="depth must be 1"):
            load_swebench_tasks(SpecVariant.FULL, depth=2)

    def test_stripped_depth_range(self) -> None:
        with pytest.raises(ValueError, match="depth 2 or 3"):
            load_swebench_tasks(SpecVariant.STRIPPED, depth=4)

    def test_full_folds_code_into_query(self) -> None:
        tasks = load_swebench_tasks(SpecVariant.FULL, depth=1)
        assert tasks
        assert all("you must honour" in t.query.lower() for t in tasks)

    def test_stripped_removes_code_from_query(self) -> None:
        tasks = load_swebench_tasks(SpecVariant.STRIPPED, depth=2)
        assert all("you must honour" not in t.query.lower() for t in tasks)


class TestCorpusAndTasks:
    def test_governing_is_remote_qualname(self) -> None:
        tasks = load_swebench_tasks(SpecVariant.STRIPPED, depth=2)
        for t in tasks:
            assert len(t.governing_decisions) == 1
            assert t.governing_decisions[0] in t.corpus_by_id  # present in the pooled corpus

    def test_corpus_pooled_per_repo_not_mixed(self) -> None:
        tasks = load_swebench_tasks(SpecVariant.STRIPPED, depth=2)
        for t in tasks:
            repos = {i.metadata.get("repo") for i in t.memory_corpus}
            assert len(repos) == 1  # a task's haystack is its own repo only

    def test_split_links_justification_to_constraint(self) -> None:
        tasks = load_swebench_tasks(SpecVariant.STRIPPED, depth=3)
        # at least one instance has a docstring -> a justification node with a link
        justs = [i for t in tasks for i in t.memory_corpus if i.node_type == "justification"]
        assert justs
        for just in justs:
            assert just.constrains is not None

    def test_repos_are_the_three_expected(self) -> None:
        tasks = load_swebench_tasks(SpecVariant.FULL, depth=1)
        repos = {t.repo_ref.split("@")[0] for t in tasks if t.repo_ref}
        assert repos == {
            "github.com/django/django",
            "github.com/sphinx-doc/sphinx",
            "github.com/pytest-dev/pytest",
        }
