"""Tests for the core contracts in :mod:`membench.types`."""

from __future__ import annotations

import dataclasses

import pytest

from membench.types import MemoryItem, RetrievedContext, Scorer, SpecVariant, Task


def _item(item_id: str, text: str = "x", **meta: object) -> MemoryItem:
    return MemoryItem(item_id=item_id, text=text, metadata=meta)


class TestMemoryItem:
    def test_rejects_empty_id(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            MemoryItem(item_id="", text="hello")

    def test_metadata_defaults_empty_and_is_readonly(self) -> None:
        item = MemoryItem(item_id="D-1", text="t")
        assert dict(item.metadata) == {}
        with pytest.raises(TypeError):
            item.metadata["k"] = "v"  # type: ignore[index]

    def test_metadata_is_frozen_even_when_passed_a_mutable_dict(self) -> None:
        source = {"node_type": "constraint"}
        item = MemoryItem(item_id="D-1", text="t", metadata=source)
        source["node_type"] = "mutated"
        # The item took a defensive copy; external mutation does not leak in.
        assert item.node_type == "constraint"

    def test_is_frozen(self) -> None:
        item = _item("D-1")
        with pytest.raises(dataclasses.FrozenInstanceError):
            item.text = "new"  # type: ignore[misc]

    def test_graph_link_accessors(self) -> None:
        constraint = _item("D-1", node_type="constraint")
        justification = _item("D-1-rationale", node_type="justification", constrains="D-1")
        assert constraint.node_type == "constraint"
        assert constraint.constrains is None
        assert justification.constrains == "D-1"

    @pytest.mark.parametrize(("text", "cpt", "expected"), [("a" * 40, 4, 10), ("abc", 4, 0)])
    def test_approx_tokens(self, text: str, cpt: int, expected: int) -> None:
        assert _item("i", text).approx_tokens(cpt) == expected

    def test_approx_tokens_rejects_nonpositive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _item("i").approx_tokens(0)


class TestRetrievedContext:
    def test_empty_factory(self) -> None:
        ctx = RetrievedContext.empty()
        assert ctx.text == "" and ctx.ids == ()

    def test_ids_normalised_to_tuple(self) -> None:
        ctx = RetrievedContext(text="t", ids=["a", "b"])  # type: ignore[arg-type]
        assert ctx.ids == ("a", "b")
        assert isinstance(ctx.ids, tuple)


class TestTask:
    def _task(self, **overrides: object) -> Task:
        base: dict[str, object] = {
            "task_id": "dcbench-TASK-1",
            "dataset": "dcbench",
            "query": "do the thing",
            "repo_ref": "github.com/x/y@deadbeef",
            "memory_corpus": (_item("D-1"), _item("D-2")),
            "governing_decisions": ("D-1",),
            "depth": 1,
            "spec_variant": SpecVariant.FULL,
            "scorer": Scorer.COMPLIANCE,
        }
        base.update(overrides)
        return Task(**base)  # type: ignore[arg-type]

    def test_coerces_string_enums(self) -> None:
        task = self._task(spec_variant="stripped", depth=2, scorer="compliance")
        assert task.spec_variant is SpecVariant.STRIPPED
        assert task.scorer is Scorer.COMPLIANCE

    def test_rejects_zero_depth(self) -> None:
        with pytest.raises(ValueError, match="depth"):
            self._task(depth=0)

    def test_rejects_duplicate_corpus_ids(self) -> None:
        with pytest.raises(ValueError, match="not unique"):
            self._task(memory_corpus=(_item("D-1"), _item("D-1")))

    def test_is_coding_reflects_repo_ref(self) -> None:
        assert self._task().is_coding is True
        assert self._task(repo_ref=None).is_coding is False

    def test_corpus_by_id_indexes_items(self) -> None:
        task = self._task()
        assert task.corpus_by_id["D-2"].item_id == "D-2"
        assert set(task.corpus_by_id) == {"D-1", "D-2"}
