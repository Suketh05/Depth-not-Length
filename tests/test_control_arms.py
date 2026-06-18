"""Tests for the three control arms and their registry wiring."""

from __future__ import annotations

from membench.retrieval import controls  # noqa: F401  (import triggers registration)
from membench.retrieval.base import ArmFamily, build_arm, get_spec
from membench.types import MemoryItem


def _corpus() -> list[MemoryItem]:
    return [MemoryItem(item_id=f"D-{i}", text="word " * 20) for i in range(10)]


class TestRegistration:
    def test_all_three_registered_as_controls(self) -> None:
        for name in ("none", "full_context", "random_context"):
            spec = get_spec(name)
            assert spec.capabilities.family is ArmFamily.CONTROL
            assert not spec.capabilities.follows_links
            assert not spec.capabilities.uses_embeddings


class TestNone:
    def test_retrieves_nothing(self) -> None:
        arm = build_arm("none")
        arm.write(_corpus())
        ctx = arm.retrieve("anything", budget_tokens=1000)
        assert ctx.text == "" and ctx.ids == ()


class TestFullContext:
    def test_packs_in_corpus_order(self) -> None:
        arm = build_arm("full_context")
        arm.write(_corpus())
        # each item is ~20 tokens ("word " = 5 chars -> ~25 tokens at cpt=4); budget
        # of 60 fits the first two in order.
        ctx = arm.retrieve("q", budget_tokens=60)
        assert ctx.ids == ("D-0", "D-1")

    def test_respects_budget(self) -> None:
        arm = build_arm("full_context")
        arm.write(_corpus())
        assert arm.retrieve("q", budget_tokens=0).ids == ()


class TestRandomContext:
    def test_is_seed_reproducible(self) -> None:
        a = build_arm("random_context", seed=42)
        b = build_arm("random_context", seed=42)
        a.write(_corpus())
        b.write(_corpus())
        assert a.retrieve("q", 80).ids == b.retrieve("q", 80).ids

    def test_ignores_query(self) -> None:
        # Two fresh same-seed arms must agree regardless of query: selection does
        # not depend on relevance. (Reusing one arm would advance its RNG.)
        a = build_arm("random_context", seed=1)
        b = build_arm("random_context", seed=1)
        a.write(_corpus())
        b.write(_corpus())
        assert a.retrieve("query one", 80).ids == b.retrieve("totally different", 80).ids

    def test_different_seeds_can_differ(self) -> None:
        a = build_arm("random_context", seed=1)
        b = build_arm("random_context", seed=2)
        corpus = _corpus()
        a.write(corpus)
        b.write(corpus)
        # not a guarantee for all seeds, but these two orders differ
        assert a.retrieve("q", 80).ids != b.retrieve("q", 80).ids
