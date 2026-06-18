"""Tests for the MemorySystem contract, budget packing, and the arm registry."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from membench.retrieval.base import (
    ArmFamily,
    ArmRegistry,
    MemorySystem,
    pack_to_budget,
)
from membench.types import MemoryItem, RetrievedContext


class _Echo(MemorySystem):
    """A trivial arm that returns its corpus in order."""

    def __init__(self, label: str = "echo") -> None:
        self.label = label
        self._items: list[MemoryItem] = []

    def write(self, items: Iterable[MemoryItem]) -> None:
        self._items.extend(items)

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        return pack_to_budget(((i.item_id, i.text) for i in self._items), budget_tokens)


class TestPackToBudget:
    def test_truncates_at_first_overflow(self) -> None:
        # Each text is 40 chars -> 10 tokens at cpt=4. Budget 25 fits exactly two.
        ranked = [("a", "x" * 40), ("b", "y" * 40), ("c", "z" * 40)]
        ctx = pack_to_budget(ranked, budget_tokens=25)
        assert ctx.ids == ("a", "b")
        assert ctx.text == ("x" * 40) + "\n\n" + ("y" * 40)

    def test_does_not_backfill_smaller_later_item(self) -> None:
        # First item overflows; packing stops even though 'b' would fit.
        ranked = [("a", "x" * 400), ("b", "y" * 4)]
        ctx = pack_to_budget(ranked, budget_tokens=10)
        assert ctx.ids == ()

    def test_zero_budget_yields_empty(self) -> None:
        assert pack_to_budget([("a", "xxxx")], 0).ids == ()

    def test_negative_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            pack_to_budget([], -1)

    def test_bad_chars_per_token_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            pack_to_budget([("a", "x")], 10, chars_per_token=0)


class TestArmRegistry:
    def test_register_build_and_capabilities(self) -> None:
        reg = ArmRegistry()

        @reg.register("echo", family=ArmFamily.CONTROL, follows_links=False, uses_embeddings=False)
        class _Registered(_Echo):
            """An echo arm for testing."""

        spec = reg.get("echo")
        assert spec.name == "echo"
        assert spec.capabilities.family is ArmFamily.CONTROL
        assert spec.description == "An echo arm for testing."
        built = reg.build("echo", label="custom")
        assert isinstance(built, MemorySystem)
        assert built.label == "custom"  # type: ignore[attr-defined]

    def test_duplicate_registration_raises(self) -> None:
        reg = ArmRegistry()
        reg.register("dup", family=ArmFamily.CONTROL, follows_links=False, uses_embeddings=False)(
            _Echo
        )
        with pytest.raises(ValueError, match="already registered"):
            reg.register(
                "dup", family=ArmFamily.CONTROL, follows_links=False, uses_embeddings=False
            )(_Echo)

    def test_unknown_arm_lists_known(self) -> None:
        reg = ArmRegistry()
        reg.register("known", family=ArmFamily.CONTROL, follows_links=False, uses_embeddings=False)(
            _Echo
        )
        with pytest.raises(KeyError, match="known"):
            reg.get("missing")

    def test_names_sorted(self) -> None:
        reg = ArmRegistry()
        for n in ("zeta", "alpha"):
            reg.register(n, family=ArmFamily.CONTROL, follows_links=False, uses_embeddings=False)(
                _Echo
            )
        assert reg.names() == ["alpha", "zeta"]
