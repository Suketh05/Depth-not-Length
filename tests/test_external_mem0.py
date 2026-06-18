"""Tests for the Mem0 adapter (logic via an injected fake client)."""

from __future__ import annotations

import re

import pytest

from membench.retrieval import external  # noqa: F401
from membench.retrieval.base import ArmFamily, build_arm, get_spec
from membench.retrieval.external import mem0_adapter  # noqa: F401  (registration)
from membench.types import MemoryItem

_TOK = re.compile(r"[a-z0-9]+")


class FakeMem0:
    """Minimal stand-in for mem0.Memory: lexical add/search keyed by metadata."""

    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def add(self, text: str, user_id: str, metadata: dict[str, object]) -> None:
        self.rows.append({"memory": text, "metadata": metadata})

    def search(self, query: str, user_id: str, limit: int) -> dict[str, object]:
        q = set(_TOK.findall(query.lower()))

        def overlap(row: dict[str, object]) -> int:
            return len(q & set(_TOK.findall(str(row["memory"]).lower())))

        ranked = sorted(self.rows, key=overlap, reverse=True)
        return {"results": ranked[:limit]}


_CORPUS = [
    MemoryItem("D-1", "all data exports must call the audit log for compliance"),
    MemoryItem("D-2", "use the date range picker for date selection"),
    MemoryItem("D-3", "every export endpoint requires an audit log entry"),
    MemoryItem("D-4", "primary actions use the button component"),
]


class TestRegistration:
    def test_external_flags(self) -> None:
        spec = get_spec("mem0")
        assert spec.capabilities.family is ArmFamily.EXTERNAL
        assert spec.capabilities.requires_network is True
        assert spec.capabilities.deterministic is False


class TestMem0Logic:
    def _arm(self) -> object:
        arm = build_arm("mem0", client=FakeMem0())
        arm.write(_CORPUS)  # type: ignore[attr-defined]
        return arm

    def test_maps_results_back_to_item_ids(self) -> None:
        ctx = self._arm().retrieve("audit log on data export", 10_000)  # type: ignore[attr-defined]
        assert set(ctx.ids[:2]) == {"D-1", "D-3"}

    def test_respects_budget(self) -> None:
        assert self._arm().retrieve("audit", 0).ids == ()  # type: ignore[attr-defined]

    def test_empty_corpus(self) -> None:
        assert build_arm("mem0", client=FakeMem0()).retrieve("x", 1000).ids == ()


class TestMissingPackage:
    def test_actionable_error_without_mem0ai(self) -> None:
        arm = build_arm("mem0")  # client=None -> tries to import mem0 (absent)
        with pytest.raises(RuntimeError, match="mem0ai"):
            arm.write(_CORPUS)  # type: ignore[attr-defined]
