"""Tests for the hybrid RRF arm."""

from __future__ import annotations

import pytest

from membench.retrieval import hybrid  # noqa: F401  (registration side effect)
from membench.retrieval.base import ArmFamily, build_arm, get_spec
from membench.types import MemoryItem

_CORPUS = [
    MemoryItem("D-1", "all data exports must call withAuditLog for SOC-2 compliance"),
    MemoryItem("D-2", "use DateRangePicker for date range selection in the UI"),
    MemoryItem("D-3", "every export endpoint requires an audit log entry before streaming"),
    MemoryItem("D-4", "primary actions use the Button component variant"),
]


def _arm(**kw: object) -> object:
    arm = build_arm("hybrid_rrf", **kw)
    arm.write(_CORPUS)  # type: ignore[attr-defined]
    return arm


class TestHybridRRF:
    def test_registered_as_hybrid(self) -> None:
        spec = get_spec("hybrid_rrf")
        assert spec.capabilities.family is ArmFamily.HYBRID
        assert spec.capabilities.uses_embeddings

    def test_fuses_to_relevant_items(self) -> None:
        ctx = _arm().retrieve("audit log on data export", 10_000)  # type: ignore[attr-defined]
        assert set(ctx.ids[:2]) == {"D-1", "D-3"}

    def test_ranks_all_items(self) -> None:
        ctx = _arm().retrieve("audit", 10_000)  # type: ignore[attr-defined]
        assert set(ctx.ids) == {"D-1", "D-2", "D-3", "D-4"}

    def test_deterministic(self) -> None:
        assert _arm().retrieve("audit export", 10_000).ids == (  # type: ignore[attr-defined]
            _arm().retrieve("audit export", 10_000).ids  # type: ignore[attr-defined]
        )

    def test_rejects_bad_k(self) -> None:
        with pytest.raises(ValueError, match="RRF k"):
            build_arm("hybrid_rrf", k=0)

    def test_empty_corpus(self) -> None:
        assert build_arm("hybrid_rrf").retrieve("x", 1000).ids == ()

    def test_respects_budget(self) -> None:
        assert _arm().retrieve("audit", 0).ids == ()  # type: ignore[attr-defined]
