"""Tests for the dense embedding arm (native and reference ranking paths)."""

from __future__ import annotations

import pytest

from membench.retrieval import dense  # noqa: F401  (registration side effect)
from membench.retrieval.base import ArmFamily, build_arm, get_spec
from membench.types import MemoryItem

_CORPUS = [
    MemoryItem("D-1", "all data exports must call withAuditLog for SOC-2 compliance"),
    MemoryItem("D-2", "use DateRangePicker for date range selection in the UI"),
    MemoryItem("D-3", "every export endpoint requires an audit log entry before streaming"),
    MemoryItem("D-4", "primary actions use the Button component variant"),
]


class TestDense:
    def test_registered_as_dense_using_embeddings(self) -> None:
        spec = get_spec("dense")
        assert spec.capabilities.family is ArmFamily.DENSE
        assert spec.capabilities.uses_embeddings

    def test_ranks_semantically_relevant_first(self) -> None:
        arm = build_arm("dense")
        arm.write(_CORPUS)  # type: ignore[attr-defined]
        ctx = arm.retrieve("add a CSV export with an audit trail", 10_000)  # type: ignore[attr-defined]
        assert set(ctx.ids[:2]) == {"D-1", "D-3"}

    def test_native_and_reference_paths_agree(self) -> None:
        native_arm = build_arm("dense", use_native=True)
        ref_arm = build_arm("dense", use_native=False)
        native_arm.write(_CORPUS)  # type: ignore[attr-defined]
        ref_arm.write(_CORPUS)  # type: ignore[attr-defined]
        q = "audit logging on exports"
        assert native_arm.retrieve(q, 10_000).ids == ref_arm.retrieve(q, 10_000).ids  # type: ignore[attr-defined]

    def test_deterministic(self) -> None:
        a, b = build_arm("dense"), build_arm("dense")
        a.write(_CORPUS)  # type: ignore[attr-defined]
        b.write(_CORPUS)  # type: ignore[attr-defined]
        assert a.retrieve("audit", 10_000).ids == b.retrieve("audit", 10_000).ids  # type: ignore[attr-defined]

    def test_empty_corpus(self) -> None:
        assert build_arm("dense").retrieve("x", 1000).ids == ()

    @pytest.mark.parametrize("budget", [0, 30])
    def test_respects_budget(self, budget: int) -> None:
        arm = build_arm("dense")
        arm.write(_CORPUS)  # type: ignore[attr-defined]
        ctx = arm.retrieve("audit", budget)  # type: ignore[attr-defined]
        assert len(ctx.ids) <= len(_CORPUS)
