"""Tests for the RAPTOR hierarchical arm."""

from __future__ import annotations

from membench.retrieval import raptor  # noqa: F401  (registration side effect)
from membench.retrieval.base import ArmFamily, build_arm, get_spec
from membench.types import MemoryItem

_CORPUS = [
    MemoryItem("D-1", "all data exports must call withAuditLog for SOC-2 compliance"),
    MemoryItem("D-2", "use DateRangePicker for date range selection in the UI"),
    MemoryItem("D-3", "every export endpoint requires an audit log entry before streaming"),
    MemoryItem("D-4", "primary actions use the Button component variant"),
    MemoryItem("D-5", "tooltips use the Popover primitive for accessibility"),
    MemoryItem("D-6", "exported files must be streamed, never buffered fully in memory"),
]


def _arm(**kw: object) -> object:
    arm = build_arm("raptor", **kw)
    arm.write(_CORPUS)  # type: ignore[attr-defined]
    return arm


class TestRaptor:
    def test_registered(self) -> None:
        spec = get_spec("raptor")
        assert spec.capabilities.family is ArmFamily.DENSE
        assert spec.capabilities.uses_embeddings

    def test_returns_all_items_with_no_duplicates(self) -> None:
        ctx = _arm().retrieve("audit", 10_000)  # type: ignore[attr-defined]
        assert sorted(ctx.ids) == [f"D-{i}" for i in range(1, 7)]
        assert len(ctx.ids) == len(set(ctx.ids))

    def test_ranks_relevant_cluster_first(self) -> None:
        ctx = _arm().retrieve("audit logging on data export", 10_000)  # type: ignore[attr-defined]
        assert ctx.ids[0] in {"D-1", "D-3", "D-6"}

    def test_single_item_corpus(self) -> None:
        arm = build_arm("raptor")
        arm.write([MemoryItem("only", "a single document")])  # type: ignore[attr-defined]
        assert arm.retrieve("anything", 10_000).ids == ("only",)  # type: ignore[attr-defined]

    def test_explicit_n_clusters(self) -> None:
        ctx = _arm(n_clusters=2).retrieve("button UI", 10_000)  # type: ignore[attr-defined]
        assert len(ctx.ids) == 6

    def test_deterministic(self) -> None:
        assert _arm().retrieve("export", 10_000).ids == (  # type: ignore[attr-defined]
            _arm().retrieve("export", 10_000).ids  # type: ignore[attr-defined]
        )

    def test_empty_corpus(self) -> None:
        assert build_arm("raptor").retrieve("x", 1000).ids == ()
