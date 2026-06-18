"""Tests for the BM25 lexical arm."""

from __future__ import annotations

from membench.retrieval import bm25  # noqa: F401  (registration side effect)
from membench.retrieval.base import ArmFamily, build_arm, get_spec
from membench.types import MemoryItem

_CORPUS = [
    MemoryItem("D-1", "all data exports must call withAuditLog for SOC-2 compliance"),
    MemoryItem("D-2", "use DateRangePicker for date range selection in the UI"),
    MemoryItem("D-3", "every export endpoint requires an audit log entry before streaming"),
    MemoryItem("D-4", "primary actions use the Button component variant"),
]


def _arm() -> object:
    arm = build_arm("bm25")
    arm.write(_CORPUS)  # type: ignore[attr-defined]
    return arm


class TestBM25:
    def test_registered_as_lexical(self) -> None:
        spec = get_spec("bm25")
        assert spec.capabilities.family is ArmFamily.LEXICAL
        assert not spec.capabilities.uses_embeddings

    def test_ranks_lexically_relevant_first(self) -> None:
        arm = _arm()
        ctx = arm.retrieve("audit log on data export", budget_tokens=10_000)  # type: ignore[attr-defined]
        # the two audit/export docs should outrank the UI docs
        assert ctx.ids[0] in {"D-1", "D-3"}
        assert set(ctx.ids[:2]) == {"D-1", "D-3"}

    def test_deterministic(self) -> None:
        a = _arm().retrieve("audit export", 10_000)  # type: ignore[attr-defined]
        b = _arm().retrieve("audit export", 10_000)  # type: ignore[attr-defined]
        assert a.ids == b.ids

    def test_empty_corpus(self) -> None:
        arm = build_arm("bm25")
        assert arm.retrieve("anything", 1000).ids == ()

    def test_empty_query_falls_back_to_corpus_order(self) -> None:
        arm = _arm()
        ctx = arm.retrieve("", budget_tokens=10_000)  # type: ignore[attr-defined]
        assert ctx.ids[0] == "D-1"  # corpus order, no crash

    def test_respects_budget(self) -> None:
        arm = _arm()
        assert arm.retrieve("audit", budget_tokens=0).ids == ()  # type: ignore[attr-defined]
