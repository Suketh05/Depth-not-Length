"""Tests for the two-stage rerank arm and its rerankers."""

from __future__ import annotations

import pytest

from membench.retrieval import rerank  # noqa: F401  (registration side effect)
from membench.retrieval.base import ArmFamily, build_arm, get_spec
from membench.retrieval.rerank import LexicalOverlapReranker, Reranker
from membench.types import MemoryItem

_CORPUS = [
    MemoryItem("D-1", "all data exports must call withAuditLog for SOC-2 compliance"),
    MemoryItem("D-2", "use DateRangePicker for date range selection in the UI"),
    MemoryItem("D-3", "every export endpoint requires an audit log entry before streaming"),
    MemoryItem("D-4", "primary actions use the Button component variant"),
]


class TestLexicalOverlapReranker:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(LexicalOverlapReranker(), Reranker)

    def test_more_overlap_scores_higher(self) -> None:
        r = LexicalOverlapReranker()
        q = "audit log export"
        assert r.score(q, "audit log on export endpoint") > r.score(q, "button component colours")

    def test_empty_inputs_zero(self) -> None:
        assert LexicalOverlapReranker().score("", "anything") == 0.0


class TestRerankMemory:
    def _arm(self, **kw: object) -> object:
        arm = build_arm("rerank_ce", **kw)
        arm.write(_CORPUS)  # type: ignore[attr-defined]
        return arm

    def test_registered(self) -> None:
        assert get_spec("rerank_ce").capabilities.family is ArmFamily.HYBRID

    def test_promotes_jointly_relevant(self) -> None:
        ctx = self._arm().retrieve("audit log on data export", 10_000)  # type: ignore[attr-defined]
        assert set(ctx.ids[:2]) == {"D-1", "D-3"}

    def test_candidates_cap_limits_pool(self) -> None:
        arm = self._arm(candidates=1)
        ctx = arm.retrieve("audit", 10_000)  # type: ignore[attr-defined]
        assert len(ctx.ids) == 1

    def test_rejects_bad_candidates(self) -> None:
        with pytest.raises(ValueError, match="candidates"):
            build_arm("rerank_ce", candidates=0)

    def test_empty_corpus(self) -> None:
        assert build_arm("rerank_ce").retrieve("x", 1000).ids == ()

    def test_deterministic(self) -> None:
        assert self._arm().retrieve("audit", 10_000).ids == (  # type: ignore[attr-defined]
            self._arm().retrieve("audit", 10_000).ids  # type: ignore[attr-defined]
        )
