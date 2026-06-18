"""Tests for the TF-IDF cosine arm."""

from __future__ import annotations

from membench.retrieval import tfidf  # noqa: F401  (registration side effect)
from membench.retrieval.base import ArmFamily, build_arm, get_spec
from membench.types import MemoryItem

_CORPUS = [
    MemoryItem("D-1", "all data exports must call the audit log for compliance"),
    MemoryItem("D-2", "use the date range picker for selecting dates in the interface"),
    MemoryItem("D-3", "every export endpoint requires an audit log entry"),
    MemoryItem("D-4", "primary actions use the button component"),
]


def _arm() -> object:
    arm = build_arm("tfidf")
    arm.write(_CORPUS)  # type: ignore[attr-defined]
    return arm


class TestTfidf:
    def test_registered_as_lexical(self) -> None:
        assert get_spec("tfidf").capabilities.family is ArmFamily.LEXICAL

    def test_ranks_relevant_first(self) -> None:
        ctx = _arm().retrieve("audit log for data export", 10_000)  # type: ignore[attr-defined]
        assert set(ctx.ids[:2]) == {"D-1", "D-3"}

    def test_deterministic(self) -> None:
        assert _arm().retrieve("audit export", 10_000).ids == (  # type: ignore[attr-defined]
            _arm().retrieve("audit export", 10_000).ids  # type: ignore[attr-defined]
        )

    def test_empty_corpus(self) -> None:
        assert build_arm("tfidf").retrieve("x", 1000).ids == ()

    def test_out_of_vocabulary_query_is_safe(self) -> None:
        # query shares no terms with corpus -> all-zero scores -> corpus order, no crash
        ctx = _arm().retrieve("zzzqqq nonexistent", 10_000)  # type: ignore[attr-defined]
        assert ctx.ids[0] == "D-1"

    def test_respects_budget(self) -> None:
        assert _arm().retrieve("audit", 0).ids == ()  # type: ignore[attr-defined]
