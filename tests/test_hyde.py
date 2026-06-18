"""Tests for the HyDE arm and its expanders."""

from __future__ import annotations

from membench.retrieval import hyde  # noqa: F401  (registration side effect)
from membench.retrieval.base import ArmFamily, build_arm, get_spec
from membench.retrieval.hyde import CallableExpander, QueryExpander, TemplateExpander
from membench.types import MemoryItem

_CORPUS = [
    MemoryItem("D-1", "all data exports must call withAuditLog for SOC-2 compliance"),
    MemoryItem("D-2", "use DateRangePicker for date range selection in the UI"),
    MemoryItem("D-3", "every export endpoint requires an audit log entry before streaming"),
    MemoryItem("D-4", "primary actions use the Button component variant"),
]


class TestExpanders:
    def test_template_expander_satisfies_protocol_and_grows_query(self) -> None:
        exp = TemplateExpander()
        assert isinstance(exp, QueryExpander)
        out = exp.expand("add a CSV export")
        assert "add a CSV export" in out and len(out) > len("add a CSV export")

    def test_callable_expander(self) -> None:
        exp = CallableExpander(lambda q: q.upper())
        assert exp.expand("hi") == "HI"
        assert isinstance(exp, QueryExpander)


class TestHyDEMemory:
    def _arm(self, **kw: object) -> object:
        arm = build_arm("hyde", **kw)
        arm.write(_CORPUS)  # type: ignore[attr-defined]
        return arm

    def test_registered_as_dense(self) -> None:
        spec = get_spec("hyde")
        assert spec.capabilities.family is ArmFamily.DENSE
        assert spec.capabilities.uses_embeddings

    def test_retrieves_relevant(self) -> None:
        ctx = self._arm().retrieve("add a CSV export with an audit trail", 10_000)  # type: ignore[attr-defined]
        assert set(ctx.ids[:2]) == {"D-1", "D-3"}

    def test_custom_expander_changes_ranking(self) -> None:
        # An expander that injects UI vocabulary should pull the UI doc up.
        arm = self._arm(expander=CallableExpander(lambda q: f"{q} DateRangePicker date range UI"))
        ctx = arm.retrieve("pick something", 10_000)  # type: ignore[attr-defined]
        assert ctx.ids[0] == "D-2"

    def test_deterministic(self) -> None:
        assert self._arm().retrieve("audit export", 10_000).ids == (  # type: ignore[attr-defined]
            self._arm().retrieve("audit export", 10_000).ids  # type: ignore[attr-defined]
        )

    def test_empty_corpus(self) -> None:
        assert build_arm("hyde").retrieve("x", 1000).ids == ()
