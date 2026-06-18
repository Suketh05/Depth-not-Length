"""Tests for the Supermemory adapter (logic via the shared fake client)."""

from __future__ import annotations

import pytest

from _external_fakes import LexicalIdSearchClient
from membench.retrieval.base import ArmFamily, build_arm, get_spec
from membench.retrieval.external import supermemory_adapter  # noqa: F401  (registration)
from membench.types import MemoryItem

_CORPUS = [
    MemoryItem("D-1", "all data exports must call the audit log for compliance"),
    MemoryItem("D-2", "use the date range picker for date selection"),
    MemoryItem("D-3", "every export endpoint requires an audit log entry"),
    MemoryItem("D-4", "primary actions use the button component"),
]


def test_registration() -> None:
    spec = get_spec("supermemory")
    assert spec.capabilities.family is ArmFamily.EXTERNAL
    assert spec.capabilities.requires_network and not spec.capabilities.deterministic


def test_maps_results_back_to_item_ids() -> None:
    arm = build_arm("supermemory", client=LexicalIdSearchClient())
    arm.write(_CORPUS)  # type: ignore[attr-defined]
    ctx = arm.retrieve("audit log on data export", 10_000)  # type: ignore[attr-defined]
    assert set(ctx.ids[:2]) == {"D-1", "D-3"}


def test_respects_budget() -> None:
    arm = build_arm("supermemory", client=LexicalIdSearchClient())
    arm.write(_CORPUS)  # type: ignore[attr-defined]
    assert arm.retrieve("audit", 0).ids == ()  # type: ignore[attr-defined]


def test_actionable_error_without_sdk() -> None:
    arm = build_arm("supermemory")  # client=None -> import supermemory (absent)
    with pytest.raises(RuntimeError, match="supermemory"):
        arm.write(_CORPUS)  # type: ignore[attr-defined]
