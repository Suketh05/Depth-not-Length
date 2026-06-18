"""Tests for the letta adapter (logic via the shared fake client)."""

from __future__ import annotations

import pytest

from _external_fakes import LexicalIdSearchClient
from membench.retrieval.base import ArmFamily, build_arm, get_spec
from membench.retrieval.external import letta_adapter  # noqa: F401  (registration)
from membench.types import MemoryItem

_CORPUS = [
    MemoryItem("D-1", "all data exports must call the audit log for compliance"),
    MemoryItem("D-2", "use the date range picker for date selection"),
    MemoryItem("D-3", "every export endpoint requires an audit log entry"),
]


def test_registration() -> None:
    spec = get_spec("letta")
    assert spec.capabilities.family is ArmFamily.EXTERNAL
    assert spec.capabilities.follows_links is False


def test_maps_results_back_to_item_ids() -> None:
    arm = build_arm("letta", client=LexicalIdSearchClient())
    arm.write(_CORPUS)  # type: ignore[attr-defined]
    ctx = arm.retrieve("audit log on data export", 10_000)  # type: ignore[attr-defined]
    assert set(ctx.ids[:2]) == {"D-1", "D-3"}


def test_actionable_error_without_package() -> None:
    arm = build_arm("letta")
    with pytest.raises(RuntimeError, match="letta-client"):
        arm.write(_CORPUS)  # type: ignore[attr-defined]
