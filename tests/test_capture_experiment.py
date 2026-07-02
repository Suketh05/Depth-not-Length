"""Tests for the capture experiment (paper sec:capture / tab:capexp).

Covers the three storage-condition transforms (determinism, surgical link
strip, shred text conservation / count reversibility), the load-bearing
*mechanism* on a hand-built 2-hop corpus (an intact typed store recovers the
governing rule through its links; the edge-stripped store loses multi-hop
reachability; a lexical arm is unaffected by the strip), and the driver's
paired, well-formed rows plus the tab:capexp-shaped aggregation.

No published tab:capexp cell value is asserted against this code's output:
the mechanism tests assert reachability events on corpora built by hand, and
the only literal goldens are hand-computed (or quoted from the canonical
golden pack with provenance).
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from membench.datasets.capture_conditions import (
    TYPED_LINK_KEYS,
    CaptureCondition,
    apply_condition,
    strip_edges,
)
from membench.datasets.synthetic import load_synthetic_tasks
from membench.types import MemoryItem, Task


def _synthetic_task(depth: int = 2, seed: int = 0) -> Task:
    return load_synthetic_tasks(depths=(depth,), per_depth=1, seed=seed)[0]


def _shredded_fragments(task: Task) -> dict[str, list[MemoryItem]]:
    """Group a Raw-scattered corpus's fragments by the item they were shredded from."""
    groups: dict[str, list[MemoryItem]] = defaultdict(list)
    for item in task.memory_corpus:
        if item.metadata.get("node_type") == "fragment":
            groups[str(item.metadata["shredded_from"])].append(item)
    for fragments in groups.values():
        fragments.sort(key=lambda i: int(i.metadata["fragment_index"]))
    return groups


class TestStripEdges:
    def test_removes_exactly_the_typed_link_keys(self) -> None:
        task = _synthetic_task(depth=3)
        # the CAPTURED corpus really carries links (else the strip is vacuous)
        assert any("edges" in item.metadata for item in task.memory_corpus)
        stripped = strip_edges(task)
        for item in stripped.memory_corpus:
            for key in TYPED_LINK_KEYS:
                assert key not in item.metadata

    def test_preserves_ids_texts_order_and_other_metadata(self) -> None:
        task = _synthetic_task(depth=3)
        stripped = strip_edges(task)
        assert len(stripped.memory_corpus) == len(task.memory_corpus)
        for before, after in zip(task.memory_corpus, stripped.memory_corpus, strict=True):
            assert after.item_id == before.item_id
            assert after.text == before.text  # no text lost
            expected = {k: v for k, v in before.metadata.items() if k not in TYPED_LINK_KEYS}
            assert dict(after.metadata) == expected  # node_type etc. survive

    def test_preserves_every_other_task_field(self) -> None:
        task = _synthetic_task(depth=2)
        stripped = strip_edges(task)
        assert stripped.task_id == task.task_id
        assert stripped.query == task.query
        assert stripped.depth == task.depth
        assert stripped.governing_decisions == task.governing_decisions
        assert stripped.dataset == task.dataset

    def test_deterministic_and_idempotent(self) -> None:
        task = _synthetic_task(depth=3)
        once = strip_edges(task)
        assert once == strip_edges(task)  # pure function
        assert strip_edges(once) == once  # fixed point

    def test_captured_condition_is_identity(self) -> None:
        task = _synthetic_task(depth=2)
        assert apply_condition(task, CaptureCondition.CAPTURED) is task
        # string values dispatch too (the enum is string-valued for row serialisation)
        assert apply_condition(task, "captured") is task
        assert apply_condition(task, "discrete_no_links") == strip_edges(task)

    def test_unknown_condition_rejected(self) -> None:
        task = _synthetic_task(depth=1)
        with pytest.raises(ValueError):
            apply_condition(task, "shredded")  # not one of the three conditions
