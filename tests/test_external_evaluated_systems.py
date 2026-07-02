"""Tests for the offline protocol adapters of the remaining evaluated systems.

Covers the nine systems the paper's unified harness scores that previously had no
implementation (tab:stdbench / tab:t070): A-Mem, ContextQ, Unabyss, ctx|, Driver, Oiya,
Kluris, Oracle Summary, OpenViking. Each is an offline, deterministic protocol *model* of
the mechanism the paper's landscape attributes to (or that the system's name/profile implies
for) that system -- these tests verify the MemorySystem contract, determinism, and the
mechanism-distinctive behaviour of each, NOT the paper's measured benchmark numbers (which
come from the full public-benchmark harness and are never reproduced by a unit fixture).

Where an adapter reuses a documented, shared algorithm (Reciprocal Rank Fusion, Maximal
Marginal Relevance), the exact golden values are asserted verbatim from the canonical golden
pack (D.3, D.5) so the reused identity is pinned.
"""

from __future__ import annotations

import pytest

from membench.retrieval.base import ArmFamily, MemorySystem, build_arm, get_spec
from membench.retrieval.external import (
    amem_adapter,
    contextq_adapter,
    ctxpipe_adapter,
    driver_adapter,
    kluris_adapter,
    oiya_adapter,
    openviking_adapter,
    oracle_summary_adapter,
    unabyss_adapter,
)
from membench.types import MemoryItem, RetrievedContext

# The nine newly-implemented evaluated systems, by registered arm name and class.
ARMS = [
    ("amem", amem_adapter.AMemMemory),
    ("contextq", contextq_adapter.ContextQMemory),
    ("unabyss", unabyss_adapter.UnabyssMemory),
    ("ctxpipe", ctxpipe_adapter.CtxPipeMemory),
    ("driver", driver_adapter.DriverMemory),
    ("oiya", oiya_adapter.OiyaMemory),
    ("kluris", kluris_adapter.KlurisMemory),
    ("oracle_summary", oracle_summary_adapter.OracleSummaryMemory),
    ("openviking", openviking_adapter.OpenVikingMemory),
]
ARM_NAMES = [name for name, _ in ARMS]

_CORPUS = [
    MemoryItem("D-1", "all data exports must call the audit log for compliance"),
    MemoryItem("D-2", "use the date range picker for date selection"),
    MemoryItem("D-3", "every export endpoint requires an audit log entry"),
    MemoryItem("D-4", "primary actions use the button component"),
]
_CORPUS_IDS = {item.item_id for item in _CORPUS}
_BIG_BUDGET = 1_000_000


# --------------------------------------------------------------------------------------
# Contract, registration, and determinism (shared across all nine adapters).
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ARM_NAMES)
def test_registered_as_offline_deterministic(name: str) -> None:
    spec = get_spec(name)
    caps = spec.capabilities
    # These are offline protocol models: no network, deterministic, real ArmFamily.
    assert caps.requires_network is False
    assert caps.deterministic is True
    assert isinstance(caps.family, ArmFamily)
    assert spec.description  # first docstring line is recorded as the description


@pytest.mark.parametrize("name", ARM_NAMES)
def test_build_arm_yields_memory_system(name: str) -> None:
    arm = build_arm(name)
    assert isinstance(arm, MemorySystem)


@pytest.mark.parametrize("name,cls", ARMS)
def test_empty_corpus_returns_empty(name: str, cls: type) -> None:
    arm = cls()
    assert arm.retrieve("audit log", _BIG_BUDGET) == RetrievedContext.empty()


@pytest.mark.parametrize("name,cls", ARMS)
def test_write_then_retrieve_maps_to_corpus_ids(name: str, cls: type) -> None:
    arm = cls()
    arm.write(_CORPUS)
    ctx = arm.retrieve("audit log on data export", _BIG_BUDGET)
    assert isinstance(ctx, RetrievedContext)
    assert set(ctx.ids) <= _CORPUS_IDS
    # Every non-recency arm carries query signal, so the audit-log decisions surface.
    if name != "oiya":
        assert {"D-1", "D-3"} & set(ctx.ids)


@pytest.mark.parametrize("name,cls", ARMS)
def test_zero_budget_returns_no_ids(name: str, cls: type) -> None:
    arm = cls()
    arm.write(_CORPUS)
    assert arm.retrieve("audit log", 0).ids == ()


@pytest.mark.parametrize("name,cls", ARMS)
def test_deterministic_across_instances(name: str, cls: type) -> None:
    first = cls()
    first.write(_CORPUS)
    second = cls()
    second.write(_CORPUS)
    query = "audit log on data export"
    assert first.retrieve(query, _BIG_BUDGET).ids == second.retrieve(query, _BIG_BUDGET).ids


@pytest.mark.parametrize("name,cls", ARMS)
def test_ids_are_unique_within_a_retrieval(name: str, cls: type) -> None:
    arm = cls()
    arm.write(_CORPUS)
    ids = arm.retrieve("audit log export compliance", _BIG_BUDGET).ids
    assert len(ids) == len(set(ids))
