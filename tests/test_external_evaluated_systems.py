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

from fractions import Fraction

import numpy as np
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
from membench.retrieval.mmr import mmr_order
from membench.retrieval.rank_fusion_graph import reciprocal_rank_fusion
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


# --------------------------------------------------------------------------------------
# A-Mem: an associative link recovers a note similarity ranks last and a budget drops.
# --------------------------------------------------------------------------------------


def test_amem_follows_associative_link_a_budget_would_drop() -> None:
    # A is the top query match; B links to A (via "ledger reconciliation") but shares no term
    # with the query, so pure similarity ranks it last; F matches the query weakly.
    corpus = [
        MemoryItem("A", "audit ledger reconciliation"),
        MemoryItem("B", "ledger reconciliation manifest"),
        MemoryItem("F", "auditor summary review"),
    ]
    amem = amem_adapter.AMemMemory()
    amem.write(corpus)
    dense = openviking_adapter.OpenVikingMemory(top_k=99)
    dense.write(corpus)

    # A budget that holds exactly two items: A plus one more.
    budget = 13
    amem_ids = set(amem.retrieve("audit", budget).ids)
    dense_ids = set(dense.retrieve("audit", budget).ids)

    # A-Mem spreads to B along the association and keeps it; pure similarity keeps F and
    # drops the linked-but-dissimilar B entirely.
    assert "B" in amem_ids and "B" not in dense_ids
    assert "F" in dense_ids and "F" not in amem_ids


def test_amem_no_association_when_texts_disjoint() -> None:
    corpus = [
        MemoryItem("A", "audit ledger reconciliation"),
        MemoryItem("Z", "quokka zephyr xylophone"),
    ]
    amem = amem_adapter.AMemMemory()
    amem.write(corpus)
    # Disjoint texts share no features -> zero cosine -> no associative edge is formed.
    assert amem._links == {0: [], 1: []}


# --------------------------------------------------------------------------------------
# ContextQ: pseudo-relevance feedback promotes a result-set term the raw query lacks.
# --------------------------------------------------------------------------------------


def test_contextq_expansion_promotes_result_set_vocabulary() -> None:
    corpus = [
        MemoryItem("D1", "audit export"),  # matches the raw query, contributes "export"
        MemoryItem("D2", "button palette"),  # matches nothing
        MemoryItem("D3", "export endpoint"),  # matches only the *expansion* term "export"
    ]
    expanded = contextq_adapter.ContextQMemory()
    expanded.write(corpus)
    raw = contextq_adapter.ContextQMemory(expansion_terms=0)
    raw.write(corpus)

    expanded_ids = expanded.retrieve("audit", _BIG_BUDGET).ids
    raw_ids = raw.retrieve("audit", _BIG_BUDGET).ids

    # Without expansion D3 scores 0 (no "audit"), tying D2 -> corpus order keeps D2 first.
    assert raw_ids.index("D2") < raw_ids.index("D3")
    # Expansion adds "export" from the feedback doc D1, lifting D3 above the non-matching D2.
    assert expanded_ids.index("D3") < expanded_ids.index("D2")


# --------------------------------------------------------------------------------------
# Unabyss: MMR diversification (verbatim golden + a lambda-controlled reorder).
# --------------------------------------------------------------------------------------


def test_mmr_golden_order_verbatim() -> None:
    # Golden pack D.5 (Carbonell & Goldstein 1998): relevance [0.9,0.8,0.5], the given
    # similarity matrix, lambda=0.5 -> selection order [0, 2, 1]; lambda=1.0 -> [0, 1, 2].
    relevance = np.array([0.9, 0.8, 0.5])
    similarity = np.array([[1.0, 0.8, 0.1], [0.8, 1.0, 0.2], [0.1, 0.2, 1.0]])
    assert mmr_order(relevance, similarity, 0.5) == [0, 2, 1]
    assert mmr_order(relevance, similarity, 1.0) == [0, 1, 2]


def test_unabyss_lambda_diversifies_near_duplicates() -> None:
    corpus = [
        MemoryItem("A", "audit log export"),
        MemoryItem("A2", "audit log export"),  # a near-duplicate of A
        MemoryItem("C", "audit compliance review policy"),
    ]
    relevance_only = unabyss_adapter.UnabyssMemory(lambda_=1.0)
    relevance_only.write(corpus)
    diversified = unabyss_adapter.UnabyssMemory(lambda_=0.0)
    diversified.write(corpus)

    rel_ids = relevance_only.retrieve("audit log export", _BIG_BUDGET).ids
    div_ids = diversified.retrieve("audit log export", _BIG_BUDGET).ids

    # Pure relevance keeps the duplicate adjacent to A; diversification pushes the distinct
    # item C ahead of the redundant duplicate A2.
    assert rel_ids.index("A2") < rel_ids.index("C")
    assert div_ids.index("C") < div_ids.index("A2")


# --------------------------------------------------------------------------------------
# ctx|: Reciprocal Rank Fusion (verbatim golden + agreement reward).
# --------------------------------------------------------------------------------------


def test_rrf_golden_values_verbatim() -> None:
    # Golden pack D.3 (Cormack et al. 2009, k=60): rankings [a,b,c] and [c,b].
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["c", "b"]], k=60)
    # Exact rational identities.
    assert Fraction(scores["a"]).limit_denominator(10**6) == Fraction(1, 61)
    assert Fraction(scores["b"]).limit_denominator(10**6) == Fraction(1, 31)  # 2/62
    assert Fraction(scores["c"]).limit_denominator(10**7) == Fraction(124, 3843)
    # Full-precision float64 goldens (verbatim from D.3).
    assert scores["a"] == pytest.approx(0.01639344262295082, abs=1e-15)
    assert scores["b"] == pytest.approx(0.03225806451612903, abs=1e-15)
    assert scores["c"] == pytest.approx(0.032266458495966696, abs=1e-15)
    # Fused order [c, b, a]: the exact-fraction tie-break 124/3843 > 124/3844 puts c first.
    order = sorted(scores, key=lambda d: -scores[d])
    assert order == ["c", "b", "a"]


def test_ctxpipe_rewards_agreement_between_lexical_and_dense() -> None:
    corpus = [
        MemoryItem("top", "audit audit audit log"),  # #1 on both lexical and dense
        MemoryItem("mid", "audit note"),
        MemoryItem("off", "button palette component"),
    ]
    arm = ctxpipe_adapter.CtxPipeMemory()
    arm.write(corpus)
    ids = arm.retrieve("audit log", _BIG_BUDGET).ids
    # The item both rankers place first wins the fusion.
    assert ids[0] == "top"
    assert set(ids) == {"top", "mid", "off"}


# --------------------------------------------------------------------------------------
# Driver: the lexical first-stage gate imposes a recall ceiling the re-ranker cannot beat.
# --------------------------------------------------------------------------------------


def test_driver_gate_ranks_candidate_above_more_similar_non_candidate() -> None:
    corpus = [
        MemoryItem("Cand", "audit zzqqxx yywwvv kkmmnn"),  # shares the term "audit"
        MemoryItem("NonCand", "auditaudit"),  # embedding-closer, but shares no term
    ]
    driver = driver_adapter.DriverMemory()
    driver.write(corpus)
    dense = openviking_adapter.OpenVikingMemory(top_k=99)
    dense.write(corpus)

    driver_ids = driver.retrieve("audit", _BIG_BUDGET).ids
    dense_ids = dense.retrieve("audit", _BIG_BUDGET).ids

    # Pure dense surfaces the embedding-closer non-candidate first; the lexical gate forces
    # the candidate above it regardless of similarity (the first-stage recall ceiling).
    assert dense_ids.index("NonCand") < dense_ids.index("Cand")
    assert driver_ids.index("Cand") < driver_ids.index("NonCand")


# --------------------------------------------------------------------------------------
# Oiya: a recency log returns write-order, independent of the query.
# --------------------------------------------------------------------------------------


def test_oiya_is_recency_ordered_and_query_independent() -> None:
    arm = oiya_adapter.OiyaMemory()
    arm.write(_CORPUS)
    a = arm.retrieve("audit log export", _BIG_BUDGET).ids
    b = arm.retrieve("completely unrelated button date", _BIG_BUDGET).ids
    # Most-recent-first, and identical for two different queries.
    assert a == ("D-4", "D-3", "D-2", "D-1")
    assert a == b


# --------------------------------------------------------------------------------------
# Kluris: Jaccard set-overlap prefers a compact fact over a longer bag with the same terms.
# --------------------------------------------------------------------------------------


def test_kluris_jaccard_prefers_compact_over_bulky_match() -> None:
    corpus = [
        MemoryItem("Long", "audit log plus many other unrelated words here today now"),
        MemoryItem("Short", "audit log"),
    ]
    arm = kluris_adapter.KlurisMemory()
    arm.write(corpus)
    ids = arm.retrieve("audit log", _BIG_BUDGET).ids
    # Both contain every query term (tf ties), but the union penalty ranks the compact item
    # first -- the entity/keyword-store behaviour.
    assert ids[0] == "Short"
    assert ids.index("Short") < ids.index("Long")


def test_kluris_overlap_math() -> None:
    q = {"audit", "log"}
    assert kluris_adapter.jaccard_overlap(q, {"audit", "log"}) == 1.0  # identical sets
    assert kluris_adapter.jaccard_overlap(q, {"audit", "log", "x", "y"}) == Fraction(2, 4)
    assert kluris_adapter.jaccard_overlap(q, {"button"}) == 0.0  # disjoint
    assert kluris_adapter.jaccard_overlap(set(), set()) == 0.0  # both empty -> chance floor


# --------------------------------------------------------------------------------------
# Oracle Summary: the store compresses each memory, so context is strictly shorter.
# --------------------------------------------------------------------------------------


def test_oracle_summary_compresses_context() -> None:
    corpus = [
        MemoryItem(
            "D-1",
            "audit logging is mandatory on every export. this rationale explains why it "
            "matters for compliance and downstream security reviews at length.",
        ),
        MemoryItem(
            "D-2",
            "the date range picker owns date selection. a long trailing justification about "
            "consistency and accessibility follows here for completeness.",
        ),
    ]
    arm = oracle_summary_adapter.OracleSummaryMemory()
    arm.write(corpus)
    ctx = arm.retrieve("audit logging export", _BIG_BUDGET)

    raw_len = sum(len(item.text) for item in corpus)
    # The assembled summaries are strictly shorter than the concatenated raw memories.
    assert len(ctx.text) < raw_len
    # Recall is still well-defined: the governing item's id is preserved.
    assert "D-1" in ctx.ids


def test_extractive_summary_takes_capped_first_sentence() -> None:
    assert oracle_summary_adapter.extractive_summary("first sentence. second one.") == (
        "first sentence."
    )
    assert oracle_summary_adapter.extractive_summary("") == ""
    long_lead = "x" * 500
    assert oracle_summary_adapter.extractive_summary(long_lead, max_chars=10) == "x" * 10


# --------------------------------------------------------------------------------------
# OpenViking: a bare vector store returns a fixed page and has no lexical gate.
# --------------------------------------------------------------------------------------


def test_openviking_returns_fixed_top_k_page() -> None:
    arm = openviking_adapter.OpenVikingMemory(top_k=2)
    arm.write(_CORPUS)
    ids = arm.retrieve("audit log export compliance", _BIG_BUDGET).ids
    # A managed store returns its fixed page even when the budget could hold the whole corpus.
    assert len(ids) == 2


def test_openviking_rejects_non_positive_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        openviking_adapter.OpenVikingMemory(top_k=0)
