"""Tests for the content-addressed LRU cache wrapper.

The golden invariant is the textbook LRU page-replacement trace (Silberschatz,
Galvin & Gagne, *Operating System Concepts*): for capacity ``C = 2`` and the
reference string of distinct keys ``[1, 2, 1, 3, 2]`` the LRU policy produces
exactly ``1`` hit and ``4`` misses, and evicts key ``1``. That outcome is
hand-computed below and diverges from FIFO (which would give ``2`` hits), so the
assertion pins LRU semantics specifically rather than any bounded cache.
"""

from __future__ import annotations

from membench.retrieval import cache  # noqa: F401  (registration side effect)
from membench.retrieval.base import ArmFamily, build_arm, get_spec
from membench.retrieval.bm25 import BM25Memory
from membench.retrieval.cache import CacheStats, CachingMemorySystem
from membench.types import MemoryItem

_CORPUS = [
    MemoryItem("D-1", "all data exports must call withAuditLog for SOC-2 compliance"),
    MemoryItem("D-2", "use DateRangePicker for date range selection in the UI"),
    MemoryItem("D-3", "every export endpoint requires an audit log entry before streaming"),
    MemoryItem("D-4", "primary actions use the Button component variant"),
]

_QUERY = "audit log on data export"


def _cached(capacity: int = 8) -> CachingMemorySystem:
    arm = CachingMemorySystem(BM25Memory(), capacity=capacity)
    arm.write(_CORPUS)
    return arm


class TestRegistration:
    def test_registered_no_required_args(self) -> None:
        # build_arm(name) is called with no kwargs by the harness.
        arm = build_arm("bm25_cached")
        assert isinstance(arm, CachingMemorySystem)

    def test_registered_capabilities_match_default_inner(self) -> None:
        spec = get_spec("bm25_cached")
        assert spec.capabilities.family is ArmFamily.LEXICAL
        assert not spec.capabilities.uses_embeddings
        assert not spec.capabilities.follows_links


class TestSemanticsPreserved:
    def test_results_identical_to_wrapped(self) -> None:
        raw = BM25Memory()
        raw.write(_CORPUS)
        cached = _cached()
        for budget in (10, 50, 1_000):
            expected = raw.retrieve(_QUERY, budget)
            got = cached.retrieve(_QUERY, budget)
            assert got == expected  # byte-for-byte identical RetrievedContext

    def test_second_call_is_a_hit(self) -> None:
        cached = _cached()
        first = cached.retrieve(_QUERY, 100)
        second = cached.retrieve(_QUERY, 100)
        assert second == first
        assert cached.hits == 1
        assert cached.misses == 1

    def test_different_budget_is_a_different_entry(self) -> None:
        cached = _cached()
        cached.retrieve(_QUERY, 50)
        cached.retrieve(_QUERY, 60)  # same query, different budget -> distinct key
        assert cached.hits == 0
        assert cached.misses == 2
        assert cached.cache_info().size == 2


class TestLruEvictionGolden:
    def test_lru_reference_string_golden(self) -> None:
        # Textbook LRU page replacement, capacity C = 2.
        # Distinct keys come from distinct budgets b1, b2, b3 (same query).
        # Reference string [1, 2, 1, 3, 2] (LRU->MRU order shown after each step):
        #   access b1: MISS  -> [b1]        (misses=1)
        #   access b2: MISS  -> [b1, b2]    (misses=2)
        #   access b1: HIT   -> [b2, b1]    (hits=1; b1 promoted to MRU)
        #   access b3: MISS  -> evict LRU b2 -> [b1, b3]   (misses=3)
        #   access b2: MISS  -> evict LRU b1 -> [b3, b2]   (misses=4)
        # => LRU yields hits=1, misses=4, and key b1 has been EVICTED.
        #    (FIFO would evict b1 at the 4th access and HIT on b2 at the 5th,
        #     giving hits=2/misses=3 -- so this trace distinguishes LRU from FIFO.)
        cached = CachingMemorySystem(BM25Memory(), capacity=2)
        cached.write(_CORPUS)
        b1, b2, b3 = 40, 60, 80
        for budget in (b1, b2, b1, b3, b2):
            cached.retrieve(_QUERY, budget)

        assert cached.hits == 1
        assert cached.misses == 4
        assert cached.cache_info().size == 2

        # b1 was evicted, so re-accessing it must be a fresh MISS.
        misses_before = cached.misses
        cached.retrieve(_QUERY, b1)
        assert cached.misses == misses_before + 1

    def test_capacity_is_bounded(self) -> None:
        cached = CachingMemorySystem(BM25Memory(), capacity=3)
        cached.write(_CORPUS)
        for budget in range(10):
            cached.retrieve(_QUERY, budget)
        assert cached.cache_info().size == 3  # never exceeds capacity


class TestWriteInvalidatesCache:
    def test_rewrite_clears_and_readdresses(self) -> None:
        cached = _cached()
        cached.retrieve(_QUERY, 100)
        assert cached.cache_info().size == 1
        cached.write([MemoryItem("E-1", "a brand new corpus that changes the fingerprint")])
        assert cached.cache_info().size == 0  # new corpus invalidates prior entries
        cached.retrieve(_QUERY, 100)
        assert cached.misses == 2  # the same request is now a miss against new content


class TestCacheStats:
    def test_hit_rate_hand_computed(self) -> None:
        cached = _cached()
        cached.retrieve(_QUERY, 100)  # miss
        cached.retrieve(_QUERY, 100)  # hit
        cached.retrieve(_QUERY, 100)  # hit
        # hits=2, misses=1 -> hit_rate = 2 / (2 + 1) = 0.6666...
        info = cached.cache_info()
        assert info.hits == 2
        assert info.misses == 1
        assert abs(info.hit_rate - 2 / 3) < 1e-12

    def test_hit_rate_zero_when_no_access(self) -> None:
        assert CacheStats(hits=0, misses=0, size=0, capacity=8).hit_rate == 0.0
