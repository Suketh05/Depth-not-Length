"""Content-addressed LRU cache wrapper for a :class:`MemorySystem`.

This module provides :class:`CachingMemorySystem`, a transparent decorator that
memoises ``retrieve`` results so an arm that is queried repeatedly with the same
``(query, budget)`` against the same corpus does not recompute its (possibly
expensive) ranking. It is a *performance* primitive only: every value it returns
is byte-for-byte the value the wrapped arm would have returned, so it never alters
retrieval semantics, ranking, or scoring.

Algorithm
---------
Two standard ideas are composed:

* **Content addressing.** The cache key is a cryptographic digest of the request
  *content* — ``key = H(fingerprint(corpus) || budget_tokens || query)`` — rather
  than an opaque counter. The corpus ``fingerprint`` is itself
  ``H(item_id_0 || text_0 || item_id_1 || text_1 || ...)`` over the written
  items. Equal content therefore maps to one entry and a changed corpus maps to a
  fresh address, the defining property of content-addressable storage
  (Merkle, R. C. *A Digital Signature Based on a Conventional Encryption
  Function*, CRYPTO '87). We use SHA-256 (NIST FIPS 180-4).
* **Least-Recently-Used eviction.** The cache is bounded to ``capacity`` entries.
  On every access the touched key is promoted to most-recently-used; when an
  insertion would exceed ``capacity`` the least-recently-used key is evicted.
  This is the classic LRU page-replacement policy (Silberschatz, Galvin & Gagne,
  *Operating System Concepts*, ch. "Virtual Memory"; Tanenbaum, *Modern
  Operating Systems*). An :class:`collections.OrderedDict` gives O(1) lookup,
  promotion (``move_to_end``) and eviction (``popitem(last=False)``); we do not
  use :func:`functools.lru_cache` because we need a custom content-addressed key,
  explicit corpus invalidation, and inspectable hit/miss counters.

The wrapper also clears the cache on :meth:`write`, since a new corpus invalidates
every prior ranking; the corpus fingerprint in the key is the defence-in-depth
that keeps the cache correct even if a caller reuses an instance across corpora.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass

from membench.retrieval.base import ArmFamily, MemorySystem, register_arm
from membench.retrieval.bm25 import BM25Memory
from membench.types import MemoryItem, RetrievedContext

__all__ = ["DEFAULT_CACHE_CAPACITY", "CacheStats", "CachingMemorySystem"]

DEFAULT_CACHE_CAPACITY = 128
"""Default bound on the number of distinct ``(corpus, budget, query)`` entries."""


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Immutable snapshot of a cache's runtime counters.

    Parameters
    ----------
    hits
        Number of ``retrieve`` calls served from the cache.
    misses
        Number of ``retrieve`` calls that fell through to the wrapped arm.
    size
        Number of entries currently resident.
    capacity
        Maximum number of resident entries before LRU eviction.
    """

    hits: int
    misses: int
    size: int
    capacity: int

    @property
    def hit_rate(self) -> float:
        """Fraction of accesses served from cache, or ``0.0`` if there were none."""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


@register_arm(
    "bm25_cached",
    family=ArmFamily.LEXICAL,
    follows_links=False,
    uses_embeddings=False,
)
class CachingMemorySystem(MemorySystem):
    """Memoise a wrapped arm's ``retrieve`` with a content-addressed bounded LRU.

    The default wrapped arm is :class:`~membench.retrieval.bm25.BM25Memory`, so the
    arm is constructible with no arguments (as the harness requires). Results are
    identical to the wrapped arm; only repeated identical requests are faster.

    Parameters
    ----------
    inner
        The arm to wrap. If ``None`` a fresh :class:`BM25Memory` is created.
    capacity
        Maximum number of cached ``(corpus, budget, query)`` results before the
        least-recently-used entry is evicted. Must be positive.

    Attributes
    ----------
    hits
        Count of requests served from the cache.
    misses
        Count of requests forwarded to the wrapped arm.
    """

    def __init__(
        self,
        inner: MemorySystem | None = None,
        *,
        capacity: int = DEFAULT_CACHE_CAPACITY,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._inner: MemorySystem = inner if inner is not None else BM25Memory()
        self._capacity = capacity
        self._cache: OrderedDict[str, RetrievedContext] = OrderedDict()
        self._fingerprint = self._corpus_fingerprint([])
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _corpus_fingerprint(items: list[MemoryItem]) -> str:
        """Return a SHA-256 digest over the ordered ``(item_id, text)`` content.

        Two writes with identical item content produce the same fingerprint; any
        change to an id or text yields a different one (up to SHA-256 collision
        resistance), which is what makes the cache key content-addressed. Metadata
        is intentionally excluded: the wrapped lexical/dense arms rank on ``text``
        alone, so it does not influence the memoised result.
        """
        digest = hashlib.sha256()
        for item in items:
            digest.update(item.item_id.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(item.text.encode("utf-8"))
            digest.update(b"\x01")
        return digest.hexdigest()

    def _key(self, query: str, budget_tokens: int) -> str:
        """Content-address a request as ``H(fingerprint || budget || query)``."""
        digest = hashlib.sha256()
        digest.update(self._fingerprint.encode("utf-8"))
        digest.update(b"|")
        digest.update(str(budget_tokens).encode("utf-8"))
        digest.update(b"|")
        digest.update(query.encode("utf-8"))
        return digest.hexdigest()

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Forward the corpus to the wrapped arm and re-address the cache.

        The materialised items are written through to the inner arm, the corpus
        fingerprint is recomputed, and the cache is cleared because a new corpus
        invalidates every previously memoised ranking.
        """
        materialised = list(items)
        self._inner.write(materialised)
        self._fingerprint = self._corpus_fingerprint(materialised)
        self._cache.clear()

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Return the wrapped arm's result for ``query``, memoised by content.

        On a hit the cached :class:`RetrievedContext` is returned and promoted to
        most-recently-used; on a miss the wrapped arm is consulted, the result is
        stored, and the least-recently-used entry is evicted if ``capacity`` is
        exceeded.
        """
        key = self._key(query, budget_tokens)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            self._cache.move_to_end(key)
            return cached
        self.misses += 1
        result = self._inner.retrieve(query, budget_tokens)
        self._cache[key] = result
        self._cache.move_to_end(key)
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
        return result

    def cache_info(self) -> CacheStats:
        """Return a snapshot of the hit/miss counters and occupancy."""
        return CacheStats(
            hits=self.hits,
            misses=self.misses,
            size=len(self._cache),
            capacity=self._capacity,
        )
