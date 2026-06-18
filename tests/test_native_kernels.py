"""Tests for the cosine-top-k NumPy reference and (when built) the native kernel.

The reference is tested unconditionally. The native parity test runs only when
the compiled library is present (built locally via `make -C native/c` or in CI);
otherwise it skips with a clear reason, so the suite is green with or without a
toolchain.
"""

from __future__ import annotations

import numpy as np
import pytest

from membench.retrieval._native import (
    cosine_topk_native,
    cosine_topk_reference,
    native_available,
)
from membench.retrieval.embeddings import HashingEmbeddingProvider


def _normalised_corpus(n: int, dim: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    corpus = rng.normal(size=(n, dim))
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    query = rng.normal(size=dim)
    query /= np.linalg.norm(query)
    return corpus, query


class TestReference:
    def test_returns_descending_topk(self) -> None:
        corpus, query = _normalised_corpus(50, 16)
        idx, scores = cosine_topk_reference(corpus, query, k=5)
        assert idx.shape == (5,) and scores.shape == (5,)
        # scores are sorted descending and match the gathered similarities
        assert np.all(np.diff(scores) <= 0)
        np.testing.assert_allclose(scores, corpus[idx] @ query, atol=1e-9)

    def test_matches_brute_force_argsort(self) -> None:
        corpus, query = _normalised_corpus(40, 8, seed=3)
        idx, _ = cosine_topk_reference(corpus, query, k=7)
        full = np.argsort(-(corpus @ query), kind="stable")[:7]
        np.testing.assert_array_equal(idx, full)

    def test_k_larger_than_corpus(self) -> None:
        corpus, query = _normalised_corpus(3, 8)
        idx, _ = cosine_topk_reference(corpus, query, k=10)
        assert idx.shape == (3,)

    def test_empty_and_zero_k(self) -> None:
        corpus, query = _normalised_corpus(5, 4)
        assert cosine_topk_reference(corpus, query, k=0)[0].shape == (0,)
        assert cosine_topk_reference(np.empty((0, 4)), query, k=3)[0].shape == (0,)

    def test_works_on_real_hashing_embeddings(self) -> None:
        prov = HashingEmbeddingProvider(dim=128)
        corpus = prov.embed(
            [
                "all data exports must call withAuditLog",
                "use DateRangePicker for date ranges",
                "every export endpoint needs an audit log entry",
                "primary buttons use the Button component",
            ]
        )
        query = prov.embed_one("add a CSV export with an audit trail")
        idx, _ = cosine_topk_reference(corpus, query, k=2)
        # the two audit-logging docs should rank above the UI docs
        assert set(idx.tolist()) == {0, 2}


@pytest.mark.skipif(not native_available(), reason="native kernel not built")
class TestNativeParity:
    @pytest.mark.parametrize("seed", [0, 1, 2, 7])
    def test_native_matches_reference(self, seed: int) -> None:
        corpus, query = _normalised_corpus(200, 64, seed=seed)
        ref_idx, ref_scores = cosine_topk_reference(corpus, query, k=10)
        nat_idx, nat_scores = cosine_topk_native(corpus, query, k=10)
        np.testing.assert_array_equal(nat_idx, ref_idx)
        # float32 kernel vs float64 reference: allow a small tolerance
        np.testing.assert_allclose(nat_scores, ref_scores, atol=1e-5)
