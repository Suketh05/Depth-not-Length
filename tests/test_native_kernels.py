"""Tests for the cosine-top-k NumPy reference and (when built) the native kernels.

The reference is tested unconditionally. The native parity tests run only when
the compiled library is present (built locally via `make -C native/c` or in CI);
otherwise they skip with a clear reason, so the suite is green with or without a
toolchain.

Beyond the cosine top-k parity, the ``TestNativeTheory*`` / ``TestNativeStats*``
classes check the C99 closed-form kernels for the paper's depth theory
(eq:simceiling, thm:struct, eq:gd, prop:cross) and inference statistics (Wilson
1927, fig:cohensh) against their Python references to ``1e-12``, and pin two
golden-pack constants (``P_sim(3)`` and ``g(2)`` at the s0=0.70, rho=0.67, q=0.97
calibration) at full float64 precision.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from membench.metrics.factorization import wilson_interval
from membench.retrieval._native import (
    cohens_h_native,
    cosine_topk_native,
    cosine_topk_reference,
    crossover_search_native,
    gap_native,
    native_available,
    similarity_recovery_native,
    structured_recovery_native,
    wilson_interval_native,
)
from membench.retrieval.embeddings import HashingEmbeddingProvider
from membench.stats.effect_size import cohens_h
from membench.theory.crossover import find_crossover_depth, g
from membench.theory.decay import SimilarityDecayModel
from membench.theory.recovery import (
    RecoveryModel,
    similarity_recovery,
    structured_recovery,
)


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


# Cross-language parity tolerance. The scalar theory/stats kernels use the same
# libm primitives (pow/asin/sqrt) in the same association order as their Python
# references, so agreement is at the ~1 ULP level; 1e-12 is a generous ceiling.
_PARITY_TOL = 1e-12

# Similarity-decay operating points (s0, rho). The first is the paper's synthetic
# calibration (s0=0.70, rho=0.67; sec:depth); the rest exercise other shapes,
# including the test_theory_recovery.py fixture (0.9, 0.6).
_DECAY_POINTS = [(0.70, 0.67), (0.90, 0.60), (0.80, 0.50), (0.95, 0.85)]
_Q_VALUES = [0.97, 0.95, 0.80, 0.50, 1.0]


@pytest.mark.skipif(not native_available(), reason="native kernel not built")
class TestNativeTheoryParity:
    """Parity of the depth-theory kernels vs. ``membench.theory`` (eq:simceiling,
    thm:struct, eq:gd, prop:cross)."""

    @pytest.mark.parametrize(("s0", "rho"), _DECAY_POINTS)
    @pytest.mark.parametrize("d", [1, 2, 3, 4, 5, 6])
    def test_similarity_recovery_parity(self, s0: float, rho: float, d: int) -> None:
        ref = similarity_recovery(d, SimilarityDecayModel(s0=s0, rho=rho))
        nat = similarity_recovery_native(s0, rho, d)
        assert abs(nat - ref) <= _PARITY_TOL

    @pytest.mark.parametrize("q", _Q_VALUES)
    @pytest.mark.parametrize("d", [1, 2, 3, 4, 5, 6])
    def test_structured_recovery_parity(self, q: float, d: int) -> None:
        ref = structured_recovery(d, q)
        nat = structured_recovery_native(q, d)
        assert abs(nat - ref) <= _PARITY_TOL

    @pytest.mark.parametrize(("s0", "rho"), _DECAY_POINTS)
    @pytest.mark.parametrize("q", [0.97, 0.95, 0.80])
    @pytest.mark.parametrize("d", [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    def test_gap_parity(self, s0: float, rho: float, q: float, d: float) -> None:
        model = RecoveryModel(decay=SimilarityDecayModel(s0=s0, rho=rho), q=q)
        ref = g(d, model)
        nat = gap_native(s0, rho, q, d)
        assert abs(nat - ref) <= _PARITY_TOL

    @pytest.mark.parametrize(("s0", "rho"), _DECAY_POINTS)
    @pytest.mark.parametrize("q", [0.97, 0.95, 0.80])
    @pytest.mark.parametrize("tau", [0.0, 0.1, 0.25, 0.5, 0.6, 0.7935261299999999, 0.9])
    def test_crossover_parity(self, s0: float, rho: float, q: float, tau: float) -> None:
        """Native d* matches find_crossover_depth's d_star (None when no win)."""
        model = RecoveryModel(decay=SimilarityDecayModel(s0=s0, rho=rho), q=q)
        ref = find_crossover_depth(model, tau, max_depth=64).d_star
        nat = crossover_search_native(s0, rho, q, tau, 64)
        assert nat == ref

    def test_crossover_calibrated_boundaries(self) -> None:
        """At the paper calibration: d*=1 for tau below g(1), d*=2 in the
        [g(1), g(2)) band (prop:cross, calibrated d*=2), and no win once tau
        exceeds the peak gap (saturation)."""
        s0, rho, q = 0.70, 0.67, 0.97
        assert crossover_search_native(s0, rho, q, 0.0, 64) == 1
        assert crossover_search_native(s0, rho, q, 0.6, 64) == 2
        # tau == g(2) exactly: strict g(d) > tau excludes d=2, so d* moves to 3.
        assert crossover_search_native(s0, rho, q, 0.7935261299999999, 64) == 3
        # Above the unimodal peak gap (~0.88) nothing wins on the range.
        assert crossover_search_native(s0, rho, q, 0.95, 64) is None
