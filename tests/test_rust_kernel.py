"""Parity tests for the optional Rust kernels against the Python reference.

Two families live here. The original ``cosine_topk`` parity checks the retrieval
kernel in ``lib.rs`` against the NumPy reference. The ``verify.rs`` kernels re-derive
the paper's theory and retrieval-scoring closed forms (recovery bounds, crossover,
BM25+, reciprocal rank fusion, Wilson interval); each is compared here against its
authoritative Python twin in ``membench`` to 1e-12 (looser only for the crossover's
numeric root, where Rust bisection and SciPy Brent converge to the same simple root
to ~1e-9 rather than bit-for-bit).

The whole file skips when the ``membench_kernels`` extension is not built (no Rust
toolchain locally); the CI native job builds it with maturin so the parity checks
run there. Stable-tie inputs are used so descending-order comparisons are unambiguous.
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from membench.metrics.factorization import wilson_interval as py_wilson_interval
from membench.retrieval._native import cosine_topk_reference
from membench.retrieval.bm25_plus import bm25_plus_scores as py_bm25_plus_scores
from membench.retrieval.rank_fusion_graph import reciprocal_rank_fusion as py_rrf
from membench.theory.crossover import find_crossover_depth
from membench.theory.crossover import g as py_gap
from membench.theory.decay import SimilarityDecayModel
from membench.theory.recovery import (
    RecoveryModel,
    log_similarity_recovery,
    similarity_recovery,
    structured_recovery,
)

_HAVE_RUST = importlib.util.find_spec("membench_kernels") is not None

pytestmark = pytest.mark.skipif(not _HAVE_RUST, reason="membench_kernels (Rust) not built")

# Paper synthetic operating point (§sec:depth): s0=0.70, rho=0.67, q=0.97.
_PAPER_S0, _PAPER_RHO, _PAPER_Q = 0.70, 0.67, 0.97


def _normalised(n: int, dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    corpus = rng.normal(size=(n, dim))
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    query = rng.normal(size=dim)
    query /= np.linalg.norm(query)
    return corpus, query


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_rust_matches_reference(seed: int) -> None:
    import membench_kernels

    corpus, query = _normalised(200, 64, seed)
    ref_idx, ref_scores = cosine_topk_reference(corpus, query, k=10)
    idx, scores = membench_kernels.cosine_topk(
        corpus.astype(np.float32).ravel().tolist(),
        corpus.shape[0],
        corpus.shape[1],
        query.astype(np.float32).tolist(),
        10,
    )
    np.testing.assert_array_equal(np.asarray(idx), ref_idx)
    np.testing.assert_allclose(np.asarray(scores), ref_scores, atol=1e-5)


# ---------------------------------------------------------------------------
# verify.rs parity: theory recovery (recovery.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [1, 2, 3, 4, 5, 8, 12])
def test_rust_similarity_recovery_matches_python(d: int) -> None:
    import membench_kernels

    model = SimilarityDecayModel(s0=_PAPER_S0, rho=_PAPER_RHO)
    got = membench_kernels.similarity_recovery(d, _PAPER_S0, _PAPER_RHO)
    assert got == pytest.approx(similarity_recovery(d, model), abs=1e-12, rel=0)


@pytest.mark.parametrize("d", [1, 2, 3, 5, 20, 60])
def test_rust_log_similarity_recovery_matches_python(d: int) -> None:
    import membench_kernels

    model = SimilarityDecayModel(s0=0.5, rho=0.3)
    got = membench_kernels.log_similarity_recovery(d, 0.5, 0.3)
    assert got == pytest.approx(log_similarity_recovery(d, model), abs=1e-12, rel=0)


@pytest.mark.parametrize("d", [1, 2, 3, 4, 5, 10])
def test_rust_structured_recovery_matches_python(d: int) -> None:
    import membench_kernels

    got = membench_kernels.structured_recovery(d, _PAPER_Q)
    assert got == pytest.approx(structured_recovery(d, _PAPER_Q), abs=1e-12, rel=0)


@pytest.mark.parametrize("d", [1.0, 1.5, 2.0, 2.7, 3.0, 4.0])
def test_rust_recovery_gap_matches_python(d: float) -> None:
    import membench_kernels

    model = RecoveryModel(decay=SimilarityDecayModel(s0=_PAPER_S0, rho=_PAPER_RHO), q=_PAPER_Q)
    got = membench_kernels.recovery_gap(d, _PAPER_S0, _PAPER_RHO, _PAPER_Q)
    assert got == pytest.approx(py_gap(d, model), abs=1e-12, rel=0)


# ---------------------------------------------------------------------------
# verify.rs parity: crossover_search (crossover.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tau", [0.0, 0.3, 0.501, 0.6, 0.7935, 0.85, 2.0])
def test_rust_crossover_search_matches_python(tau: float) -> None:
    import membench_kernels

    model = RecoveryModel(decay=SimilarityDecayModel(s0=_PAPER_S0, rho=_PAPER_RHO), q=_PAPER_Q)
    ref = find_crossover_depth(model, tau, max_depth=64)
    exists, d_star, d_cont, win_lo, win_hi, peak, g_curve = membench_kernels.crossover_search(
        _PAPER_S0, _PAPER_RHO, _PAPER_Q, tau, 64
    )

    assert exists == ref.exists
    assert peak == ref.peak_depth
    np.testing.assert_allclose(np.asarray(g_curve), ref.g_curve, atol=1e-12, rtol=0)

    if not ref.exists:
        assert d_star == -1
        assert math.isnan(d_cont)
        assert (win_lo, win_hi) == (-1, -1)
        return

    assert d_star == ref.d_star
    assert (win_lo, win_hi) == ref.win_region
    # Rust bisection vs SciPy Brent: same simple root of g(d) - tau, ~1e-9.
    assert d_cont == pytest.approx(ref.d_star_continuous, abs=1e-9, rel=0)


# ---------------------------------------------------------------------------
# verify.rs parity: BM25+ (bm25_plus.py)
# ---------------------------------------------------------------------------


def _bm25_corpus() -> tuple[list[list[str]], list[str]]:
    corpus = [
        ["alpha", "beta", "beta"],
        ["beta", "gamma"],
        ["alpha", "alpha", "delta", "beta"],
        ["gamma", "gamma", "gamma"],
        ["epsilon"],
    ]
    query = ["beta", "alpha", "gamma"]
    return corpus, query


@pytest.mark.parametrize("delta", [0.0, 1.0])
def test_rust_bm25_plus_matches_python(delta: float) -> None:
    import membench_kernels

    corpus, query = _bm25_corpus()
    ref = py_bm25_plus_scores(corpus, query, k1=1.5, b=0.75, delta=delta)
    got = membench_kernels.bm25_plus_scores(corpus, query, 1.5, 0.75, delta)
    np.testing.assert_allclose(np.asarray(got), ref, atol=1e-12, rtol=0)


def test_rust_bm25_plus_single_doc_symbolic_golden() -> None:
    import membench_kernels

    # golden_pack.md D.1: ln(2)*17/7, the exact BM25+ score for the single-doc case.
    got = membench_kernels.bm25_plus_scores([["alpha", "beta", "beta"]], ["beta"], 1.5, 0.75, 1.0)
    assert got[0] == pytest.approx(math.log(2.0) * 17.0 / 7.0, rel=1e-9)


# ---------------------------------------------------------------------------
# verify.rs parity: reciprocal rank fusion (rank_fusion_graph.py)
# ---------------------------------------------------------------------------


def test_rust_rrf_matches_python_and_orders_c_over_b() -> None:
    import membench_kernels

    rankings = [["a", "b", "c"], ["c", "b"]]
    ref = py_rrf(rankings, k=60)
    ids, scores = membench_kernels.reciprocal_rank_fusion(rankings, 60)
    got = dict(zip(ids, scores, strict=True))

    assert set(got) == set(ref)
    for key, value in ref.items():
        assert got[key] == pytest.approx(value, abs=1e-12, rel=0)

    # golden_pack.md D.3: exact-fraction tie-break c=124/3843 > b=124/3844.
    assert got["c"] > got["b"]
    ranked = sorted(got, key=lambda i: -got[i])
    assert ranked == ["c", "b", "a"]


@pytest.mark.parametrize("k", [1, 10, 60, 200])
def test_rust_rrf_matches_python_across_k(k: int) -> None:
    import membench_kernels

    rankings = [["x", "y", "z", "w"], ["z", "y"], ["w", "x", "z"]]
    ref = py_rrf(rankings, k=k)
    ids, scores = membench_kernels.reciprocal_rank_fusion(rankings, k)
    got = dict(zip(ids, scores, strict=True))
    for key, value in ref.items():
        assert got[key] == pytest.approx(value, abs=1e-12, rel=0)


# ---------------------------------------------------------------------------
# verify.rs parity: Wilson interval (factorization.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("successes", "trials", "confidence"),
    [
        (8.0, 10.0, 0.95),
        (6.0, 8.0, 0.95),
        (0.0, 10.0, 0.95),
        (10.0, 10.0, 0.95),
        (344.0, 604.0, 0.95),  # tab:realret bm25 pooled counts scale
        (5.5, 12.0, 0.90),
        (30.0, 100.0, 0.99),
    ],
)
def test_rust_wilson_interval_matches_python(
    successes: float, trials: float, confidence: float
) -> None:
    import membench_kernels

    ref_low, ref_high = py_wilson_interval(successes, trials, confidence=confidence)
    low, high = membench_kernels.wilson_interval(successes, trials, confidence)
    assert low == pytest.approx(ref_low, abs=1e-12, rel=0)
    assert high == pytest.approx(ref_high, abs=1e-12, rel=0)


# ---------------------------------------------------------------------------
# verify.rs parity: seeded randomized sweep (beyond the fixed goldens)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7, 13, 42])
def test_rust_parity_randomized_sweep(seed: int) -> None:
    import membench_kernels

    rng = np.random.default_rng(seed)
    for _ in range(40):
        s0 = float(rng.uniform(0.3, 1.0))
        rho = float(rng.uniform(0.3, 0.99))
        q = float(rng.uniform(0.5, 1.0))
        d = int(rng.integers(1, 12))
        decay = SimilarityDecayModel(s0=s0, rho=rho)

        assert membench_kernels.similarity_recovery(d, s0, rho) == pytest.approx(
            similarity_recovery(d, decay), abs=1e-12, rel=0
        )
        assert membench_kernels.structured_recovery(d, q) == pytest.approx(
            structured_recovery(d, q), abs=1e-12, rel=0
        )

        # Wilson over random fractional pooled counts and confidence levels.
        trials = float(rng.uniform(1.0, 500.0))
        successes = float(rng.uniform(0.0, trials))
        confidence = float(rng.uniform(0.80, 0.995))
        ref_low, ref_high = py_wilson_interval(successes, trials, confidence=confidence)
        low, high = membench_kernels.wilson_interval(successes, trials, confidence)
        assert low == pytest.approx(ref_low, abs=1e-12, rel=0)
        assert high == pytest.approx(ref_high, abs=1e-12, rel=0)
