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

import numpy as np
import pytest

from membench.retrieval._native import cosine_topk_reference
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
