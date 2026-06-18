"""Parity test for the optional Rust kernel against the NumPy reference.

Skips when the `membench_kernels` extension is not built (no Rust toolchain locally);
the CI native job builds it with maturin so the parity check runs there. Stable-tie
inputs are used so the descending-order comparison is unambiguous.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from membench.retrieval._native import cosine_topk_reference

_HAVE_RUST = importlib.util.find_spec("membench_kernels") is not None

pytestmark = pytest.mark.skipif(not _HAVE_RUST, reason="membench_kernels (Rust) not built")


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
