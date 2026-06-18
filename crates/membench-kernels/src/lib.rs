//! Native Rust retrieval kernels for membench, exposed to Python via PyO3.
//!
//! These mirror the NumPy reference in `membench.retrieval._native` exactly; the
//! reference is authoritative and produces the reported numbers, while this
//! extension exists so that sweeping 20+ memory systems over thousands of tasks and
//! multiple seeds stays tractable. Correctness is enforced by a byte-for-byte parity
//! test against the reference (see `tests/test_rust_kernel.py`), which runs in CI
//! once the extension is built with maturin. The Python layer falls back to the C
//! kernel or NumPy when this extension is absent, so it is purely optional.

use pyo3::prelude::*;

/// Dot product of two equal-length f32 slices (cosine, given L2-normalised inputs).
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Top-`k` rows of a row-major `(n, dim)` L2-normalised corpus by cosine similarity
/// to an L2-normalised `query`, returned as `(indices, scores)` in descending order.
#[pyfunction]
fn cosine_topk(
    corpus: Vec<f32>,
    n: usize,
    dim: usize,
    query: Vec<f32>,
    k: usize,
) -> PyResult<(Vec<i64>, Vec<f32>)> {
    if corpus.len() != n * dim {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "corpus length must equal n * dim",
        ));
    }
    if query.len() != dim {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "query length must equal dim",
        ));
    }
    let kk = k.min(n);
    // (score, index) pairs, maintained as a sorted-descending top-k buffer.
    let mut best: Vec<(f32, i64)> = Vec::with_capacity(kk + 1);
    for i in 0..n {
        let row = &corpus[i * dim..(i + 1) * dim];
        let score = dot(&query, row);
        if best.len() < kk {
            best.push((score, i as i64));
            best.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        } else if kk > 0 && score > best[kk - 1].0 {
            best[kk - 1] = (score, i as i64);
            best.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        }
    }
    let idx: Vec<i64> = best.iter().map(|(_, i)| *i).collect();
    let scores: Vec<f32> = best.iter().map(|(s, _)| *s).collect();
    Ok((idx, scores))
}

/// The membench native-kernels module.
#[pymodule]
fn membench_kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cosine_topk, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
