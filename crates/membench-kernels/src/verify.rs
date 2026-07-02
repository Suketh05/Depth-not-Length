//! Cross-language verification kernels for the "Depth, Not Length" theory and
//! retrieval mathematics, exposed to Python via PyO3.
//!
//! Every function here is a byte-for-byte re-derivation of a closed form that the
//! paper proves or reports, implemented in Rust so a reader can cross-check the
//! published identities in a second language. The **Python reference in
//! `src/membench/` is authoritative** and produces the reported numbers; this
//! crate exists purely to give an independent, statically-typed witness that the
//! same algebra yields the same float64 (and where the value is rational, the same
//! exact fraction). Parity is enforced by `tests/test_rust_kernel.py`, which
//! compares each kernel against its Python twin to 1e-12 (looser only where a
//! numeric root solver is involved, noted per test).
//!
//! Every golden constant asserted in the pure-Rust unit tests below is quoted
//! VERBATIM from the canonical golden pack (sections A.1, A.3, C.6, D.1, D.3) —
//! never a value produced by running this code — and each maps to a labelled
//! anchor in the paper (`depth_not_length_FIXED.tex`):
//!   * similarity / structured recovery — Thm `thm:sim` (`eq:simceiling`),
//!     Thm `thm:struct`; §`sec:depth` synthetic operating point s0=0.70, rho=0.67,
//!     q=0.97 (golden pack A.1/A.2).
//!   * gap and crossover — `eq:gd`, Prop `prop:cross` (golden pack A.3).
//!   * BM25+ scorer — retrieval harness §`sec:offline`; Lv & Zhai 2011
//!     (golden pack D.1).
//!   * reciprocal rank fusion — `hybrid_rrf`/`rank_fusion_graph` arm; Cormack et
//!     al. 2009 (golden pack D.3).
//!   * Wilson interval — the use-factor kappa CI, `eq:factor` / tab:realret;
//!     Wilson 1927 (golden pack C.6).

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// A. Theory: full-chain recovery (recovery.py, bounds.py, crossover.py)
// ---------------------------------------------------------------------------

/// Similarity full-chain recovery `P_sim(d) = s0^d * rho^(d(d+1)/2)`.
///
/// Verifies `membench.theory.recovery.similarity_recovery` (paper `eq:simceiling`,
/// Thm `thm:sim`). The triangular exponent `d(d+1)/2` is evaluated as a float, as
/// in the reference, so the arithmetic matches bit-for-bit.
fn similarity_recovery_impl(d: i64, s0: f64, rho: f64) -> Result<f64, String> {
    check_depth(d)?;
    check_positive("s0", s0)?;
    check_positive("rho", rho)?;
    let df = d as f64;
    let tri = df * (df + 1.0) / 2.0;
    Ok(s0.powf(df) * rho.powf(tri))
}

/// Natural log of `similarity_recovery`: `d*ln(s0) + (d(d+1)/2)*ln(rho)`.
///
/// Verifies `membench.theory.recovery.log_similarity_recovery`; the log form stays
/// finite where the product underflows float64 (paper §`sec:simceiling`).
fn log_similarity_recovery_impl(d: i64, s0: f64, rho: f64) -> Result<f64, String> {
    check_depth(d)?;
    check_positive("s0", s0)?;
    check_positive("rho", rho)?;
    let df = d as f64;
    Ok(df * s0.ln() + (df * (df + 1.0) / 2.0) * rho.ln())
}

/// Structured full-chain recovery `P_struct(d) = q^d`.
///
/// Verifies `membench.theory.recovery.structured_recovery` (Thm `thm:struct`):
/// following typed links pays only a steady per-hop cost `q in (0, 1]`.
fn structured_recovery_impl(d: i64, q: f64) -> Result<f64, String> {
    check_depth(d)?;
    check_probability("q", q)?;
    Ok(q.powf(d as f64))
}

/// Net recovery gain `g(d) = q^d - s0^d * rho^(d(d+1)/2)` for continuous `d`.
///
/// Verifies `membench.theory.crossover.g` (paper `eq:gd`); defined for real `d`
/// so the crossover can be located between integers.
fn recovery_gap_impl(d: f64, s0: f64, rho: f64, q: f64) -> f64 {
    let p_struct = q.powf(d);
    let p_sim = s0.powf(d) * rho.powf(d * (d + 1.0) / 2.0);
    p_struct - p_sim
}

fn check_depth(d: i64) -> Result<(), String> {
    if d < 1 {
        return Err(format!("depth d must be >= 1, got {d}"));
    }
    Ok(())
}

/// Reject non-positive or NaN parameters (mirrors the Python `not x > 0` guard,
/// which also rejects NaN because `NaN > 0` is false).
fn check_positive(name: &str, value: f64) -> Result<(), String> {
    if value <= 0.0 || value.is_nan() {
        return Err(format!("{name} must be positive, got {value}"));
    }
    Ok(())
}

/// Reject values outside `(0, 1]` or NaN (mirrors the Python `not 0 < q <= 1`).
fn check_probability(name: &str, value: f64) -> Result<(), String> {
    if value <= 0.0 || value > 1.0 || value.is_nan() {
        return Err(format!("{name} must be in (0, 1], got {value}"));
    }
    Ok(())
}

fn to_py_err(msg: String) -> PyErr {
    PyValueError::new_err(msg)
}

/// `P_sim(d) = s0^d * rho^(d(d+1)/2)` — see [`similarity_recovery_impl`].
#[pyfunction]
pub fn similarity_recovery(d: i64, s0: f64, rho: f64) -> PyResult<f64> {
    similarity_recovery_impl(d, s0, rho).map_err(to_py_err)
}

/// `log P_sim(d)` — see [`log_similarity_recovery_impl`].
#[pyfunction]
pub fn log_similarity_recovery(d: i64, s0: f64, rho: f64) -> PyResult<f64> {
    log_similarity_recovery_impl(d, s0, rho).map_err(to_py_err)
}

/// `P_struct(d) = q^d` — see [`structured_recovery_impl`].
#[pyfunction]
pub fn structured_recovery(d: i64, q: f64) -> PyResult<f64> {
    structured_recovery_impl(d, q).map_err(to_py_err)
}

/// `g(d) = P_struct(d) - P_sim(d)` — see [`recovery_gap_impl`].
#[pyfunction]
pub fn recovery_gap(d: f64, s0: f64, rho: f64, q: f64) -> PyResult<f64> {
    Ok(recovery_gap_impl(d, s0, rho, q))
}

/// Outcome of a crossover scan, mirroring `crossover.CrossoverResult`.
struct CrossoverOutcome {
    exists: bool,
    d_star: i64,
    d_star_continuous: f64,
    win_lo: i64,
    win_hi: i64,
    peak_depth: i64,
    g_curve: Vec<f64>,
}

/// Locate the crossover depth `d*` for overhead ratio `tau`, verifying
/// `membench.theory.crossover.find_crossover_depth` (paper Prop `prop:cross`).
///
/// Scans integer depths `1..=max_depth` for the contiguous region where
/// `g(d) > tau`, takes its smallest integer as `d*`, and refines a continuous
/// crossover on the rising edge by bisection (the reference uses Brent's method;
/// both converge to the same simple root of `g(d) - tau`). None-valued fields are
/// sentinelled: `d_star = -1`, `d_star_continuous = NaN`, `win_lo = win_hi = -1`.
fn crossover_search_impl(
    s0: f64,
    rho: f64,
    q: f64,
    tau: f64,
    max_depth: i64,
) -> Result<CrossoverOutcome, String> {
    if tau < 0.0 || tau.is_nan() {
        return Err(format!("tau must be non-negative, got {tau}"));
    }
    if max_depth < 1 {
        return Err(format!("max_depth must be >= 1, got {max_depth}"));
    }

    let g = |d: f64| recovery_gap_impl(d, s0, rho, q);

    let g_curve: Vec<f64> = (1..=max_depth).map(|d| g(d as f64)).collect();

    // peak = first depth attaining the maximum (np.argmax semantics).
    let mut peak_idx = 0usize;
    for (i, &v) in g_curve.iter().enumerate() {
        if v > g_curve[peak_idx] {
            peak_idx = i;
        }
    }
    let peak_depth = (peak_idx + 1) as i64;

    // First integer depth with g(d) > tau (np.argmax over the boolean mask).
    let first = g_curve.iter().position(|&v| v > tau);
    let Some(first_idx) = first else {
        return Ok(CrossoverOutcome {
            exists: false,
            d_star: -1,
            d_star_continuous: f64::NAN,
            win_lo: -1,
            win_hi: -1,
            peak_depth,
            g_curve,
        });
    };
    let d_star = (first_idx + 1) as i64;

    // Contiguous win region starting at d_star (a closed interval for q < 1).
    let mut last_idx = first_idx;
    while last_idx + 1 < g_curve.len() && g_curve[last_idx + 1] > tau {
        last_idx += 1;
    }
    let win_hi = (last_idx + 1) as i64;

    // Continuous crossover on the rising edge: g(d) - tau changes sign in
    // (d_star - 1, d_star] unless the model already wins at d = 1.
    let lo = (d_star - 1) as f64;
    let d_continuous = if lo < 1e-9 || g(lo) - tau >= 0.0 {
        d_star as f64
    } else {
        bisect_root(&g, tau, lo, d_star as f64)
    };

    Ok(CrossoverOutcome {
        exists: true,
        d_star,
        d_star_continuous: d_continuous,
        win_lo: d_star,
        win_hi,
        peak_depth,
        g_curve,
    })
}

/// Bisection root of `g(d) - tau` on `[lo, hi]`, where `g(lo) < tau < g(hi)`.
///
/// Iterated to machine-precision bracket width so the returned root matches the
/// reference's Brent solution to well within the parity tolerance.
fn bisect_root(g: &impl Fn(f64) -> f64, tau: f64, mut lo: f64, mut hi: f64) -> f64 {
    let f = |d: f64| g(d) - tau;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if hi - lo <= f64::EPSILON * hi.abs().max(1.0) {
            return mid;
        }
        if f(mid) > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

// ---------------------------------------------------------------------------
// D. Retrieval scoring: BM25+ (bm25_plus.py, Lv & Zhai 2011)
// ---------------------------------------------------------------------------

/// BM25+ score of every document for a tokenized query.
///
/// Verifies `membench.retrieval.bm25_plus.bm25_plus_scores` (paper retrieval
/// harness §`sec:offline`; Lv & Zhai, CIKM 2011). For each query term that occurs
/// in a document,
/// `S += idf * (delta + (k1+1)*tf / (k1*(1 - b + b*|D|/avgdl) + tf))`, with the
/// non-negative IDF `ln((N+1)/df)`. The lower bound `delta` is added only for
/// present terms — the property that guarantees a score gap between documents that
/// contain a term and those that do not. `delta = 0` recovers Okapi BM25 exactly.
fn bm25_plus_scores_impl(
    corpus: &[Vec<String>],
    query: &[String],
    k1: f64,
    b: f64,
    delta: f64,
) -> Vec<f64> {
    let n = corpus.len();
    let mut scores = vec![0.0f64; n];
    if n == 0 {
        return scores;
    }
    let doc_len: Vec<f64> = corpus.iter().map(|d| d.len() as f64).collect();
    let avgdl = doc_len.iter().sum::<f64>() / n as f64;
    if avgdl == 0.0 {
        // Degenerate corpus: every document is empty, so nothing can match.
        return scores;
    }
    // Per-document term counts.
    let counters: Vec<HashMap<&str, f64>> = corpus
        .iter()
        .map(|doc| {
            let mut c: HashMap<&str, f64> = HashMap::new();
            for tok in doc {
                *c.entry(tok.as_str()).or_insert(0.0) += 1.0;
            }
            c
        })
        .collect();
    // Length-normalized denominator term k1*(1 - b + b*|D|/avgdl), per document.
    let norm: Vec<f64> = doc_len
        .iter()
        .map(|&dl| k1 * (1.0 - b + b * dl / avgdl))
        .collect();

    for term in query {
        let key = term.as_str();
        let df = counters.iter().filter(|c| c.contains_key(key)).count();
        if df == 0 {
            continue;
        }
        let idf = ((n + 1) as f64 / df as f64).ln();
        for i in 0..n {
            let tf = counters[i].get(key).copied().unwrap_or(0.0);
            let ratio = (k1 + 1.0) * tf / (norm[i] + tf);
            let lower = if tf > 0.0 { delta } else { 0.0 };
            scores[i] += idf * (lower + ratio);
        }
    }
    scores
}

/// BM25+ scores aligned with corpus order. See [`bm25_plus_scores_impl`].
#[pyfunction]
#[pyo3(signature = (corpus, query, k1=1.5, b=0.75, delta=1.0))]
pub fn bm25_plus_scores(
    corpus: Vec<Vec<String>>,
    query: Vec<String>,
    k1: f64,
    b: f64,
    delta: f64,
) -> PyResult<Vec<f64>> {
    Ok(bm25_plus_scores_impl(&corpus, &query, k1, b, delta))
}

// ---------------------------------------------------------------------------
// C. Statistics: Wilson score interval (factorization.py; Wilson 1927)
// ---------------------------------------------------------------------------

/// Error function via the all-positive scaled Maclaurin series
/// `erf(x) = (2/sqrt(pi)) e^{-x^2} x sum_{n>=0} (2x^2)^n / (1*3*...*(2n+1))`.
///
/// Every term is positive, so there is no catastrophic cancellation and the sum
/// reaches full double precision for the `|x| <= ~4` range the probit needs. Used
/// only to Newton-refine the inverse-normal seed below.
fn erf(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let a = x.abs();
    let a2 = a * a;
    let mut term = a; // n = 0 term (before the 2/sqrt(pi) e^{-x^2} factor)
    let mut sum = term;
    let mut n = 1.0;
    loop {
        term *= 2.0 * a2 / (2.0 * n + 1.0);
        sum += term;
        if term <= sum * 1e-18 || n > 300.0 {
            break;
        }
        n += 1.0;
    }
    sign * std::f64::consts::FRAC_2_SQRT_PI * (-a2).exp() * sum
}

/// Standard-normal CDF `Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))`.
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x * std::f64::consts::FRAC_1_SQRT_2))
}

/// Peter Acklam's rational approximation to the inverse standard-normal CDF; used
/// as a seed (relative error < 1.15e-9) that Newton refinement drives to full
/// double precision.
// Acklam's published coefficients carry 16 significant digits (correctly rounded
// nearest doubles); keep them verbatim for provenance.
#[allow(clippy::excessive_precision)]
fn acklam_ppf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    let p_high = 1.0 - P_LOW;
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Inverse standard-normal CDF (probit) to full double precision.
///
/// Works in the upper half `p' = max(p, 1-p) >= 0.5` (so the Newton residual
/// `Phi(x) - p'` is evaluated with a non-negative erf and never cancels), then
/// restores the sign. Two Newton steps take Acklam's ~1e-9 seed to ~1e-15.
fn probit(p: f64) -> Result<f64, String> {
    if !(0.0 < p && p < 1.0) {
        return Err(format!("probit argument must be in (0, 1), got {p}"));
    }
    let (pp, sign) = if p < 0.5 { (1.0 - p, -1.0) } else { (p, 1.0) };
    let mut x = acklam_ppf(pp);
    let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    for _ in 0..2 {
        let e = normal_cdf(x) - pp;
        let pdf = (-0.5 * x * x).exp() * inv_sqrt_2pi;
        x -= e / pdf;
    }
    Ok(sign * x)
}

/// Wilson score interval for a binomial proportion.
///
/// Verifies `membench.metrics.factorization.wilson_interval` (paper `eq:factor` /
/// tab:realret use-factor kappa CI; Wilson 1927):
/// `(p_hat + z^2/2n +/- z sqrt(p_hat(1-p_hat)/n + z^2/4n^2)) / (1 + z^2/n)`,
/// clamped to `[0, 1]`, with `z` the standard-normal quantile for `confidence`.
/// Fractional (pooled expected) counts are allowed.
fn wilson_interval_impl(
    successes: f64,
    trials: f64,
    confidence: f64,
) -> Result<(f64, f64), String> {
    if trials <= 0.0 {
        return Err("trials must be positive".to_string());
    }
    if successes < 0.0 {
        return Err("successes must be non-negative".to_string());
    }
    if !(0.0 < confidence && confidence < 1.0) {
        return Err("confidence must be in (0, 1)".to_string());
    }
    let p_hat = (successes / trials).clamp(0.0, 1.0);
    let z = probit(1.0 - (1.0 - confidence) / 2.0)?;
    let z2 = z * z;
    let denominator = 1.0 + z2 / trials;
    let center = (p_hat + z2 / (2.0 * trials)) / denominator;
    let margin =
        z * (p_hat * (1.0 - p_hat) / trials + z2 / (4.0 * trials * trials)).sqrt() / denominator;
    let low = (center - margin).max(0.0);
    let high = (center + margin).min(1.0);
    Ok((low, high))
}

/// Wilson score `(low, high)` interval. See [`wilson_interval_impl`].
#[pyfunction]
#[pyo3(signature = (successes, trials, confidence=0.95))]
pub fn wilson_interval(successes: f64, trials: f64, confidence: f64) -> PyResult<(f64, f64)> {
    wilson_interval_impl(successes, trials, confidence).map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// D. Retrieval scoring: Reciprocal Rank Fusion (rank_fusion_graph.py; Cormack 2009)
// ---------------------------------------------------------------------------

/// Reciprocal Rank Fusion `RRF(d) = sum_r 1/(k + rank_r(d))` over 1-based ranks.
///
/// Verifies `membench.retrieval.rank_fusion_graph.reciprocal_rank_fusion` (Cormack,
/// Clarke & Buettcher, SIGIR 2009). A document contributes a term only for the
/// rankings that contain it, so an item present in two lists is rewarded over one
/// present in a single list at the same position — the property that lets RRF fuse
/// incommensurable score scales using ranks alone. Returns ids in first-seen order
/// (matching the reference dict's insertion order) with their fused scores.
fn reciprocal_rank_fusion_impl(
    rankings: &[Vec<String>],
    k: i64,
) -> Result<Vec<(String, f64)>, String> {
    if k <= 0 {
        return Err(format!("RRF k must be positive, got {k}"));
    }
    let kf = k as f64;
    let mut order: Vec<String> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut scores: Vec<f64> = Vec::new();
    for ranking in rankings {
        for (rank0, item_id) in ranking.iter().enumerate() {
            let rank = (rank0 + 1) as f64;
            let contribution = 1.0 / (kf + rank);
            match index.get(item_id) {
                Some(&i) => scores[i] += contribution,
                None => {
                    index.insert(item_id.clone(), scores.len());
                    order.push(item_id.clone());
                    scores.push(contribution);
                }
            }
        }
    }
    Ok(order.into_iter().zip(scores).collect())
}

/// RRF fusion; returns `(ids, scores)` in first-seen order. `k` must be positive.
/// See [`reciprocal_rank_fusion_impl`].
#[pyfunction]
#[pyo3(signature = (rankings, k=60))]
pub fn reciprocal_rank_fusion(
    rankings: Vec<Vec<String>>,
    k: i64,
) -> PyResult<(Vec<String>, Vec<f64>)> {
    let fused = reciprocal_rank_fusion_impl(&rankings, k).map_err(to_py_err)?;
    let ids = fused.iter().map(|(id, _)| id.clone()).collect();
    let scores = fused.iter().map(|(_, s)| *s).collect();
    Ok((ids, scores))
}

/// Crossover scan; returns `(exists, d_star, d_star_continuous, win_lo, win_hi,
/// peak_depth, g_curve)` with `-1`/`NaN` sentinels when no crossover exists.
/// See [`crossover_search_impl`].
#[pyfunction]
#[pyo3(signature = (s0, rho, q, tau, max_depth=64))]
// The flat tuple is the deliberate wire format for the Python parity test, which
// unpacks it field-by-field against `CrossoverResult`; a struct would need a
// pyclass round-trip that adds no clarity here.
#[allow(clippy::type_complexity)]
pub fn crossover_search(
    s0: f64,
    rho: f64,
    q: f64,
    tau: f64,
    max_depth: i64,
) -> PyResult<(bool, i64, f64, i64, i64, i64, Vec<f64>)> {
    let o = crossover_search_impl(s0, rho, q, tau, max_depth).map_err(to_py_err)?;
    Ok((
        o.exists,
        o.d_star,
        o.d_star_continuous,
        o.win_lo,
        o.win_hi,
        o.peak_depth,
        o.g_curve,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tolerance for float64 identities that should be exact modulo IEEE-754 dust.
    const TOL: f64 = 1e-12;

    // Golden pack A.1: paper calibration s0=0.70, rho=0.67, q=0.97, d=1..5.
    // Values asserted VERBATIM from golden_pack.md (never produced by this code).
    #[test]
    fn similarity_recovery_matches_paper_calibration() {
        let s0 = 0.70;
        let rho = 0.67;
        // (d, P_sim(d)) from golden_pack.md table A.1.
        let expected = [
            (1, 0.469),
            (2, 0.14737387000000002),
            (3, 0.031027225083967002),
            (4, 0.004376633568728781),
            (5, 0.0004136302005695817),
        ];
        for (d, want) in expected {
            let got = similarity_recovery_impl(d, s0, rho).unwrap();
            assert!((got - want).abs() <= TOL, "d={d}: got {got}, want {want}");
        }
    }

    #[test]
    fn structured_recovery_matches_paper_calibration() {
        let q = 0.97;
        // P_struct(d) = q^d from golden_pack.md table A.1.
        let expected = [
            (1, 0.97),
            (2, 0.9409),
            (3, 0.912673),
            (4, 0.8852928099999999),
            (5, 0.8587340256999999),
        ];
        for (d, want) in expected {
            let got = structured_recovery_impl(d, q).unwrap();
            assert!((got - want).abs() <= TOL, "d={d}: got {got}, want {want}");
        }
    }

    #[test]
    fn recovery_gap_matches_paper_calibration() {
        let (s0, rho, q) = (0.70, 0.67, 0.97);
        // g(d) = q^d - P_sim(d) from golden_pack.md table A.1.
        let expected = [
            (1.0, 0.501),
            (2.0, 0.7935261299999999),
            (3.0, 0.8816457749160329),
            (4.0, 0.8809161764312712),
            (5.0, 0.8583203954994303),
        ];
        for (d, want) in expected {
            let got = recovery_gap_impl(d, s0, rho, q);
            assert!((got - want).abs() <= TOL, "d={d}: got {got}, want {want}");
        }
    }

    // Golden pack A.1 test-file goldens: closed form vs explicit product for
    // s0=0.9, rho=0.6, and the structured_recovery(5, 0.95) == 0.95^5 identity.
    #[test]
    fn similarity_recovery_equals_explicit_product() {
        let (s0, rho): (f64, f64) = (0.9, 0.6);
        for d in 1..=6 {
            let mut prod = 1.0f64;
            for i in 1..=d {
                prod *= s0 * rho.powi(i);
            }
            let got = similarity_recovery_impl(d as i64, s0, rho).unwrap();
            assert!((got - prod).abs() <= TOL, "d={d}: {got} vs {prod}");
        }
    }

    #[test]
    fn structured_recovery_pow_identity() {
        let got = structured_recovery_impl(5, 0.95).unwrap();
        assert!((got - 0.95f64.powi(5)).abs() <= TOL);
    }

    // Golden pack A.1: underflow guard. s0=0.5, rho=0.3 -> P_sim(60) underflows to
    // 0.0 while log form stays finite. Provenance: test_theory_recovery.py.
    #[test]
    fn similarity_recovery_underflows_but_log_is_finite() {
        let (s0, rho) = (0.5, 0.3);
        assert_eq!(similarity_recovery_impl(60, s0, rho).unwrap(), 0.0);
        assert!(log_similarity_recovery_impl(60, s0, rho)
            .unwrap()
            .is_finite());
    }

    // Cross-check: ratio q^3 / P_sim(3) ~ 29x gap (paper §sec:depth "~29x gap").
    #[test]
    fn depth_three_ratio_is_the_twenty_nine_fold_gap() {
        let p_sim = similarity_recovery_impl(3, 0.70, 0.67).unwrap();
        let p_struct = structured_recovery_impl(3, 0.97).unwrap();
        let ratio = p_struct / p_sim;
        assert!((ratio - 29.415231221293272).abs() <= 1e-9, "ratio={ratio}");
    }

    #[test]
    fn recovery_rejects_bad_domain() {
        assert!(similarity_recovery_impl(0, 0.7, 0.67).is_err());
        assert!(structured_recovery_impl(1, 0.0).is_err());
        assert!(structured_recovery_impl(1, 1.5).is_err());
        assert!(similarity_recovery_impl(1, 0.0, 0.67).is_err());
    }

    // ---- Crossover (golden_pack.md A.3, paper prop:cross) ----

    // Calibrated d* = 2: strict-win when g(1) <= tau < g(2), i.e.
    // tau in [0.501, 0.7935261299999999). Paper: "d*=2 for 0.50 < tau <= 0.79".
    #[test]
    fn crossover_calibrated_d_star_is_two() {
        let o = crossover_search_impl(0.70, 0.67, 0.97, 0.6, 64).unwrap();
        assert!(o.exists);
        assert_eq!(o.d_star, 2);
        // g(1)=0.501 < 0.6, g(2)=0.7935 > 0.6, so the win region opens at d=2.
        assert_eq!(o.win_lo, 2);
    }

    // Closed-form crossover d* = 1 at tau = 0 (structure already wins at d=1,
    // since g(1)=0.501 > 0). Peak advantage is at d=3 (g(3)=0.8816 is maximal).
    #[test]
    fn crossover_closed_form_d_star_is_one_at_zero_overhead() {
        let o = crossover_search_impl(0.70, 0.67, 0.97, 0.0, 64).unwrap();
        assert!(o.exists);
        assert_eq!(o.d_star, 1);
        assert_eq!(o.peak_depth, 3);
        // lo = d_star - 1 = 0 < 1e-9, so the continuous root is pinned to d_star.
        assert_eq!(o.d_star_continuous, 1.0);
    }

    // Continuous crossover root satisfies g(d_star_continuous) ~ tau (abs 1e-6),
    // the invariant from test_theory_crossover.py.
    #[test]
    fn crossover_continuous_root_hits_tau() {
        let (s0, rho, q, tau) = (0.70, 0.67, 0.97, 0.6);
        let o = crossover_search_impl(s0, rho, q, tau, 64).unwrap();
        let root = o.d_star_continuous;
        assert!(root > 1.0 && root < 2.0, "root={root}");
        assert!((recovery_gap_impl(root, s0, rho, q) - tau).abs() <= 1e-9);
    }

    // No crossover when the overhead exceeds the maximal gain: exists=false and
    // every None field is sentinelled.
    #[test]
    fn crossover_absent_when_overhead_too_high() {
        let o = crossover_search_impl(0.70, 0.67, 0.97, 2.0, 64).unwrap();
        assert!(!o.exists);
        assert_eq!(o.d_star, -1);
        assert!(o.d_star_continuous.is_nan());
        assert_eq!((o.win_lo, o.win_hi), (-1, -1));
    }

    #[test]
    fn crossover_rejects_bad_domain() {
        assert!(crossover_search_impl(0.7, 0.67, 0.97, -0.1, 64).is_err());
        assert!(crossover_search_impl(0.7, 0.67, 0.97, 0.0, 0).is_err());
    }

    // ---- BM25+ (golden_pack.md D.1, Lv & Zhai 2011) ----

    fn toks(words: &[&str]) -> Vec<String> {
        words.iter().map(|s| s.to_string()).collect()
    }

    // Single-doc corpus [["alpha","beta","beta"]], query ["beta"], k1=1.5, b=0.75.
    // BM25+ (delta=1) = ln(2)*17/7; BM25 (delta=0) = ln(2)*10/7; gap = idf*delta = ln 2.
    #[test]
    fn bm25_plus_single_doc_golden() {
        let corpus = vec![toks(&["alpha", "beta", "beta"])];
        let query = toks(&["beta"]);
        let plus = bm25_plus_scores_impl(&corpus, &query, 1.5, 0.75, 1.0);
        let bm25 = bm25_plus_scores_impl(&corpus, &query, 1.5, 0.75, 0.0);
        let ln2 = 2.0f64.ln();
        assert!(
            (plus[0] - ln2 * 17.0 / 7.0).abs() <= 1e-12,
            "got {}",
            plus[0]
        );
        assert!((plus[0] - 1.6833574385027244).abs() <= 1e-9);
        assert!((bm25[0] - ln2 * 10.0 / 7.0).abs() <= 1e-12);
        assert!((bm25[0] - 0.9902102579427790).abs() <= 1e-9);
        assert!((plus[0] - bm25[0] - ln2).abs() <= 1e-12);
    }

    // Long-doc property: a 2-doc corpus where doc0 is "audit" + 40 fillers and doc1
    // shares no query term. BM25+ - BM25 on doc0 = idf*delta = ln 3; the
    // non-containing doc scores exactly 0.0 (presence-gated).
    #[test]
    fn bm25_plus_long_doc_gap_is_idf_delta() {
        let mut doc0 = toks(&["audit"]);
        doc0.extend(std::iter::repeat_n("x".to_string(), 40));
        let corpus = vec![doc0, toks(&["button", "component"])];
        let query = toks(&["audit"]);
        let plus = bm25_plus_scores_impl(&corpus, &query, 1.5, 0.75, 1.0);
        let bm25 = bm25_plus_scores_impl(&corpus, &query, 1.5, 0.75, 0.0);
        assert!((plus[0] - bm25[0] - 3.0f64.ln()).abs() <= 1e-12);
        assert!((plus[0] - bm25[0] - 1.0986122886681098).abs() <= 1e-9);
        assert_eq!(plus[1], 0.0);
        assert_eq!(bm25[1], 0.0);
    }

    #[test]
    fn bm25_plus_empty_corpus_is_empty() {
        assert!(bm25_plus_scores_impl(&[], &toks(&["beta"]), 1.5, 0.75, 1.0).is_empty());
    }

    // ---- Reciprocal Rank Fusion (golden_pack.md D.3, Cormack 2009) ----

    fn rrf_score(fused: &[(String, f64)], id: &str) -> f64 {
        fused.iter().find(|(k, _)| k == id).unwrap().1
    }

    // Rankings A=[a,b,c], B=[c,b] with k=60. The exact-fraction tie-break
    // c=124/3843 > b=124/3844 (same numerator, smaller denominator) fixes the
    // fused order [c, b, a]. Values asserted VERBATIM from golden_pack.md D.3.
    #[test]
    fn rrf_exact_fraction_tie_break_orders_c_over_b() {
        let a = toks(&["a", "b", "c"]);
        let b = toks(&["c", "b"]);
        let fused = reciprocal_rank_fusion_impl(&[a, b], 60).unwrap();

        let sa = rrf_score(&fused, "a");
        let sb = rrf_score(&fused, "b");
        let sc = rrf_score(&fused, "c");

        // a = 1/61 = 63/3843; b = 2/62 = 124/3844 = 1/31; c = 1/63 + 1/61 = 124/3843.
        assert!((sa - 1.0 / 61.0).abs() <= 1e-15);
        assert!((sa - 0.01639344262295082).abs() <= 1e-15);
        assert!((sb - 124.0 / 3844.0).abs() <= 1e-15);
        assert!((sb - 0.03225806451612903).abs() <= 1e-15);
        assert!((sc - (1.0 / 63.0 + 1.0 / 61.0)).abs() <= 1e-15);
        assert!((sc - 0.032266458495966696).abs() <= 1e-15);

        // The crux: c strictly beats b despite the shared numerator 124.
        assert!(sc > sb, "c={sc} must beat b={sb}");

        // First-seen order is [a, b, c]; the *ranked* order is [c, b, a].
        assert_eq!(
            fused.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );
        let mut ranked: Vec<&(String, f64)> = fused.iter().collect();
        ranked.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
        assert_eq!(
            ranked.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
            vec!["c", "b", "a"]
        );
    }

    // Agreement golden: [[x,y],[x]] -> x = 2/61, y = 1/62.
    #[test]
    fn rrf_agreement_rewards_repeated_item() {
        let fused = reciprocal_rank_fusion_impl(&[toks(&["x", "y"]), toks(&["x"])], 60).unwrap();
        assert!((rrf_score(&fused, "x") - 2.0 / 61.0).abs() <= 1e-15);
        assert!((rrf_score(&fused, "x") - 0.03278688524590164).abs() <= 1e-15);
        assert!((rrf_score(&fused, "y") - 1.0 / 62.0).abs() <= 1e-15);
    }

    #[test]
    fn rrf_rejects_nonpositive_k() {
        assert!(reciprocal_rank_fusion_impl(&[toks(&["a"])], 0).is_err());
        assert!(reciprocal_rank_fusion_impl(&[toks(&["a"])], -1).is_err());
    }

    // ---- Wilson interval + supporting probit (golden_pack.md C.6) ----

    // erf reference values (DLMF / standard tables) confirm the series is correct.
    #[test]
    fn erf_matches_reference_values() {
        assert_eq!(erf(0.0), 0.0);
        assert!((erf(1.0) - 0.842_700_792_949_714_9).abs() <= 1e-15);
        assert!((erf(-1.0) + 0.842_700_792_949_714_9).abs() <= 1e-15);
        assert!((erf(2.0) - 0.995_322_265_018_952_7).abs() <= 1e-15);
    }

    // The 95% two-sided z-quantile, asserted to full precision (golden_pack.md C.6).
    #[test]
    fn probit_matches_ninety_five_percent_z() {
        let z = probit(0.975).unwrap();
        assert!((z - 1.959963984540054).abs() <= 1e-12, "z={z}");
    }

    #[test]
    fn probit_is_antisymmetric_and_zero_at_half() {
        assert!((probit(0.5).unwrap()).abs() <= 1e-14);
        assert!((probit(0.1).unwrap() + probit(0.9).unwrap()).abs() <= 1e-12);
    }

    // (8 successes, 10 trials): center 0.7167401600411333, margin 0.22657768850449148,
    // (low, high) = (0.49016247153664183, 0.9433178485456247). VERBATIM golden C.6.
    #[test]
    fn wilson_interval_eight_of_ten_golden() {
        let (low, high) = wilson_interval_impl(8.0, 10.0, 0.95).unwrap();
        assert!((low - 0.49016247153664183).abs() <= 1e-12, "low={low}");
        assert!((high - 0.9433178485456247).abs() <= 1e-12, "high={high}");
    }

    // (6, 8), p_hat = 0.75 (the pooled-kappa CI case): center 0.6688981087790299,
    // margin 0.25962267846886106, (0.40927543031016883, 0.9285207872478909).
    #[test]
    fn wilson_interval_six_of_eight_golden() {
        let (low, high) = wilson_interval_impl(6.0, 8.0, 0.95).unwrap();
        assert!((low - 0.40927543031016883).abs() <= 1e-12, "low={low}");
        assert!((high - 0.9285207872478909).abs() <= 1e-12, "high={high}");
    }

    #[test]
    fn wilson_interval_rejects_bad_domain() {
        assert!(wilson_interval_impl(1.0, 0.0, 0.95).is_err());
        assert!(wilson_interval_impl(-1.0, 10.0, 0.95).is_err());
        assert!(wilson_interval_impl(1.0, 10.0, 1.5).is_err());
    }

    // ---- Hardening: edge cases (saturation, chance floor, degeneracy) ----

    // q = 1 removes structured decay, so g(d) = 1 - P_sim(d) is monotone increasing
    // and the win is a half-line: the region runs to the scan edge (crossover.py note
    // that Prop 2's monotonicity holds only in the q = 1 limit). max_depth is kept
    // shallow so P_sim does not underflow to 0 (which would flat-top the curve).
    #[test]
    fn crossover_q_one_is_monotone_half_line() {
        let o = crossover_search_impl(0.70, 0.67, 1.0, 0.6, 8).unwrap();
        assert!(o.exists);
        // g is strictly increasing over 1..=8, so it is non-decreasing and the win
        // region runs to the deepest scanned depth (the half-line, clipped to edge).
        for w in o.g_curve.windows(2) {
            assert!(w[1] >= w[0], "g must be non-decreasing at q=1");
        }
        assert_eq!(o.peak_depth, 8);
        assert_eq!(o.win_hi, 8);
    }

    // Wilson bounds clamp into [0, 1]: 0 successes floors low at 0.0; all successes
    // ceilings high at 1.0 (golden_pack.md C.6 clamp behaviour).
    #[test]
    fn wilson_interval_clamps_at_bounds() {
        let (low, _) = wilson_interval_impl(0.0, 10.0, 0.95).unwrap();
        assert_eq!(low, 0.0);
        let (_, high) = wilson_interval_impl(10.0, 10.0, 0.95).unwrap();
        assert_eq!(high, 1.0);
    }

    // BM25+ degeneracies: an all-empty corpus (avgdl = 0) scores zeros, and a query
    // term absent from every document (df = 0) contributes nothing.
    #[test]
    fn bm25_plus_degenerate_and_oov() {
        let empty =
            bm25_plus_scores_impl(&[toks(&[]), toks(&[])], &toks(&["beta"]), 1.5, 0.75, 1.0);
        assert_eq!(empty, vec![0.0, 0.0]);

        let corpus = vec![toks(&["alpha", "beta"]), toks(&["gamma"])];
        let oov = bm25_plus_scores_impl(&corpus, &toks(&["zeta"]), 1.5, 0.75, 1.0);
        assert_eq!(oov, vec![0.0, 0.0]);
    }

    // The probit stays accurate deep in the tail (e.g. the 99.95% quantile), so the
    // Wilson z is right at high confidence too.
    #[test]
    fn probit_is_accurate_in_the_tail() {
        // Phi(3.290526731491958) = 0.9995 (standard-normal table value).
        let z = probit(0.9995).unwrap();
        assert!((z - 3.2905267314919255).abs() <= 1e-10, "z={z}");
        // Round-trip: Phi(probit(p)) == p.
        for p in [0.6, 0.8, 0.95, 0.99, 0.999] {
            assert!((normal_cdf(probit(p).unwrap()) - p).abs() <= 1e-12);
        }
    }
}
