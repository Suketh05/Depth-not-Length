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
//! Paper anchors (`/Users/.../depth_not_length_FIXED.tex`):
//!   * similarity / structured recovery — Thm `thm:sim` (`eq:simceiling`),
//!     Thm `thm:struct`; §`sec:depth` synthetic operating point s0=0.70, rho=0.67,
//!     q=0.97.
//!   * gap and crossover — `eq:gd`, Prop `prop:cross`.
//!   * BM25+ scorer — retrieval harness §`sec:offline`; Lv & Zhai 2011.
//!   * reciprocal rank fusion — `hybrid_rrf`/`rank_fusion_graph` arm; Cormack et
//!     al. 2009.
//!   * Wilson interval — the use-factor kappa CI, `eq:factor` / tab:realret;
//!     Wilson 1927.

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
}
