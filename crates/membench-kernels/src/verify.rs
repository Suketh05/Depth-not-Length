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
}
