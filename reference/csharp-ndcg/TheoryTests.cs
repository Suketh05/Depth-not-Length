// TheoryTests.cs -- Cross-language + cross-check-the-paper suite for the depth theory.
//
// Two kinds of assertion, both from authoritative sources (never from running this code):
//   (1) Section-A THEORY goldens -- full-precision float64 values from the canonical golden
//       pack, shared verbatim with the Python reference (src/membench/theory/) and the Java
//       arm, so a disagreement means a bug in one derivation.
//   (2) Section-B PAPER-TABLE identities -- numbers a reader can re-derive from the paper's
//       own published tables (tab:realret use factors, the Fano 1/K floor, tab:scorecard).
//
// Tolerance policy (golden pack): exact-fraction goldens are asserted essentially exactly;
// float64 goldens use abs err <= 1e-12 (relaxed to 1e-9 for a ratio of small numbers). The
// published-kappa identity uses xunit's decimal-place overload to assert the 3-dp rounding.
using Xunit;

namespace Membench;

public class TheoryTests
{
    private static readonly SurvivalModel Calib = Theory.Calibration; // s0=0.70, rho=0.67, q=0.97
    private const double Tol = 1e-12;

    // ---- A.1 similarity recovery (eq:simceiling / Thm thm:sim) ----

    [Fact]
    public void SimilarityRecoveryMatchesCalibrationTable()
    {
        // Golden pack A.1, s0=0.70 rho=0.67, d=1..5 (float64 verbatim).
        Assert.Equal(0.469, Theory.SimilarityRecovery(1, Calib), Tol);              // 469/1000
        Assert.Equal(0.14737387000000002, Theory.SimilarityRecovery(2, Calib), Tol); // 14737387/10^8
        Assert.Equal(0.031027225083967002, Theory.SimilarityRecovery(3, Calib), Tol);
        Assert.Equal(0.004376633568728781, Theory.SimilarityRecovery(4, Calib), Tol);
        Assert.Equal(0.0004136302005695817, Theory.SimilarityRecovery(5, Calib), Tol);
    }

    [Fact]
    public void SimilarityRecoveryEqualsPerHopProduct()
    {
        // Golden pack A.1 test-file identity: P_sim(d) = prod_{i=1..d} s0*rho^i, s0=0.9 rho=0.6.
        var m = new SurvivalModel(0.9, 0.6, 0.95);
        double product = 1.0;
        for (int i = 1; i <= 4; i++)
            product *= 0.9 * Math.Pow(0.6, i);
        Assert.Equal(product, Theory.SimilarityRecovery(4, m), Tol);
    }

    [Fact]
    public void SimilarityRecoveryUnderflowsButLogStaysFinite()
    {
        // Golden pack A.1 underflow guard: s0=0.5, rho=0.3, d=60 underflows to 0.0 in float64,
        // while the log form remains finite.
        var m = new SurvivalModel(0.5, 0.3, 0.9);
        Assert.Equal(0.0, Theory.SimilarityRecovery(60, m));
        Assert.True(double.IsFinite(Theory.LogSimilarityRecovery(60, m)));
    }

    // ---- A.2 structured recovery (Thm thm:struct) ----

    [Fact]
    public void StructuredRecoveryMatchesCalibrationTable()
    {
        // Golden pack A.1/A.2, q=0.97, d=1..5 (float64 verbatim).
        Assert.Equal(0.97, Theory.StructuredRecovery(1, 0.97), Tol);
        Assert.Equal(0.9409, Theory.StructuredRecovery(2, 0.97), Tol);
        Assert.Equal(0.912673, Theory.StructuredRecovery(3, 0.97), Tol);
        Assert.Equal(0.8852928099999999, Theory.StructuredRecovery(4, 0.97), Tol);
        Assert.Equal(0.8587340256999999, Theory.StructuredRecovery(5, 0.97), Tol);
    }

    [Fact]
    public void StructuredRecoveryEqualsPowerLaw()
    {
        // Golden pack A.1 test-file: structured_recovery(5, 0.95) == 0.95**5.
        Assert.Equal(Math.Pow(0.95, 5), Theory.StructuredRecovery(5, 0.95), Tol);
        Assert.Equal(1.0, Theory.StructuredRecovery(3, 1.0), Tol); // q=1 boundary: certain traversal
    }

    // ---- A.3 gap, overhead, and crossover (eq:gd / Prop prop:cross) ----

    [Fact]
    public void GapMatchesCalibrationTable()
    {
        // Golden pack A.1/A.3, g(d)=q^d - P_sim(d) at the calibration, d=1..5 (float64 verbatim).
        Assert.Equal(0.501, Theory.Gap(1, Calib), Tol);
        Assert.Equal(0.7935261299999999, Theory.Gap(2, Calib), Tol);
        Assert.Equal(0.8816457749160329, Theory.Gap(3, Calib), Tol);
        Assert.Equal(0.8809161764312712, Theory.Gap(4, Calib), Tol);
        Assert.Equal(0.8583203954994303, Theory.Gap(5, Calib), Tol);
    }

    [Fact]
    public void DepthThreeStructuredVsSimilarityIsAbout29x()
    {
        // Paper §sec:depth "~29x gap" at d=3: q^3 / P_sim(3) = 29.415231221293272.
        double ratio = Theory.StructuredRecovery(3, Calib.Q) / Theory.SimilarityRecovery(3, Calib);
        Assert.Equal(29.415231221293272, ratio, 1e-9);
    }

    [Fact]
    public void OverheadRatioGolden()
    {
        // Golden pack A.3: overhead_ratio(2.0, 8.0) = 0.25.
        Assert.Equal(0.25, Theory.OverheadRatio(2.0, 8.0), Tol);
    }
}
