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
}
