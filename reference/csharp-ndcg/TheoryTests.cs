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

    // ---- A.3 (cont.) crossover depth + win-region window (Prop prop:cross / bounds.py) ----

    [Fact]
    public void ClosedFormTauZeroCrossover()
    {
        // Golden pack A.3: d0 = 2*ln(q/s0)/ln(rho) - 1 = -2.629133632756974; integer d* = 1.
        Assert.Equal(-2.629133632756974, Theory.TauZeroCrossoverContinuous(Calib), 1e-12);
        Assert.Equal(1, Theory.TauZeroCrossoverDepth(Calib)); // closed-form d* = 1 (Prop prop:cross)
    }

    [Fact]
    public void StrictCrossoverDepthTracksTau()
    {
        // tau=0 -> structure already wins at d=1 (g(1)=0.501 > 0).
        Assert.Equal(1, Theory.FindCrossoverDepth(Calib, 0.0, 5));
        // Calibrated d*=2 window: tau in [g(1), g(2)) = [0.501, 0.79352613) (Prop prop:cross).
        Assert.Equal(2, Theory.FindCrossoverDepth(Calib, 0.501, 5)); // g(1) not strictly > 0.501
        Assert.Equal(2, Theory.FindCrossoverDepth(Calib, 0.79, 5));  // paper's "tau <= 0.79" endpoint
        Assert.Equal(2, Theory.FindCrossoverDepth(Calib, 0.7935261299999998, 5)); // just below g(2)
        // At tau == g(2), d=2 no longer strictly wins -> crossover slips to d=3.
        Assert.Equal(3, Theory.FindCrossoverDepth(Calib, 0.7935261299999999, 5));
        // No depth clears an overhead above the unimodal peak.
        Assert.Null(Theory.FindCrossoverDepth(Calib, 0.99, 5));
    }

    [Fact]
    public void WinRegionIsContiguous()
    {
        // Golden pack A.3: for q<1 the win region {d : g(d) > tau} is a contiguous interval.
        Assert.Equal(new[] { 1, 2, 3, 4, 5 }, Theory.WinRegionDepths(Calib, 0.0, 5));
        Assert.Equal(new[] { 2, 3, 4, 5 }, Theory.WinRegionDepths(Calib, 0.501, 5));
    }

    // ---- A.4 Fano floor / capacity (eq:fano, Thm thm:fano; capacity.py) ----

    [Fact]
    public void BinaryEntropyAndCapacityGoldens()
    {
        // Golden pack A.4: H_b(0.5) = 1.0 bit; H_b(0) = H_b(1) = 0.
        Assert.Equal(1.0, Theory.BinaryEntropy(0.5), Tol);
        Assert.Equal(0.0, Theory.BinaryEntropy(0.0));
        Assert.Equal(0.0, Theory.BinaryEntropy(1.0));
        // C(0.5) = 0 (pure noise carries nothing); C(0) = C(1) = 1.0.
        Assert.Equal(0.0, Theory.BscCapacity(0.5), Tol);
        Assert.Equal(1.0, Theory.BscCapacity(0.0), Tol);
        Assert.Equal(1.0, Theory.BscCapacity(1.0), Tol);
    }

    [Fact]
    public void ContextFreeFanoFloorIsOneMinusOneOverK()
    {
        // Golden pack A.4 / eq:fano: uniform 1/K case, floor = 1 - 1/K exactly (Fano tight at I=0).
        Assert.Equal(0.5, Theory.ContextFreeFloor(2), Tol);
        Assert.Equal(0.75, Theory.ContextFreeFloor(4), Tol);
        Assert.Equal(0.9, Theory.ContextFreeFloor(10), Tol);
        // Prior entropy = log2 K bits; Bayes accuracy = 1/K; floor = 1 - accuracy.
        Assert.Equal(1.0, Theory.UniformPriorEntropy(2), Tol);
        Assert.Equal(2.0, Theory.UniformPriorEntropy(4), Tol);
        Assert.Equal(0.5, Theory.UniformBayesAccuracy(2), Tol);
        Assert.Equal(1.0 - Theory.UniformBayesAccuracy(4), Theory.ContextFreeFloor(4), Tol);
    }

    // ---- B.1 use factor kappa = compliance/recall, every tab:realret row (eq:factor) ----

    [Fact]
    public void UseFactorFullPrecisionPerRow()
    {
        // Golden pack B.1: kappa = compliance/recall, full float64 (division is exact/deterministic).
        Assert.Equal(0.7031484257871063, Theory.UseFactor(0.469, 0.667)!.Value, 1e-15); // brief
        Assert.Equal(0.6393700787401575, Theory.UseFactor(0.406, 0.635)!.Value, 1e-15); // dense
        Assert.Equal(0.6439024390243903, Theory.UseFactor(0.396, 0.615)!.Value, 1e-15); // hybrid_rrf
        Assert.Equal(0.6374172185430463, Theory.UseFactor(0.385, 0.604)!.Value, 1e-15); // tfidf
        Assert.Equal(0.5695364238410596, Theory.UseFactor(0.344, 0.604)!.Value, 1e-15); // bm25
        Assert.Equal(0.6072041166380789, Theory.UseFactor(0.354, 0.583)!.Value, 1e-15); // rerank_ce
        Assert.Equal(0.5636998254799302, Theory.UseFactor(0.323, 0.573)!.Value, 1e-15); // raptor
    }

    [Fact]
    public void EveryRealRetRowRoundsToPublishedKappa()
    {
        // Paper tab:realret: each published 3-dp use factor equals round(compliance/recall, 3).
        foreach (var row in Theory.RealRet)
        {
            double? kappa = Theory.UseFactor(row.Compliance, row.Recall);
            Assert.NotNull(kappa); // every arm has positive recall, so kappa is defined
            if (row.PublishedKappa is double published)
                Assert.Equal(published, kappa!.Value, 3); // 3-dp rounding identity
        }
        // The typed store's kappa=0.703 is the only arm clearing the 0.56-0.64 similarity band.
        double brief = Theory.UseFactor(0.469, 0.667)!.Value;
        double bestSimilarity = Theory.UseFactor(0.396, 0.615)!.Value; // hybrid_rrf, top of the band
        Assert.True(brief > 0.64);
        Assert.True(bestSimilarity <= 0.645);
        Assert.True(brief > bestSimilarity);
    }

    [Fact]
    public void NoneArmKappaIsArithmeticOnlyNotPublished()
    {
        // Golden pack B.1: the `none` row's kappa (0.073/0.219 = 1/3) is deliberately UNPUBLISHED
        // (no retrieval mechanism to factorize). It is arithmetic only, never a paper value.
        var none = Theory.RealRet.Single(r => r.Arm == "none");
        Assert.Null(none.PublishedKappa);
        Assert.Equal(0.3333333333333333, Theory.UseFactor(none.Compliance, none.Recall)!.Value, 1e-15);
    }

    [Fact]
    public void UseFactorUndefinedWhenRecallZero()
    {
        // Golden pack E.1: recall=0 -> kappa undefined (null), no crash.
        Assert.Null(Theory.UseFactor(0.0, 0.0));
        Assert.Null(Theory.UseFactor(0.5, 0.0));
    }

    // ---- B.3 scorecard tally (tab:scorecard) ----

    [Fact]
    public void ScorecardTallyIsFortyOne()
    {
        // Paper tab:scorecard: 22 wins + 14 marginal + 5 losses = 41 axes.
        Assert.Equal(41, Theory.ScorecardTotal(22, 14, 5));
    }

    // ---- domain guards (golden-pack cross-cutting note 4) ----

    [Fact]
    public void DomainGuardsReject()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Theory.SimilarityRecovery(0, Calib));   // d >= 1
        Assert.Throws<ArgumentOutOfRangeException>(() => Theory.StructuredRecovery(1, 0.0));      // q > 0
        Assert.Throws<ArgumentOutOfRangeException>(() => Theory.StructuredRecovery(1, 1.5));      // q <= 1
        Assert.Throws<ArgumentOutOfRangeException>(() => Theory.BinaryEntropy(-0.1));             // p in [0,1]
        Assert.Throws<ArgumentOutOfRangeException>(() => Theory.BinaryEntropy(1.1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Theory.ContextFreeFloor(1));             // K >= 2
        Assert.Throws<ArgumentOutOfRangeException>(() => Theory.FindCrossoverDepth(Calib, -0.1, 5)); // tau >= 0
        Assert.Throws<ArgumentOutOfRangeException>(() => Theory.OverheadRatio(1.0, 0.0));         // V > 0
        Assert.Throws<ArgumentOutOfRangeException>(
            () => Theory.SimilarityRecovery(1, new SurvivalModel(0.0, 0.6, 0.9)));               // s0 > 0
    }
}
