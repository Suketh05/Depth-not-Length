package com.briefbench.reference;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Cross-language verification of the recovery laws in {@link Recovery} against the
 * canonical golden pack (section A.1 similarity ceiling, A.2 structured recovery).
 *
 * <p>Every asserted literal is transcribed VERBATIM from the golden pack, which
 * derives from the Python reference ({@code tests/test_theory_recovery.py}) and the
 * paper's synthetic operating point s0=0.70, rho=0.67, q=0.97 (sec:depth /
 * eq:simceiling / thm:struct). No value here was produced by running this Java code:
 * the tolerance {@code TOL = 1e-12} covers only last-ULP {@code Math.pow} differences
 * between the two runtimes (the golden pack's recommended float64 tolerance).
 */
class RecoveryTest {

    /** Golden-pack float64 tolerance; three table cells differ from Python by 1 ULP. */
    private static final double TOL = 1e-12;

    private static final Recovery.Model CAL = Recovery.Model.calibration();

    // Golden pack A.1, calibration s0=0.70, rho=0.67, q=0.97. Columns:
    //   d, s0^d, rho^{d(d+1)/2}, P_sim(d) = s0^d * rho^T
    private static final double[] S0_POW = {
        0.7, 0.48999999999999994, 0.3429999999999999, 0.24009999999999995, 0.16806999999999994
    };
    private static final double[] RHO_POW = {
        0.67, 0.30076300000000006, 0.09045838216900003, 0.018228378045517624,
        0.0024610590859140942
    };
    private static final double[] P_SIM = {
        0.469, 0.14737387000000002, 0.031027225083967002, 0.004376633568728781,
        0.0004136302005695817
    };

    @Test
    void similarityCeilingMatchesGoldenTableD1toD5() {
        for (int d = 1; d <= 5; d++) {
            int i = d - 1;
            // Intermediate factors (golden pack A.1 columns 2-3).
            assertEquals(S0_POW[i], Math.pow(CAL.s0(), d), TOL, "s0^" + d);
            assertEquals(RHO_POW[i], Math.pow(CAL.rho(), Recovery.triangular(d)), TOL,
                    "rho^T at d=" + d);
            // The composed similarity ceiling P_sim(d) = s0^d * rho^{d(d+1)/2}.
            assertEquals(P_SIM[i], Recovery.similarityRecovery(d, CAL), TOL, "P_sim(" + d + ")");
        }
    }

    @Test
    void similarityCeilingD3IsThePaperHeadlineValue() {
        // Paper sec:depth: at d=3 similarity recovery "~ 0.031" (a ~29x gap vs traversal).
        assertEquals(0.031027225083967002, Recovery.similarityRecovery(3, CAL), TOL);
    }

    @Test
    void triangularExponentIsTheDthTriangleNumber() {
        // T = d(d+1)/2 = 1, 3, 6, 10, 15 for d = 1..5 (the sum 1+2+...+d).
        double[] tri = {1, 3, 6, 10, 15};
        for (int d = 1; d <= 5; d++) {
            assertEquals(tri[d - 1], Recovery.triangular(d), 0.0, "T at d=" + d);
        }
    }

    // ---- A.2 structured recovery P_struct(d) = q^d ----

    // Golden pack A.1/A.2, q = 0.97: q^1..q^5.
    private static final double[] P_STRUCT = {
        0.97, 0.9409, 0.912673, 0.8852928099999999, 0.8587340256999999
    };

    @Test
    void structuredRecoveryMatchesGoldenTableD1toD5() {
        for (int d = 1; d <= 5; d++) {
            assertEquals(P_STRUCT[d - 1], Recovery.structuredRecovery(d, 0.97), TOL,
                    "q^" + d);
            // The model overload must agree with the raw-q form.
            assertEquals(Recovery.structuredRecovery(d, 0.97),
                    Recovery.structuredRecovery(d, CAL), 0.0, "model overload at d=" + d);
        }
    }

    @Test
    void structuredRecoveryEqualsQtoTheD() {
        // Golden pack A.2: structured_recovery(5, 0.95) == 0.95**5 (exact identity;
        // the reference's `**` and Java Math.pow share libm pow, so equality is exact).
        assertEquals(Math.pow(0.95, 5), Recovery.structuredRecovery(5, 0.95), 0.0);
        // And to full float64 precision it agrees with the repeated per-edge product.
        assertEquals(0.95 * 0.95 * 0.95 * 0.95 * 0.95,
                Recovery.structuredRecovery(5, 0.95), TOL);
    }

    @Test
    void similarityCeilingEqualsExplicitPerHopProduct() {
        // Golden pack A.2: closed form checked against the explicit product
        // prod_{i=1..d} s0 * rho^i for s0=0.9, rho=0.6 (an independent derivation,
        // not a re-run of the closed form).
        Recovery.Model m = new Recovery.Model(0.9, 0.6, 0.95);
        for (int d = 1; d <= 8; d++) {
            double product = 1.0;
            for (int i = 1; i <= d; i++) {
                product *= 0.9 * Math.pow(0.6, i);
            }
            assertEquals(product, Recovery.similarityRecovery(d, m), TOL, "d=" + d);
        }
    }

    @Test
    void deepChainUnderflowsWhileLogStaysFinite() {
        // Golden pack A.2: s0=0.5, rho=0.3 => similarity_recovery(60) underflows to
        // 0.0 (float64) while log_similarity_recovery(60) is finite.
        Recovery.Model m = new Recovery.Model(0.5, 0.3, 0.9);
        assertEquals(0.0, Recovery.similarityRecovery(60, m), 0.0);
        double logVal = Recovery.logSimilarityRecovery(60, m);
        assertTrue(Double.isFinite(logVal), "log form must stay finite: " + logVal);
        assertTrue(logVal < 0.0, "deep-chain log-recovery must be strongly negative");
    }

    @Test
    void logSimilarityRecoveryEqualsLogOfDirectFormWhereNoUnderflow() {
        // For shallow depths the two forms must agree exactly (log of the product).
        for (int d = 1; d <= 5; d++) {
            assertEquals(Math.log(Recovery.similarityRecovery(d, CAL)),
                    Recovery.logSimilarityRecovery(d, CAL), TOL, "d=" + d);
        }
    }

    // ---- monotonicity / saturation properties ----

    @Test
    void similarityRecoveryStrictlyDecreasesInDepth() {
        // Super-geometric fall-off: P_sim(d+1) < P_sim(d) for the calibration.
        for (int d = 1; d <= 10; d++) {
            assertTrue(Recovery.similarityRecovery(d + 1, CAL)
                    < Recovery.similarityRecovery(d, CAL), "monotone at d=" + d);
        }
    }

    @Test
    void structuredRecoveryStrictlyDecreasesForSubUnitFidelity() {
        for (int d = 1; d <= 20; d++) {
            assertTrue(Recovery.structuredRecovery(d + 1, 0.97)
                    < Recovery.structuredRecovery(d, 0.97), "monotone at d=" + d);
        }
    }

    @Test
    void structuralAdvantageRatioStrictlyGrowsWithDepth() {
        // P_struct/P_sim -> inf as d -> inf: the ratio must be strictly increasing.
        double prev = Double.NEGATIVE_INFINITY;
        for (int d = 1; d <= 8; d++) {
            double ratio = Recovery.structuredRecovery(d, CAL) / Recovery.similarityRecovery(d, CAL);
            assertTrue(ratio > prev, "advantage ratio grows at d=" + d);
            prev = ratio;
        }
    }

    @Test
    void perfectFidelitySaturatesStructuredRecoveryAtOne() {
        // Chance/saturation floor: q = 1 keeps every-depth structured recovery at 1.0.
        for (int d = 1; d <= 12; d++) {
            assertEquals(1.0, Recovery.structuredRecovery(d, 1.0), 0.0, "q=1 at d=" + d);
        }
    }

    // ---- domain guards (golden pack cross-cutting note 4) ----

    @Test
    void depthBelowOneIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> Recovery.similarityRecovery(0, CAL));
        assertThrows(IllegalArgumentException.class, () -> Recovery.logSimilarityRecovery(0, CAL));
        assertThrows(IllegalArgumentException.class, () -> Recovery.structuredRecovery(-1, 0.9));
    }

    @Test
    void fidelityOutsideUnitIntervalIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> Recovery.structuredRecovery(2, 0.0));
        assertThrows(IllegalArgumentException.class, () -> Recovery.structuredRecovery(2, 1.5));
        assertEquals(1.0, Recovery.structuredRecovery(1, 1.0), 0.0); // q = 1 is the closed endpoint
    }

    @Test
    void modelRejectsIllegalParameters() {
        assertThrows(IllegalArgumentException.class, () -> new Recovery.Model(0.0, 0.6, 0.9));
        assertThrows(IllegalArgumentException.class, () -> new Recovery.Model(0.7, 0.0, 0.9));
        assertThrows(IllegalArgumentException.class, () -> new Recovery.Model(0.7, 0.6, 0.0));
        assertThrows(IllegalArgumentException.class, () -> new Recovery.Model(0.7, 0.6, 1.5));
    }

    @Test
    void calibrationModelHasThePaperOperatingPoint() {
        assertEquals(0.70, CAL.s0(), 0.0);
        assertEquals(0.67, CAL.rho(), 0.0);
        assertEquals(0.97, CAL.q(), 0.0);
        assertTrue(CAL.isDecaying());
        assertFalse(new Recovery.Model(0.7, 1.0, 0.9).isDecaying());
    }
}
