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
}
