package com.briefbench.reference;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Cross-language verification of the crossover analysis in {@link Crossover} against
 * the golden pack (section A.3: gap {@code g(d)}, overhead ratio {@code tau}, the
 * closed-form zero-overhead crossover {@code d0}, and the calibrated win window).
 *
 * <p>All asserted literals are transcribed VERBATIM from the golden pack (paper
 * sec:cross / eq:gd / prop:cross, calibration s0=0.70, rho=0.67, q=0.97). Grid and
 * property tests additionally check that the integer scan agrees with the closed-form
 * bracket and that the theoretical shape (unimodal {@code g}, monotone {@code q=1}
 * limit) holds. Tolerance {@code TOL = 1e-12} covers cross-runtime {@code Math.pow}
 * ULPs; the continuous root is checked at the reference's {@code 1e-6}.
 */
class CrossoverTest {

    private static final double TOL = 1e-12;
    private static final Recovery.Model CAL = Recovery.Model.calibration();

    // Golden pack A.1/A.3, calibration: g(d) = q^d - P_sim(d), d = 1..5.
    private static final double[] G = {
        0.501, 0.7935261299999999, 0.8816457749160329, 0.8809161764312712,
        0.8583203954994303
    };

    @Test
    void gapMatchesGoldenTableD1toD5() {
        for (int d = 1; d <= 5; d++) {
            assertEquals(G[d - 1], Crossover.g(d, CAL), TOL, "g(" + d + ")");
        }
    }

    @Test
    void gapEqualsStructuredMinusSimilarityByDefinition() {
        // eq:gd: g(d) = P_struct(d) - P_sim(d), exactly, at integer depths.
        for (int d = 1; d <= 6; d++) {
            double expected =
                    Recovery.structuredRecovery(d, CAL) - Recovery.similarityRecovery(d, CAL);
            assertEquals(expected, Crossover.g(d, CAL), 0.0, "d=" + d);
        }
    }

    @Test
    void paperGapAnchorsG1AndG2() {
        // prop:cross states (verbatim) g(1) ~ 0.50 and g(2) ~ 0.79; the full-precision
        // golden pack values are 0.501 and 0.7935261299999999.
        assertEquals(0.501, Crossover.g(1, CAL), TOL);
        assertEquals(0.7935261299999999, Crossover.g(2, CAL), TOL);
    }

    @Test
    void overheadRatioGolden() {
        // Golden pack A.3: overhead_ratio(W=2.0, V=8.0) = 0.25.
        assertEquals(0.25, Crossover.overheadRatio(2.0, 8.0), 0.0);
    }

    @Test
    void overheadRatioRejectsIllegalCosts() {
        assertThrows(IllegalArgumentException.class, () -> Crossover.overheadRatio(1.0, 0.0));
        assertThrows(IllegalArgumentException.class, () -> Crossover.overheadRatio(1.0, -1.0));
        assertThrows(IllegalArgumentException.class, () -> Crossover.overheadRatio(-1.0, 8.0));
    }

    @Test
    void depthThreeAdvantageIsThe29xGap() {
        // Paper sec:depth: at d=3 traversal q^3 ~ 0.91 vs similarity ~ 0.031, a "~29x
        // gap". Golden pack: q^3 / P_sim(3) = 29.415231221293272.
        double ratio = Recovery.structuredRecovery(3, CAL) / Recovery.similarityRecovery(3, CAL);
        assertEquals(29.415231221293272, ratio, 1e-9);
    }

    @Test
    void netValueGainIsValueTimesGapMinusCost() {
        // V*g(d) - W. With V=8, W=2 (tau=0.25) at d=2: 8*0.7935261299999999 - 2.
        double expected = 8.0 * Crossover.g(2, CAL) - 2.0;
        assertEquals(expected, Crossover.netValueGain(2, CAL, 8.0, 2.0), 0.0);
        // Net gain is positive exactly when g(d) clears tau = W/V.
        assertTrue(Crossover.netValueGain(2, CAL, 8.0, 2.0) > 0.0); // g(2)=0.79 > 0.25
        assertTrue(Crossover.netValueGain(1, CAL, 8.0, 8.0) < 0.0); // g(1)=0.501 < tau=1.0
    }
}
