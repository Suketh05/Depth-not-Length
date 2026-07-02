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

    // ---- A.3 closed-form crossover d0 and the calibrated win window ----

    @Test
    void zeroOverheadCrossoverIsTheGoldenD0() {
        // Golden pack A.3: d0 = 2*ln(0.97/0.70)/ln(0.67) - 1 = -2.629133632756974.
        assertEquals(-2.629133632756974, Crossover.tauZeroCrossover(CAL), TOL);
    }

    @Test
    void closedFormCrossoverIsDepthOneAtTheCalibration() {
        // d_lower = max(1, floor(d0) + 1) = max(1, floor(-2.629)+1) = max(1, -2) = 1:
        // structure already wins at d = 1 since g(1) = 0.501 > 0 (prop:cross).
        Crossover.CrossoverBounds b = Crossover.bracketCrossoverDepth(CAL, 0.0);
        assertEquals(1, b.dLower());
        Crossover.CrossoverResult r = Crossover.findCrossoverDepth(CAL, 0.0);
        assertTrue(r.exists());
        assertEquals(1, r.dStar());
    }

    @Test
    void continuousRootHasZeroGap() {
        // d0 is the depth where P_struct(d) = P_sim(d), so g(d0) = 0 identically
        // (even though d0 < 0 for this calibration).
        double d0 = Crossover.tauZeroCrossover(CAL);
        assertEquals(0.0, Crossover.g(d0, CAL), 1e-12);
    }

    @Test
    void calibratedCrossoverIsDepthTwoInTheGoldenWindow() {
        // Golden pack A.3: with strict-win semantics, d* = 2 exactly when
        // g(1) <= tau < g(2), i.e. tau in [0.501, 0.7935261299999999).
        // Points strictly inside the window (paper's "0.50 < tau <= 0.79"):
        assertEquals(2, Crossover.findCrossoverDepth(CAL, 0.6).dStar());
        assertEquals(2, Crossover.findCrossoverDepth(CAL, 0.79).dStar());
        // Just below g(1): structure already wins at d = 1.
        assertEquals(1, Crossover.findCrossoverDepth(CAL, 0.5).dStar());
    }

    @Test
    void winWindowEndpointsArePinnedByG1AndG2() {
        // The half-open window [g(1), g(2)) is pinned exactly by the computed edges:
        // at tau = g(1) depth 1 no longer strictly wins so d* = 2; a hair below flips
        // to d* = 1. At tau = g(2), depth 2 no longer wins so d* = 3.
        double g1 = Crossover.g(1, CAL); // = 0.501
        double g2 = Crossover.g(2, CAL); // = 0.7935261299999999
        assertEquals(0.501, g1, TOL);
        assertEquals(0.7935261299999999, g2, TOL);

        assertEquals(2, Crossover.findCrossoverDepth(CAL, g1).dStar());
        assertEquals(1, Crossover.findCrossoverDepth(CAL, Math.nextDown(g1)).dStar());
        assertEquals(3, Crossover.findCrossoverDepth(CAL, g2).dStar());
        assertEquals(2, Crossover.findCrossoverDepth(CAL, Math.nextDown(g2)).dStar());
    }

    @Test
    void zeroOverheadCrossoverRejectsNonDecayingModel() {
        // rho >= 1: the quadratic depth-ceiling argument does not apply.
        assertThrows(IllegalArgumentException.class,
                () -> Crossover.tauZeroCrossover(new Recovery.Model(0.7, 1.0, 0.9)));
        assertThrows(IllegalArgumentException.class,
                () -> Crossover.bracketCrossoverDepth(new Recovery.Model(0.7, 1.2, 0.9), 0.1));
    }

    // ---- grid consistency: the integer scan agrees with the closed-form bracket ----

    private static final double[] GRID_S0 = {0.5, 0.7, 0.9};
    private static final double[] GRID_RHO = {0.3, 0.6, 0.8};
    private static final double[] GRID_Q = {0.9, 0.95, 0.99};

    @Test
    void continuousRootHasZeroGapAcrossGrid() {
        for (double s0 : GRID_S0) {
            for (double rho : GRID_RHO) {
                for (double q : GRID_Q) {
                    Recovery.Model m = new Recovery.Model(s0, rho, q);
                    double d0 = Crossover.tauZeroCrossover(m);
                    assertEquals(0.0, Crossover.g(d0, m), 1e-9,
                            "g(d0) at s0=" + s0 + " rho=" + rho + " q=" + q);
                }
            }
        }
    }

    @Test
    void zeroOverheadScanMatchesClosedFormLowerBoundAcrossGrid() {
        // At tau = 0 the smallest winning integer depth is exactly floor(d0)+1 (clipped
        // at 1). The integer scan and the O(1) bracket must return the same d*.
        for (double s0 : GRID_S0) {
            for (double rho : GRID_RHO) {
                for (double q : GRID_Q) {
                    Recovery.Model m = new Recovery.Model(s0, rho, q);
                    Crossover.CrossoverBounds b = Crossover.bracketCrossoverDepth(m, 0.0);
                    Crossover.CrossoverResult r = Crossover.findCrossoverDepth(m, 0.0);
                    int closed = Math.max(1, (int) Math.floor(Crossover.tauZeroCrossover(m)) + 1);
                    assertEquals(closed, b.dLower(), "bracket d_lower");
                    assertTrue(r.exists(), "structure must eventually win at tau=0");
                    assertEquals(closed, r.dStar(),
                            "scan d* at s0=" + s0 + " rho=" + rho + " q=" + q);
                }
            }
        }
    }

    @Test
    void closedFormBracketContainsScanResultAcrossGrid() {
        // For nonzero overheads the O(1) bracket must bracket the scan's d* and its
        // win window must contain the scan's win region (bounds.py contract).
        double[] taus = {0.05, 0.2, 0.4, 0.6};
        for (double s0 : GRID_S0) {
            for (double rho : GRID_RHO) {
                for (double q : GRID_Q) {
                    Recovery.Model m = new Recovery.Model(s0, rho, q);
                    for (double tau : taus) {
                        Crossover.CrossoverResult r = Crossover.findCrossoverDepth(m, tau);
                        Crossover.CrossoverBounds b = Crossover.bracketCrossoverDepth(m, tau);
                        if (!r.exists()) {
                            continue; // nothing to bracket
                        }
                        String at = "s0=" + s0 + " rho=" + rho + " q=" + q + " tau=" + tau;
                        assertTrue(b.hasWinWindow(), "window must exist when scan wins @ " + at);
                        assertTrue(b.dLower() <= r.dStar(), "d_lower <= d* @ " + at);
                        assertTrue(r.dStar() <= b.dUpper(), "d* <= d_upper @ " + at);
                        assertTrue(b.winLo() <= r.winLo(), "window contains region lo @ " + at);
                        assertTrue(b.winHi() >= r.winHi(), "window contains region hi @ " + at);
                    }
                }
            }
        }
    }

    // ---- shape properties: unimodal g (q<1), monotone q=1 limit, edge cases ----

    @Test
    void gainIsUnimodalAtTheCalibration() {
        // For q < 1, g rises, peaks, then falls back toward 0 (an interval win, not a
        // half-line). The calibration peaks at d = 3 (g = 0.8816...).
        Crossover.CrossoverResult r = Crossover.findCrossoverDepth(CAL, 0.0);
        assertEquals(3, r.peakDepth());
        // Strictly rising before the peak, strictly falling after.
        for (int d = 1; d < 3; d++) {
            assertTrue(Crossover.g(d + 1, CAL) > Crossover.g(d, CAL), "rising at d=" + d);
        }
        for (int d = 3; d < 12; d++) {
            assertTrue(Crossover.g(d + 1, CAL) < Crossover.g(d, CAL), "falling at d=" + d);
        }
    }

    @Test
    void gCurveMirrorsGAtEveryScannedDepth() {
        Crossover.CrossoverResult r = Crossover.findCrossoverDepth(CAL, 0.3, 20);
        assertEquals(20, r.gCurve().length);
        for (int i = 0; i < 20; i++) {
            assertEquals(Crossover.g(i + 1, CAL), r.gCurve()[i], 0.0, "gCurve at d=" + (i + 1));
        }
    }

    @Test
    void perfectFidelityGivesAMonotoneHalfLineWin() {
        // q = 1: g(d) = 1 - P_sim(d) is strictly increasing, so the win is the
        // half-line [d*, maxDepth] (Proposition 2 holds only in this limit).
        Recovery.Model m = new Recovery.Model(0.7, 0.67, 1.0);
        for (int d = 1; d < 10; d++) {
            assertTrue(Crossover.g(d + 1, m) > Crossover.g(d, m), "monotone g at d=" + d);
        }
        int maxDepth = 32;
        Crossover.CrossoverResult r = Crossover.findCrossoverDepth(m, 0.5, maxDepth);
        assertTrue(r.exists());
        assertEquals(maxDepth, r.winHi(), "win region runs to the scan edge");
        Crossover.CrossoverBounds b = Crossover.bracketCrossoverDepth(m, 0.5, maxDepth);
        assertTrue(b.hasWinWindow());
        assertEquals(maxDepth, b.winHi(), "closed-form window is the half-line");
    }

    @Test
    void noCrossoverWhenOverheadExceedsPeakGain() {
        // Max gain for the calibration is g(3) = 0.8816...; an overhead above it is
        // never cleared, so no crossover exists (saturation of the win region).
        Crossover.CrossoverResult r = Crossover.findCrossoverDepth(CAL, 0.9);
        assertFalse(r.exists());
        assertEquals(-1, r.dStar());
        assertTrue(Double.isNaN(r.dStarContinuous()));
        assertEquals(3, r.peakDepth(), "peak is still reported when nobody wins");
    }

    @Test
    void noCrossoverWhenOverheadExceedsStructuredTerm() {
        // tau >= q: q^d <= q <= tau for every depth, so structure can never clear the
        // overhead and the closed-form window is empty.
        Crossover.CrossoverBounds b = Crossover.bracketCrossoverDepth(CAL, 0.98);
        assertFalse(b.hasWinWindow());
        assertFalse(Crossover.findCrossoverDepth(CAL, 0.98).exists());
    }

    @Test
    void continuousRootLandsOnTauOnTheRisingEdge() {
        // For a calibrated overhead the continuous crossover sits on the rising edge
        // in (d*-1, d*] and satisfies g(d_continuous) ~ tau (reference tol 1e-6).
        double tau = 0.6;
        Crossover.CrossoverResult r = Crossover.findCrossoverDepth(CAL, tau);
        assertEquals(2, r.dStar());
        assertTrue(r.dStarContinuous() > 1.0 && r.dStarContinuous() <= 2.0,
                "continuous root on rising edge: " + r.dStarContinuous());
        assertEquals(tau, Crossover.g(r.dStarContinuous(), CAL), 1e-6);
    }

    @Test
    void continuousRootPinsToDStarWhenAlreadyWinningAtDepthOne() {
        // When structure wins at d = 1 there is no sub-integer rising edge, so the
        // continuous crossover collapses to d* = 1 exactly (matches the reference).
        Crossover.CrossoverResult r = Crossover.findCrossoverDepth(CAL, 0.0);
        assertEquals(1, r.dStar());
        assertEquals(1.0, r.dStarContinuous(), 0.0);
    }

    @Test
    void findCrossoverRejectsIllegalArguments() {
        assertThrows(IllegalArgumentException.class,
                () -> Crossover.findCrossoverDepth(CAL, -0.1));
        assertThrows(IllegalArgumentException.class,
                () -> Crossover.findCrossoverDepth(CAL, 0.1, 0));
        assertThrows(IllegalArgumentException.class,
                () -> Crossover.bracketCrossoverDepth(CAL, -0.1));
    }
}
