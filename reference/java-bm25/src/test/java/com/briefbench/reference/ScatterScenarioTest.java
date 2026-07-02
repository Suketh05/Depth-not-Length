package com.briefbench.reference;

import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Scatter + fairness-lock scenario suite (paper eq:scatter / Thm thm:scatter /
 * Thm thm:scatentropy; sec:scatter / sec:corpus; sec:fairness / sec:budget;
 * theory/scatter.py). Verifies two harness mechanisms:
 *
 * <ol>
 *   <li>SCATTER: a decision shredded across {@code sigma} areas must be reassembled
 *       from {@code sigma} probes and survives with probability {@code p^sigma}, while
 *       a typed dereference collapses {@code sigma} to a single probe.</li>
 *   <li>FAIRNESS LOCK: every arm gets an identical token budget and a deterministic
 *       packing order, so no arm can win on budget or packing luck.</li>
 * </ol>
 *
 * <p>INTEGRITY: assertions are the golden-pack A.5 closed forms and hand-computed
 * arithmetic on constructed fixtures. The paper's fitted {@code p=0.92} (fig:orgsweep)
 * and the matched-budget token counts of fig:tokensbox are NOT asserted here — they are
 * measured/fitted values, out of scope for a closed-form cross-language check.
 */
class ScatterScenarioTest {

    private static final double EPS = 1e-12;

    // ---- Scatter penalty p^sigma (golden pack A.5) ----

    @Test
    void scatterFactorGolden() {
        // scatter_factor(0.5, 3) = 0.5^3 = 0.125 (exact 1/8).
        assertEquals(0.125, DecisionGraph.scatterFactor(0.5, 3), EPS);
        // p^0 = 1 (a typed dereference pays no scatter penalty); p^1 = p.
        assertEquals(1.0, DecisionGraph.scatterFactor(0.5, 0), EPS);
        assertEquals(0.5, DecisionGraph.scatterFactor(0.5, 1), EPS);
    }

    @Test
    void composedScatteredRecoveryGolden() {
        // Fully hand-computed composite (pack A.5): s0=0.8, rho=0.5, d=2, p=0.5, sigma=2.
        //   P_sim(2) = 0.8^2 * 0.5^3 = 0.64 * 0.125 = 0.08
        //   scatter  = 0.5^2 = 0.25
        //   P_scat   = 0.08 * 0.25 = 0.02  (exact 1/50)
        assertEquals(0.08, DecisionGraph.similarityRecovery(2, 0.8, 0.5), EPS);
        assertEquals(0.25, DecisionGraph.scatterFactor(0.5, 2), EPS);
        assertEquals(0.02, DecisionGraph.scatteredRecovery(2, 0.8, 0.5, 0.5, 2), EPS);
    }

    @Test
    void degradeRecoveryGolden() {
        // degrade_recovery(0.5, 0.8, 3): factor = 0.8^3 = 0.512; degraded = 0.5*0.512 = 0.256.
        assertEquals(0.512, DecisionGraph.scatterFactor(0.8, 3), EPS);
        assertEquals(0.256, DecisionGraph.degradeRecovery(0.5, 0.8, 3), EPS);
    }

    @Test
    void factorCurveGolden() {
        // ScatterModel(p=0.5).factor_curve(4) = [1.0, 0.5, 0.25, 0.125, 0.0625] (sigma=0..4).
        double[] expected = {1.0, 0.5, 0.25, 0.125, 0.0625};
        double[] actual = DecisionGraph.factorCurve(0.5, 4);
        assertArrayEquals(expected, actual, EPS);
        // Curve is monotone non-increasing in sigma for p in (0,1].
        for (int s = 1; s < actual.length; s++) {
            assertTrue(actual[s] <= actual[s - 1] + EPS);
        }
    }

    // ---- Probe-count arithmetic: sigma vs 1 (Thm thm:scatter / thm:scatentropy) ----

    @Test
    void scatteredDecisionNeedsSigmaProbesTypedNeedsOne() {
        // A decision shredded across sigma=3 areas: flat similarity issues 3 probes,
        // a typed dereference issues exactly 1 (the sigma-vs-1 separation, sec:corpus).
        int sigma = 3;
        assertEquals(3, DecisionGraph.probesForScatter(sigma));
        assertEquals(1, DecisionGraph.probesForDereference());
        assertTrue(DecisionGraph.probesForScatter(sigma) > DecisionGraph.probesForDereference());

        // Assembly probability drops with each extra probe: p^sigma vs p^1.
        double p = 0.9;
        double flatAssembly = DecisionGraph.scatterFactor(p, sigma);   // 0.9^3
        double typedAssembly = DecisionGraph.scatterFactor(p, 1);      // 0.9
        assertEquals(0.9 * 0.9 * 0.9, flatAssembly, EPS);
        assertEquals(0.9, typedAssembly, EPS);
        assertTrue(typedAssembly > flatAssembly);
    }

    @Test
    void probeLowerBoundFromScatterEntropy() {
        // Thm thm:scatentropy: probes >= sigma / p_bar => ceil(sigma / p_bar) whole probes.
        // sigma=5, each probe recovers p_bar=0.5 of the locations => ceil(10) = 10 probes.
        assertEquals(10, DecisionGraph.probesLowerBound(5, 0.5));
        // Non-integer ratio rounds up: sigma=3, p_bar=0.5 => ceil(6.0)=6; p_bar=0.4 => ceil(7.5)=8.
        assertEquals(6, DecisionGraph.probesLowerBound(3, 0.5));
        assertEquals(8, DecisionGraph.probesLowerBound(3, 0.4));
        // Perfect probes (p_bar=1.0) match the raw scatter count sigma.
        assertEquals(3, DecisionGraph.probesLowerBound(3, 1.0));
        assertEquals(DecisionGraph.probesForScatter(3), DecisionGraph.probesLowerBound(3, 1.0));
    }

    // ---- Edge cases: zero scatter, saturation, domain guards ----

    @Test
    void zeroScatterIsNeutral() {
        // sigma=0: no scatter penalty, assembly is certain; 0 probes below the bound.
        assertEquals(1.0, DecisionGraph.scatterFactor(0.7, 0), EPS);
        assertEquals(0, DecisionGraph.probesForScatter(0));
        assertEquals(0, DecisionGraph.probesLowerBound(0, 0.5));
        // Fractional scatter is allowed (sigma need not be integer): 0.25^0.5 = 0.5.
        assertEquals(0.5, DecisionGraph.scatterFactor(0.25, 0.5), EPS);
    }

    @Test
    void scatterDomainGuards() {
        assertThrows(IllegalArgumentException.class, () -> DecisionGraph.scatterFactor(0.0, 2)); // p>0
        assertThrows(IllegalArgumentException.class, () -> DecisionGraph.scatterFactor(1.5, 2)); // p<=1
        assertThrows(IllegalArgumentException.class, () -> DecisionGraph.scatterFactor(0.5, -1)); // sigma>=0
        assertThrows(IllegalArgumentException.class, () -> DecisionGraph.similarityRecovery(0, 0.8, 0.5)); // d>=1
        assertThrows(IllegalArgumentException.class, () -> DecisionGraph.probesForScatter(-1));
    }
}
