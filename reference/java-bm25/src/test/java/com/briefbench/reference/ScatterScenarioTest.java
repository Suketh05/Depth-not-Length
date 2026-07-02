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

    @Test
    void flatSimilarityOverDepthChainPaysSigmaEqualsD() {
        // Thm thm:scatter: flat similarity split over a depth-d chain pays sigma=d, so
        // its assembly probability is p^d; a typed dereference of the same chain pays
        // sigma=1 (probability p). We sweep d=1..4 at the calibration rho=0.67 stand-in
        // p=0.67 and check the flat penalty equals p^d while typed stays p.
        double p = 0.67;
        for (int d = 1; d <= 4; d++) {
            assertEquals(d, DecisionGraph.scatterForFlatChain(d));
            double flat = DecisionGraph.scatterFactor(p, DecisionGraph.scatterForFlatChain(d));
            assertEquals(Math.pow(p, d), flat, EPS);
            double typed = DecisionGraph.scatterFactor(p, DecisionGraph.probesForDereference());
            assertEquals(p, typed, EPS);
            if (d >= 2) {
                // Deeper chains strictly widen the typed-vs-flat gap.
                assertTrue(typed > flat);
            }
        }
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

    // ---- Fairness lock: equal budgets + deterministic packing (sec:fairness) ----

    /**
     * The five constructed context candidates. Costs chosen so the greedy pack is
     * non-trivial (an item is skipped, then a later smaller item still fits exactly).
     */
    private static List<DecisionGraph.Item> candidates() {
        return new ArrayList<>(Arrays.asList(
                new DecisionGraph.Item("A", 500),
                new DecisionGraph.Item("B", 300),
                new DecisionGraph.Item("C", 300),
                new DecisionGraph.Item("D", 400),
                new DecisionGraph.Item("E", 200)));
    }

    @Test
    void everyArmGetsAnIdenticalBudget() {
        // The fairness lock: identical token budget across all arms (matched to ~1% in
        // fig:tokensbox; here we assert exact equality of the ALLOCATED budgets).
        List<String> arms = Arrays.asList("none", "bm25", "dense", "brief_graph_3hop");
        Map<String, Integer> budgets = DecisionGraph.equalBudgets(arms, 1500);
        assertEquals(arms.size(), budgets.size());
        for (String arm : arms) {
            assertEquals(1500, budgets.get(arm), "arm " + arm + " must get the locked budget");
        }
        // All values equal => zero spread across arms.
        assertEquals(1, new HashSet<>(budgets.values()).size());
    }

    @Test
    void greedyPackIsDeterministicGolden() {
        // Hand-computed greedy pack of candidates() under budget=1000, insertion order:
        //   A(500) -> used 500; B(300) -> used 800; C(300) -> 1100>1000 SKIP;
        //   D(400) -> 1200>1000 SKIP; E(200) -> 800+200=1000 FIT.
        //   packed = [A, B, E], used = 1000.
        DecisionGraph.PackResult r = DecisionGraph.pack(candidates(), 1000);
        assertEquals(Arrays.asList("A", "B", "E"), r.packedIds);
        assertEquals(1000, r.usedTokens);
        assertTrue(r.usedTokens <= 1000);
    }

    @Test
    void identicalArmsPackIdenticallyUnderTheSameBudget() {
        // Fairness: two arms handed the same candidate list and the same locked budget
        // pack byte-for-byte identically (packing adds no hidden per-arm advantage).
        int budget = 1000;
        DecisionGraph.PackResult armA = DecisionGraph.pack(candidates(), budget);
        DecisionGraph.PackResult armB = DecisionGraph.pack(candidates(), budget);
        assertEquals(armA.packedIds, armB.packedIds);
        assertEquals(armA.usedTokens, armB.usedTokens);
    }

    @Test
    void canonicalOrderMakesPackingSeedIndependent() {
        // Packing-order determinism: an arbitrary arrival order must not change the pack.
        // canonicalOrder (tokens asc, then id) normalises to [E,B,C,D,A]; packing that
        // under budget=1000 gives [E,B,C] used=800. A seeded shuffle of the SAME items
        // canonicalises to the SAME order and packs identically.
        List<DecisionGraph.Item> canon = DecisionGraph.canonicalOrder(candidates());
        List<String> canonIds = new ArrayList<>();
        for (DecisionGraph.Item it : canon) {
            canonIds.add(it.id);
        }
        assertEquals(Arrays.asList("E", "B", "C", "D", "A"), canonIds);

        DecisionGraph.PackResult packedCanon = DecisionGraph.pack(canon, 1000);
        assertEquals(Arrays.asList("E", "B", "C"), packedCanon.packedIds);
        assertEquals(800, packedCanon.usedTokens);

        List<DecisionGraph.Item> shuffled = candidates();
        Collections.shuffle(shuffled, new Random(1234567L));
        DecisionGraph.PackResult packedShuffled = DecisionGraph.pack(
                DecisionGraph.canonicalOrder(shuffled), 1000);
        assertEquals(packedCanon.packedIds, packedShuffled.packedIds);
        assertEquals(packedCanon.usedTokens, packedShuffled.usedTokens);
    }

    @Test
    void budgetEdgeCases() {
        // Zero budget admits nothing; nothing is used.
        DecisionGraph.PackResult zero = DecisionGraph.pack(candidates(), 0);
        assertTrue(zero.packedIds.isEmpty());
        assertEquals(0, zero.usedTokens);
        // A single item larger than the whole budget is skipped, not partially admitted.
        List<DecisionGraph.Item> oneBig = Collections.singletonList(new DecisionGraph.Item("X", 900));
        DecisionGraph.PackResult tight = DecisionGraph.pack(oneBig, 500);
        assertTrue(tight.packedIds.isEmpty());
        assertEquals(0, tight.usedTokens);
        // Negative budget / negative token cost are rejected.
        assertThrows(IllegalArgumentException.class, () -> DecisionGraph.pack(candidates(), -1));
        assertThrows(IllegalArgumentException.class, () -> new DecisionGraph.Item("Z", -5));
    }
}
