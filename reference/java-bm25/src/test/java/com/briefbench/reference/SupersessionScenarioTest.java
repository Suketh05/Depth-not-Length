package com.briefbench.reference;

import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Supersession scenario suite (paper tab:super / fig:supersession; sec:resmech-super;
 * theory/supersession.py). Verifies the MECHANISM the paper claims — a typed store that
 * follows {@code supersedes} links returns the current head, while a recency-agnostic
 * similarity retriever that ranks the n near-identical versions equally returns the head
 * only with the chance floor {@code 1/n}.
 *
 * <p>INTEGRITY: this suite asserts only the closed forms in the golden pack (edge hops
 * {@code n-1}, structured {@code q^(n-1)}, uniform floor {@code 1/n}, break-even
 * {@code n^(-1/(n-1))}). It deliberately does NOT assert the measured 92.3% typed /
 * 64-69% similarity band of tab:super — those are empirical numbers, out of scope for a
 * closed-form cross-language check (the chance baseline 50% = 1/n at n=2 is the only
 * tab:super number that is also a closed form, and we assert it as such).
 *
 * <p>Fixture: a deterministic 4-version chain of one decision ("auth access-token TTL"),
 * revised from 24h down to 15m; each newer revision supersedes the previous.
 */
class SupersessionScenarioTest {

    // Versions oldest -> newest; the last is the current HEAD.
    private static final String[] TTL_CHAIN = {"ttl-v1", "ttl-v2", "ttl-v3", "ttl-v4"};
    private static final String HEAD = "ttl-v4";
    private static final double GATE = 0.5; // min-confidence gate (as in traverse)

    /** A realistic 4-version supersession chain at per-edge fidelity q. */
    private static DecisionGraph ttlChain(double q) {
        DecisionGraph g = new DecisionGraph()
                .addNode("ttl-v1", "access-token TTL = 24h")
                .addNode("ttl-v2", "access-token TTL = 4h")
                .addNode("ttl-v3", "access-token TTL = 1h")
                .addNode("ttl-v4", "access-token TTL = 15m");
        return g.addSupersessionChain(q, TTL_CHAIN);
    }

    // ---- Mechanism: link-follower returns head from ANY version ----

    @Test
    void linkFollowerResolvesHeadFromEveryVersion() {
        DecisionGraph g = ttlChain(0.95);
        // A high-fidelity chain (q=0.95 >= gate) resolves to the head from any entry point.
        for (String v : TTL_CHAIN) {
            assertEquals(HEAD, g.resolveCurrent(v, GATE),
                    "resolveCurrent from " + v + " must reach the current head");
        }
        // Only the head is un-superseded; every earlier version is superseded.
        assertFalse(g.isSuperseded(HEAD, GATE));
        assertTrue(g.isSuperseded("ttl-v1", GATE));
        assertTrue(g.isSuperseded("ttl-v3", GATE));
    }

    @Test
    void similarityPickIsRecencyBlindAndReturnsStale() {
        DecisionGraph g = ttlChain(0.95);
        // The similarity retriever ranks all n versions equally (near-identical text);
        // exactly one of the n is the head, so a uniform pick returns the head with
        // frequency 1/n. We enumerate the candidate set deterministically.
        int n = TTL_CHAIN.length;
        long headHits = 0;
        for (String candidate : TTL_CHAIN) {
            if (candidate.equals(HEAD)) {
                headHits++;
            }
        }
        double uniformHeadRate = (double) headHits / n;
        assertEquals(DecisionGraph.similaritySupersessionRecovery(n), uniformHeadRate, 1e-12);

        // Concretely: a recency-blind pick of the OLDEST version is stale, yet the
        // link-follower starting from that same stale hit still returns the head.
        String stalePick = "ttl-v1";
        assertNotEquals(HEAD, stalePick);
        assertEquals(HEAD, g.resolveCurrent(stalePick, GATE));
    }

    @Test
    void seededUniformPickIsDeterministicButRecencyAgnostic() {
        // Reproducibility check for the similarity baseline: the same seed yields the
        // same "pick", and across seeds the head is chosen at the 1/n rate (no recency
        // signal). We assert the analytic floor 1/n, never a Monte-Carlo literal.
        int n = TTL_CHAIN.length;
        long draws = 40_000L; // matches the n=40 paired-query design's order of magnitude
        long headHits = 0;
        Random rng = new Random(20240701L);
        String firstPickSeedA = null;
        for (long i = 0; i < draws; i++) {
            String pick = TTL_CHAIN[rng.nextInt(n)];
            if (i == 0) {
                firstPickSeedA = pick;
            }
            if (pick.equals(HEAD)) {
                headHits++;
            }
        }
        // Same seed -> same first draw (determinism).
        Random rngB = new Random(20240701L);
        assertEquals(firstPickSeedA, TTL_CHAIN[rngB.nextInt(n)]);
        // Empirical head-rate hugs the analytic floor 1/n (loose MC tolerance; the
        // asserted GOLDEN is the closed form, this is only a sanity band).
        double empirical = (double) headHits / draws;
        assertEquals(DecisionGraph.similaritySupersessionRecovery(n), empirical, 0.01);
    }

    // ---- Bounded governance traversal: the typed per-hop mechanism ----

    @Test
    void typedStoreReachesGoverningDecisionInDHopsAtQPerHop() {
        // A task at the surface (feature) is governed by a decision 3 causal hops away:
        //   feature --implements--> spec --constrains--> policy --supersedes--> old-policy
        // The typed store follows the stored links and reaches the governing node in a
        // bounded d=3 hops (steady per-hop cost q^d), where flat similarity would decay
        // super-geometrically. We verify reachability + the q^d recovery golden.
        DecisionGraph g = new DecisionGraph()
                .addNode("feature", "add SSO login button")
                .addNode("spec", "SSO uses OIDC")
                .addNode("policy", "tokens are short-lived")
                .addNode("old-policy", "tokens live 24h")
                .addEdge("feature", "spec", DecisionGraph.RelationshipType.IMPLEMENTS, 0.90)
                .addEdge("spec", "policy", DecisionGraph.RelationshipType.CONSTRAINS, 0.85)
                .addEdge("policy", "old-policy", DecisionGraph.RelationshipType.SUPERSEDES, 0.80);

        Map<String, Integer> reached = g.traverse("feature", 3, 0.5);
        assertEquals(0, reached.get("feature"));
        assertEquals(1, reached.get("spec"));
        assertEquals(2, reached.get("policy"));
        assertEquals(3, reached.get("old-policy"));

        // A 2-hop budget stops before the governing decision at depth 3.
        Map<String, Integer> shallow = g.traverse("feature", 2, 0.5);
        assertFalse(shallow.containsKey("old-policy"));
        assertEquals(2, shallow.get("policy"));

        // Per-hop structured recovery over d hops is q^d (Thm thm:struct).
        // q=0.97, d=3 -> 0.912673 (golden pack A.2).
        assertEquals(0.912673, DecisionGraph.structuredRecovery(3, 0.97), 1e-12);
        // structured_recovery(5, 0.95) == 0.95^5 (golden pack A.2).
        assertEquals(Math.pow(0.95, 5), DecisionGraph.structuredRecovery(5, 0.95), 1e-12);
    }

    @Test
    void confidenceGateBoundsTraversal() {
        // Same chain, but the spec->policy link is only weakly inferred (0.30). A 0.5
        // gate truncates the walk at spec; the governing policy is unreachable until the
        // gate drops below 0.30. Mirrors the schema CTE's `e.confidence >= :min_confidence`.
        DecisionGraph g = new DecisionGraph()
                .addNode("feature", "f").addNode("spec", "s").addNode("policy", "p")
                .addEdge("feature", "spec", DecisionGraph.RelationshipType.IMPLEMENTS, 0.90)
                .addEdge("spec", "policy", DecisionGraph.RelationshipType.CONSTRAINS, 0.30);
        assertFalse(g.traverse("feature", 3, 0.5).containsKey("policy"));
        assertTrue(g.traverse("feature", 3, 0.25).containsKey("policy"));
    }

    // ---- Golden closed forms (golden pack A.7, asserted verbatim) ----

    @Test
    void structuredRecoveryGoldens() {
        // n=2, q=0.95: hops=1, p_struct=0.95, p_sim=0.5, advantage=0.45, structured wins.
        assertEquals(1, DecisionGraph.edgeHops(2));
        assertEquals(0.95, DecisionGraph.structuredSupersessionRecovery(2, 0.95), 1e-12);
        assertEquals(0.5, DecisionGraph.similaritySupersessionRecovery(2), 1e-12);
        assertEquals(0.45, DecisionGraph.supersessionAdvantage(2, 0.95), 1e-12);
        assertTrue(DecisionGraph.supersessionAdvantage(2, 0.95) > 0.0);

        // n=4, q=0.9: p_struct = 0.9^3 = 0.729 (exact 729/1000).
        assertEquals(3, DecisionGraph.edgeHops(4));
        assertEquals(0.729, DecisionGraph.structuredSupersessionRecovery(4, 0.9), 1e-12);
        // Uniform baseline: P_sim(2)=0.5, P_sim(4)=0.25.
        assertEquals(0.25, DecisionGraph.similaritySupersessionRecovery(4), 1e-12);
    }

    @Test
    void breakEvenGoldens() {
        // Break-even q where q^(n-1) == 1/n: n=2 -> 0.5; n=4 -> 4^(-1/3).
        assertEquals(0.5, DecisionGraph.breakEvenQ(2), 1e-12);
        assertEquals(0.6299605249474366, DecisionGraph.breakEvenQ(4), 1e-12);
        // Sanity from the pack: q_be^(n-1) == 1/n exactly (to float tolerance).
        for (int n = 2; n <= 6; n++) {
            double qbe = DecisionGraph.breakEvenQ(n);
            assertEquals(1.0 / n, Math.pow(qbe, n - 1), 1e-12);
        }
    }

    // ---- Edge cases: single version, saturation, gate, cycle ----

    @Test
    void singleVersionChainIsDegenerate() {
        DecisionGraph g = new DecisionGraph().addNode("solo", "only decision");
        // No supersedes edge; the sole version is its own head.
        assertEquals("solo", g.resolveCurrent("solo", GATE));
        assertFalse(g.isSuperseded("solo", GATE));
        assertEquals(0, DecisionGraph.edgeHops(1));
        assertEquals(1.0, DecisionGraph.structuredSupersessionRecovery(1, 0.9), 1e-12);
        assertEquals(1.0, DecisionGraph.similaritySupersessionRecovery(1), 1e-12);
        assertEquals(0.0, DecisionGraph.breakEvenQ(1), 1e-12);
    }

    @Test
    void lowConfidenceLinkStopsTheWalk() {
        // The v3->v4 supersession is only inferred at low confidence (0.30). With a 0.5
        // gate the walk cannot cross it, so resolving from v1 saturates at v3, NOT v4:
        // the resolver refuses to assert a currency it cannot verify at the gate.
        DecisionGraph g = new DecisionGraph()
                .addNode("ttl-v1", "24h").addNode("ttl-v2", "4h")
                .addNode("ttl-v3", "1h").addNode("ttl-v4", "15m")
                .addEdge("ttl-v2", "ttl-v1", DecisionGraph.RelationshipType.SUPERSEDES, 0.90)
                .addEdge("ttl-v3", "ttl-v2", DecisionGraph.RelationshipType.SUPERSEDES, 0.90)
                .addEdge("ttl-v4", "ttl-v3", DecisionGraph.RelationshipType.SUPERSEDES, 0.30);
        assertEquals("ttl-v3", g.resolveCurrent("ttl-v1", 0.5));
        // Drop the gate below 0.30 and the full chain resolves to the true head.
        assertEquals("ttl-v4", g.resolveCurrent("ttl-v1", 0.25));
    }

    @Test
    void supersessionCycleIsRejected() {
        DecisionGraph g = new DecisionGraph()
                .addNode("a", "a").addNode("b", "b")
                .addEdge("b", "a", DecisionGraph.RelationshipType.SUPERSEDES, 1.0)
                .addEdge("a", "b", DecisionGraph.RelationshipType.SUPERSEDES, 1.0);
        assertThrows(IllegalStateException.class, () -> g.resolveCurrent("a", GATE));
    }

    @Test
    void domainGuardsRejectIllegalInputs() {
        assertThrows(IllegalArgumentException.class, () -> DecisionGraph.edgeHops(0));
        assertThrows(IllegalArgumentException.class, () -> DecisionGraph.structuredSupersessionRecovery(2, 0.0));
        assertThrows(IllegalArgumentException.class, () -> DecisionGraph.structuredSupersessionRecovery(2, 1.5));
        assertThrows(IllegalArgumentException.class, () -> DecisionGraph.similaritySupersessionRecovery(0));
    }
}
