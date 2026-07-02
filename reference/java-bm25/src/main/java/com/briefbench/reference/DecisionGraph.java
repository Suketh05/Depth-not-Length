package com.briefbench.reference;

import java.util.*;

/**
 * Typed decision-graph model + depth-theory kernels, reimplemented in Java as an
 * independent cross-language check of membench's structured-graph arm
 * ("brief_graph_3hop") and the paper's mechanism theory.
 *
 * <p>The node/edge model mirrors {@code schema/graph.sql}: {@code entity_objects}
 * are nodes (decisions, constraints, features, justifications) and {@code entity_link}
 * are typed, directed, confidence-weighted edges. Bounded multi-hop traversal mirrors
 * the {@code traverseEntityGraph} recursive-CTE template in that schema (parameterised
 * by start / max_hops / min_confidence).
 *
 * <p>On top of the model this class implements the paper's closed-form kernels so the
 * scenario tests can assert them against the canonical golden pack:
 * <ul>
 *   <li>Supersession recovery (tab:super, theory/supersession.py): a chain of {@code n}
 *       versions has {@code n-1} {@code supersedes} edges; a link-follower recovers the
 *       current head with probability {@code q^(n-1)}, a recency-agnostic similarity pick
 *       with the uniform chance floor {@code 1/n}.</li>
 *   <li>Scatter penalty (eq:scatter / Thm thm:scatter, theory/scatter.py): a decision
 *       shredded across {@code sigma} areas assembles with probability {@code p^sigma};
 *       a typed dereference collapses {@code sigma} to 1.</li>
 *   <li>Similarity recovery ceiling (eq:simceiling / Thm thm:sim, theory/recovery.py):
 *       {@code P_sim(d) = s0^d * rho^(d(d+1)/2)} (super-geometric decay).</li>
 *   <li>The fairness lock (sec:fairness / sec:budget): every arm receives an identical
 *       token budget and a deterministic packing order.</li>
 * </ul>
 *
 * <p>All measured benchmark numbers (e.g. the 92.3% supersession figure, fitted p=0.92)
 * are deliberately NOT asserted here; this class asserts only the closed forms, exactly
 * as the Python modules do. Java-17 compatible (no newer language features).
 */
public final class DecisionGraph {

    /**
     * The 16 relationship types from the {@code entity_link.relationship_type} CHECK
     * constraint in {@code schema/graph.sql}. {@code SUPERSEDES} carries the version
     * chain used by the supersession scenario; {@code CONSTRAINS} / {@code IMPLEMENTS}
     * are the other governance edges named in tab:litcompare.
     */
    public enum RelationshipType {
        RELATED_TO, CONSTRAINS, CONSTRAINED_BY, IMPLEMENTS, VALIDATES,
        CONTRADICTS, INFORMS, SUPERSEDES, BLOCKS, DERIVED_FROM,
        SUPPORTS_GOAL, MOTIVATES, ENABLES, IMPACTS, REFERENCES, DUPLICATE_OF
    }

    /** A typed record (mirrors {@code entity_objects}); {@code id} is the primary key. */
    public static final class Node {
        public final String id;
        public final String title;

        public Node(String id, String title) {
            this.id = Objects.requireNonNull(id, "id");
            this.title = title;
        }

        @Override
        public String toString() {
            return "Node[" + id + "]";
        }
    }

    /**
     * A typed, directed, confidence-weighted edge (mirrors {@code entity_link}).
     * {@code confidence} is constrained to [0, 1] exactly as the schema CHECK.
     */
    public static final class Edge {
        public final String source;
        public final String target;
        public final RelationshipType type;
        public final double confidence;

        public Edge(String source, String target, RelationshipType type, double confidence) {
            if (confidence < 0.0 || confidence > 1.0) {
                throw new IllegalArgumentException("confidence must be in [0,1]: " + confidence);
            }
            this.source = Objects.requireNonNull(source, "source");
            this.target = Objects.requireNonNull(target, "target");
            this.type = Objects.requireNonNull(type, "type");
            this.confidence = confidence;
        }
    }

    private final Map<String, Node> nodes = new LinkedHashMap<>();
    private final List<Edge> edges = new ArrayList<>();

    /** Register a node; re-registering the same id replaces the title (idempotent by id). */
    public DecisionGraph addNode(String id, String title) {
        nodes.put(id, new Node(id, title));
        return this;
    }

    /**
     * Add a typed edge. Both endpoints must already be registered (FK semantics of
     * {@code entity_link ... REFERENCES entity_objects}). Duplicate (source, target,
     * type) triples are rejected, mirroring the schema UNIQUE constraint.
     */
    public DecisionGraph addEdge(String source, String target, RelationshipType type, double confidence) {
        if (!nodes.containsKey(source)) {
            throw new IllegalArgumentException("unknown source node: " + source);
        }
        if (!nodes.containsKey(target)) {
            throw new IllegalArgumentException("unknown target node: " + target);
        }
        for (Edge e : edges) {
            if (e.source.equals(source) && e.target.equals(target) && e.type == type) {
                throw new IllegalArgumentException("duplicate edge: " + source + "->" + target + " " + type);
            }
        }
        edges.add(new Edge(source, target, type, confidence));
        return this;
    }

    /** Full edge-with-default-confidence (1.0) convenience, matching the schema DEFAULT. */
    public DecisionGraph addEdge(String source, String target, RelationshipType type) {
        return addEdge(source, target, type, 1.0);
    }

    /**
     * Bounded, confidence-gated multi-hop reachability. Direct Java port of the
     * recursive-CTE template in {@code schema/graph.sql} ({@code traverseEntityGraph}):
     *
     * <pre>
     *   WITH RECURSIVE walk(node, depth) AS (
     *       SELECT :start, 0
     *     UNION
     *       SELECT e.target, walk.depth + 1
     *       FROM walk JOIN entity_link e ON e.source = walk.node
     *       WHERE walk.depth &lt; :max_hops AND e.confidence &gt;= :min_confidence
     *   )
     *   SELECT DISTINCT node, depth FROM walk;
     * </pre>
     *
     * The {@code UNION} (set semantics) makes each node appear once at its
     * shortest gated depth; edges below {@code minConfidence} are not followed,
     * exactly as the paper's 3-hop typed traversal drops low-fidelity links.
     *
     * @return map of reachable node id -&gt; shortest hop-depth from {@code start}
     *     (the start itself is always present at depth 0).
     */
    public Map<String, Integer> traverse(String start, int maxHops, double minConfidence) {
        if (!nodes.containsKey(start)) {
            throw new IllegalArgumentException("unknown start node: " + start);
        }
        if (maxHops < 0) {
            throw new IllegalArgumentException("maxHops must be >= 0: " + maxHops);
        }
        // Adjacency filtered by the confidence gate, source -> targets.
        Map<String, List<String>> adj = new HashMap<>();
        for (Edge e : edges) {
            if (e.confidence >= minConfidence) {
                adj.computeIfAbsent(e.source, k -> new ArrayList<>()).add(e.target);
            }
        }
        // BFS realises the CTE's shortest-depth set semantics deterministically.
        Map<String, Integer> depth = new LinkedHashMap<>();
        depth.put(start, 0);
        Deque<String> frontier = new ArrayDeque<>();
        frontier.add(start);
        while (!frontier.isEmpty()) {
            String cur = frontier.removeFirst();
            int d = depth.get(cur);
            if (d >= maxHops) {
                continue;
            }
            for (String next : adj.getOrDefault(cur, Collections.emptyList())) {
                if (!depth.containsKey(next)) {
                    depth.put(next, d + 1);
                    frontier.addLast(next);
                }
            }
        }
        return depth;
    }

    public Node node(String id) {
        return nodes.get(id);
    }

    public int nodeCount() {
        return nodes.size();
    }

    public int edgeCount() {
        return edges.size();
    }

    // ------------------------------------------------------------------
    // Supersession chains (tab:super; theory/supersession.py)
    //
    // A decision revised n times is a chain of n versions. "Later supersedes
    // earlier": for versions ordered oldest..newest [v1, v2, ..., vn] the edge
    // v_{i+1} --supersedes--> v_i is added, so a chain of n versions has exactly
    // n-1 supersedes edges (edge_hops(n) = n-1). The current HEAD vn is the only
    // version that is a supersedes-source but never a supersedes-target.
    // ------------------------------------------------------------------

    /**
     * Wire a supersession chain over already-registered nodes, oldest first.
     * Adds {@code v[i+1] --SUPERSEDES--> v[i]} for each adjacent pair at the given
     * per-edge confidence (the per-hop fidelity {@code q} of the theory).
     */
    public DecisionGraph addSupersessionChain(double confidence, String... oldestToNewest) {
        if (oldestToNewest.length < 1) {
            throw new IllegalArgumentException("chain needs at least one version");
        }
        for (int i = 1; i < oldestToNewest.length; i++) {
            addEdge(oldestToNewest[i], oldestToNewest[i - 1], RelationshipType.SUPERSEDES, confidence);
        }
        return this;
    }

    /**
     * Link-following supersession resolver: from any version in a chain, walk the
     * {@code supersedes} edges toward the newest revision and return the current HEAD.
     * Only edges at or above {@code minConfidence} are followed (the same gate as
     * {@link #traverse}); a low-confidence link stops the walk, so the resolver never
     * fabricates a currency it cannot verify. Cycle-guarded.
     *
     * <p>This is the typed-store behaviour of tab:super: it returns the current
     * decision by construction, not by recency-blind similarity.
     */
    public String resolveCurrent(String versionId, double minConfidence) {
        if (!nodes.containsKey(versionId)) {
            throw new IllegalArgumentException("unknown version node: " + versionId);
        }
        Set<String> seen = new LinkedHashSet<>();
        String cur = versionId;
        seen.add(cur);
        while (true) {
            String newer = null;
            for (Edge e : edges) {
                if (e.type == RelationshipType.SUPERSEDES
                        && e.target.equals(cur)
                        && e.confidence >= minConfidence) {
                    newer = e.source;
                    break;
                }
            }
            if (newer == null) {
                return cur; // nobody supersedes cur => cur is the head
            }
            if (!seen.add(newer)) {
                throw new IllegalStateException("supersession cycle at " + newer);
            }
            cur = newer;
        }
    }

    /** True iff some (gated) {@code supersedes} edge targets this version. */
    public boolean isSuperseded(String versionId, double minConfidence) {
        for (Edge e : edges) {
            if (e.type == RelationshipType.SUPERSEDES
                    && e.target.equals(versionId)
                    && e.confidence >= minConfidence) {
                return true;
            }
        }
        return false;
    }

    // ==================================================================
    // THEORY KERNELS (closed forms only; goldens live in the golden pack).
    // ==================================================================

    /** Number of supersedes edges in a chain of {@code n} versions: {@code n-1}. */
    public static int edgeHops(int n) {
        requireChainLength(n);
        return n - 1;
    }

    /**
     * Structured supersession recovery (theory/supersession.py): a link-follower
     * traverses {@code n-1} supersedes edges of per-edge fidelity {@code q} to reach
     * the head, recovering it with probability {@code q^(n-1)} (1.0 when {@code n=1}).
     * Golden: n=2,q=0.95 -&gt; 0.95; n=4,q=0.9 -&gt; 0.9^3 = 0.729.
     */
    public static double structuredSupersessionRecovery(int n, double q) {
        requireChainLength(n);
        requireFidelity(q);
        return Math.pow(q, n - 1);
    }

    /**
     * Similarity supersession recovery: a recency-agnostic retriever picks uniformly
     * among the {@code n} near-identical versions, so it returns the current head with
     * the chance floor {@code 1/n} (0.5 at n=2 — the chance baseline in tab:super).
     */
    public static double similaritySupersessionRecovery(int n) {
        requireChainLength(n);
        return 1.0 / n;
    }

    /** Structured advantage {@code q^(n-1) - 1/n}. Golden: n=2,q=0.95 -&gt; 0.45. */
    public static double supersessionAdvantage(int n, double q) {
        return structuredSupersessionRecovery(n, q) - similaritySupersessionRecovery(n);
    }

    /**
     * Break-even per-edge fidelity where structured recovery equals the uniform
     * similarity floor: solve {@code q^(n-1) = 1/n} =&gt; {@code q = n^(-1/(n-1))}.
     * Golden: n=2 -&gt; 0.5; n=4 -&gt; 4^(-1/3) = 0.6299605249474366. Returns 0.0 at n=1
     * (degenerate: a single version has no edge to traverse).
     */
    public static double breakEvenQ(int n) {
        requireChainLength(n);
        if (n == 1) {
            return 0.0;
        }
        return Math.pow(n, -1.0 / (n - 1));
    }

    // ------------------------------------------------------------------
    // Scatter penalty law (eq:scatter / Thm thm:scatter; theory/scatter.py)
    //
    // A decision shredded across sigma areas must be re-assembled from sigma
    // probes; assembly succeeds with probability p^sigma. A typed dereference
    // collapses sigma to 1 (one link, one probe). Flat similarity over a depth-d
    // chain pays sigma = d.
    // ------------------------------------------------------------------

    /**
     * Similarity recovery ceiling (eq:simceiling / Thm thm:sim, theory/recovery.py):
     * {@code P_sim(d) = s0^d * rho^(d(d+1)/2)} with triangular exponent
     * {@code T = d(d+1)/2}. Super-geometric decay in causal-hop distance {@code d}.
     * Golden (s0=0.8, rho=0.5, d=2): 0.8^2 * 0.5^3 = 0.64 * 0.125 = 0.08.
     */
    public static double similarityRecovery(int d, double s0, double rho) {
        requireDepth(d);
        requirePositive(s0, "s0");
        requirePositive(rho, "rho");
        long t = (long) d * (d + 1) / 2;
        return Math.pow(s0, d) * Math.pow(rho, t);
    }

    /**
     * Scatter factor {@code p^sigma}, p in (0,1], sigma &gt;= 0 (fractional allowed).
     * {@code p^0 = 1} (typed dereference, no scatter), {@code p^1 = p}.
     * Golden: scatterFactor(0.5, 3) = 0.5^3 = 0.125.
     */
    public static double scatterFactor(double p, double sigma) {
        requireProbability(p);
        requireScatter(sigma);
        return Math.pow(p, sigma);
    }

    /**
     * Composed scattered recovery {@code P_sim(d) * p^sigma}.
     * Golden (s0=0.8, rho=0.5, d=2, p=0.5, sigma=2): 0.08 * 0.25 = 0.02.
     */
    public static double scatteredRecovery(int d, double s0, double rho, double p, double sigma) {
        return similarityRecovery(d, s0, rho) * scatterFactor(p, sigma);
    }

    /**
     * Degrade a base recovery probability by a scatter factor: {@code base * p^sigma}.
     * Golden: degradeRecovery(0.5, 0.8, 3) = 0.5 * 0.8^3 = 0.5 * 0.512 = 0.256.
     */
    public static double degradeRecovery(double base, double p, double sigma) {
        if (base < 0.0 || base > 1.0) {
            throw new IllegalArgumentException("base must be in [0,1]: " + base);
        }
        return base * scatterFactor(p, sigma);
    }

    /**
     * The assembly-probability curve {@code [p^0, p^1, ..., p^maxSigma]} over integer
     * scatter. Golden: factorCurve(0.5, 4) = [1.0, 0.5, 0.25, 0.125, 0.0625].
     */
    public static double[] factorCurve(double p, int maxSigma) {
        requireProbability(p);
        if (maxSigma < 0) {
            throw new IllegalArgumentException("maxSigma must be >= 0: " + maxSigma);
        }
        double[] curve = new double[maxSigma + 1];
        for (int s = 0; s <= maxSigma; s++) {
            curve[s] = Math.pow(p, s);
        }
        return curve;
    }

    /**
     * Probes a flat-similarity retriever must issue to reassemble a decision scattered
     * across {@code sigma} areas: one probe per area, so {@code sigma}. A typed
     * dereference needs exactly 1 ({@link #probesForDereference}). This is the
     * sigma-vs-1 separation of Thm thm:scatter / sec:corpus.
     */
    public static int probesForScatter(int sigma) {
        if (sigma < 0) {
            throw new IllegalArgumentException("sigma must be >= 0: " + sigma);
        }
        return sigma;
    }

    /** A typed dereference collapses scatter to a single probe (sigma -&gt; 1). */
    public static int probesForDereference() {
        return 1;
    }

    /**
     * Information-theoretic probe lower bound from Thm thm:scatentropy:
     * {@code probes >= sigma / p_bar}, i.e. {@code ceil(sigma / p_bar)} whole probes
     * when each probe recovers a fraction {@code p_bar} of the scattered locations.
     */
    public static int probesLowerBound(int sigma, double pBar) {
        if (sigma < 0) {
            throw new IllegalArgumentException("sigma must be >= 0: " + sigma);
        }
        requireProbability(pBar);
        return (int) Math.ceil(sigma / pBar);
    }

    // ------------------------------------------------------------------
    // The fairness lock (sec:fairness / sec:budget)
    //
    // Every arm is handed an IDENTICAL token budget (matched across arms to ~1%
    // in the paper; fig:tokensbox), and the context packer is deterministic, so
    // any measured win reflects the memory mechanism and not an unequal budget or
    // a lucky packing order. We model that as: (1) equal budgets, (2) a greedy
    // packer that is a pure function of (item order, budget).
    // ------------------------------------------------------------------

    /** A context item with a fixed token cost (a candidate for the packed window). */
    public static final class Item {
        public final String id;
        public final int tokens;

        public Item(String id, int tokens) {
            if (tokens < 0) {
                throw new IllegalArgumentException("tokens must be >= 0: " + tokens);
            }
            this.id = Objects.requireNonNull(id, "id");
            this.tokens = tokens;
        }
    }

    /** Result of a packing pass: the packed ids (in packed order) and tokens used. */
    public static final class PackResult {
        public final List<String> packedIds;
        public final int usedTokens;

        PackResult(List<String> packedIds, int usedTokens) {
            this.packedIds = Collections.unmodifiableList(packedIds);
            this.usedTokens = usedTokens;
        }
    }

    /**
     * The fairness lock: hand every arm the SAME token budget. Returns an
     * insertion-ordered map arm -&gt; budget with every value equal to {@code budget}.
     */
    public static Map<String, Integer> equalBudgets(List<String> arms, int budget) {
        if (budget < 0) {
            throw new IllegalArgumentException("budget must be >= 0: " + budget);
        }
        Map<String, Integer> out = new LinkedHashMap<>();
        for (String arm : arms) {
            out.put(arm, budget);
        }
        return out;
    }

    /**
     * Deterministic greedy packer: walk {@code items} in the given order and admit
     * each one whose token cost still fits under {@code budget}, skipping (not
     * stopping at) any item that would overflow. Pure function of (order, budget) —
     * no randomness, no tie ambiguity — so two arms with identical item lists and the
     * same budget pack identically. Mirrors the harness's construct-cap packing.
     */
    public static PackResult pack(List<Item> items, int budget) {
        if (budget < 0) {
            throw new IllegalArgumentException("budget must be >= 0: " + budget);
        }
        List<String> packed = new ArrayList<>();
        int used = 0;
        for (Item it : items) {
            if (used + it.tokens <= budget) {
                packed.add(it.id);
                used += it.tokens;
            }
        }
        return new PackResult(packed, used);
    }

    /**
     * Canonical packing order (tokens ascending, then id lexicographic) — a stable
     * total order that removes any dependence on the arbitrary arrival order of
     * candidates, so the packer is reproducible across runs / languages / seeds.
     */
    public static List<Item> canonicalOrder(List<Item> items) {
        List<Item> sorted = new ArrayList<>(items);
        sorted.sort(Comparator.comparingInt((Item it) -> it.tokens).thenComparing(it -> it.id));
        return sorted;
    }

    // ---- domain guards (replicate the Python module invariants) ----

    private static void requireChainLength(int n) {
        if (n < 1) {
            throw new IllegalArgumentException("chain length n must be >= 1: " + n);
        }
    }

    private static void requireFidelity(double q) {
        if (!(q > 0.0) || q > 1.0) {
            throw new IllegalArgumentException("fidelity q must be in (0,1]: " + q);
        }
    }

    private static void requireDepth(int d) {
        if (d < 1) {
            throw new IllegalArgumentException("depth d must be >= 1: " + d);
        }
    }

    private static void requireProbability(double p) {
        if (!(p > 0.0) || p > 1.0) {
            throw new IllegalArgumentException("probability must be in (0,1]: " + p);
        }
    }

    private static void requireScatter(double sigma) {
        if (!(sigma >= 0.0)) {
            throw new IllegalArgumentException("sigma must be >= 0: " + sigma);
        }
    }

    private static void requirePositive(double v, String name) {
        if (!(v > 0.0)) {
            throw new IllegalArgumentException(name + " must be > 0: " + v);
        }
    }
}
