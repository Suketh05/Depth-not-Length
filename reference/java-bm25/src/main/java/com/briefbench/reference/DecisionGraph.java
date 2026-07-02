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
}
