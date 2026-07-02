package com.briefbench.reference;

/**
 * Independent Java re-implementation of the full-chain <em>recovery</em> laws from
 * "Depth, Not Length: A Benchmark and Theory of Memory, Retrieval, and Decision
 * Compliance in LLM Coding Agents" (Varanasi et al.). This class exists to
 * cross-check the Python reference arm ({@code membench.theory.recovery}) in a
 * second language: the closed forms below are re-derived from the paper equations,
 * and the accompanying {@code RecoveryTest} asserts the canonical golden values
 * (golden pack section A) at full float64 precision.
 *
 * <p>Two regimes are modelled, both for a task at the head of a length-{@code d}
 * dependence chain {@code n_0 -> n_1 -> ... -> n_d}:
 *
 * <ul>
 *   <li><b>Similarity memory</b> (eq:simceiling / Thm thm:sim). Per-hop
 *       retrievability follows the geometric decay {@code s_i = s0 * rho^i}
 *       (Assumption ass:geometric). Recovering the whole chain multiplies the
 *       per-hop terms, giving the <em>similarity ceiling</em>
 *       <pre>  P_sim(d) = prod_{i=1..d} s0 * rho^i = s0^d * rho^{d(d+1)/2}.</pre>
 *       The triangular exponent {@code d(d+1)/2} makes the fall-off
 *       super-geometric (slope -&gt; -inf), which is the whole story of the depth
 *       ceiling.</li>
 *   <li><b>Structured memory</b> (Thm thm:struct). Following an explicit typed
 *       link succeeds with per-edge fidelity {@code q} in {@code (0, 1]},
 *       independent of surface similarity, so the chain is recovered with
 *       <pre>  P_struct(d) = q^d,</pre>
 *       which decays only gently and carries no {@code M_d} size penalty.</li>
 * </ul>
 *
 * <p>Their ratio {@code P_struct/P_sim = q^d / (s0^d rho^{d(d+1)/2}) -> inf} as
 * {@code d -> inf}: the advantage of structure grows without bound with depth
 * (the crossover of the two curves is analysed in {@link Crossover}).
 *
 * <p>Every quantity is also exposed in log-space, because the similarity product
 * underflows float64 for even moderately deep chains and any honest solver or plot
 * must stay numerically stable there.
 *
 * <p>Paper anchors: sec:depth (per-hop survival model), eq:simceiling / thm:sim
 * (similarity ceiling), thm:struct (bounded traversal), sec:depth "Synthetic
 * operating point" (calibration s0=0.70, rho=0.67, q=0.97).
 */
public final class Recovery {

    private Recovery() {
        // static-only utility; the theory carries no per-instance state
    }

    private static void checkDepth(int d) {
        if (d < 1) {
            throw new IllegalArgumentException("depth d must be >= 1, got " + d);
        }
    }

    /**
     * Triangular exponent {@code T = d(d+1)/2}, the sum {@code 1 + 2 + ... + d} of
     * per-hop decay powers. Computed in {@code double} (matching Python's true
     * division) so it composes with the continuous-{@code d} forms in
     * {@link Crossover}.
     */
    static double triangular(double d) {
        return d * (d + 1.0) / 2.0;
    }

    /**
     * Probability that similarity memory recovers a full depth-{@code d} chain,
     * {@code P_sim(d) = s0^d * rho^{d(d+1)/2}} (eq:simceiling / Thm thm:sim).
     *
     * <p>Evaluated directly; underflows to {@code 0.0} for deep chains (float64),
     * which is faithful behaviour — use {@link #logSimilarityRecovery} when a
     * finite log is needed.
     */
    public static double similarityRecovery(int d, Model model) {
        checkDepth(d);
        return Math.pow(model.s0(), d) * Math.pow(model.rho(), triangular(d));
    }

    /**
     * Natural log of {@link #similarityRecovery}, stable for large {@code d}:
     * {@code log P_sim(d) = d*ln(s0) + (d(d+1)/2)*ln(rho)} (eq:simceiling).
     *
     * <p>This is concave in {@code d} with slope {@code -> -inf}, the analytic
     * signature of super-geometric decay.
     */
    public static double logSimilarityRecovery(int d, Model model) {
        checkDepth(d);
        return d * Math.log(model.s0()) + triangular(d) * Math.log(model.rho());
    }

    /**
     * Probability that structured (typed-link) memory recovers a full depth-{@code d}
     * chain, {@code P_struct(d) = q^d} (Thm thm:struct).
     *
     * <p>Only {@code Theta(d)} work and no size penalty: the fall-off is a plain
     * geometric {@code q^d}, gentle compared with the similarity ceiling.
     *
     * @param d chain depth ({@code >= 1})
     * @param q per-edge fidelity in {@code (0, 1]}
     */
    public static double structuredRecovery(int d, double q) {
        checkDepth(d);
        if (!(q > 0.0 && q <= 1.0)) {
            throw new IllegalArgumentException("q must be in (0, 1], got " + q);
        }
        return Math.pow(q, d);
    }

    /** Convenience: {@link #structuredRecovery(int, double)} using the model's {@code q}. */
    public static double structuredRecovery(int d, Model model) {
        return structuredRecovery(d, model.q());
    }

    /**
     * Immutable calibration bundle: the geometric similarity-decay parameters
     * ({@code s0}, {@code rho}) plus the structured per-edge fidelity {@code q}.
     *
     * <p>Domain guards mirror the Python reference and the golden pack
     * (section A cross-cutting note 4): {@code s0 > 0}, {@code rho > 0},
     * {@code q in (0, 1]}. A model is <em>decaying</em> when {@code rho < 1}, which
     * every depth-ceiling / crossover argument requires.
     */
    public record Model(double s0, double rho, double q) {
        public Model {
            if (!(s0 > 0.0)) {
                throw new IllegalArgumentException("s0 must be > 0, got " + s0);
            }
            if (!(rho > 0.0)) {
                throw new IllegalArgumentException("rho must be > 0, got " + rho);
            }
            if (!(q > 0.0 && q <= 1.0)) {
                throw new IllegalArgumentException("q must be in (0, 1], got " + q);
            }
        }

        /** Whether the similarity model genuinely decays ({@code rho < 1}). */
        public boolean isDecaying() {
            return rho < 1.0;
        }

        /**
         * The paper's recurring synthetic operating point: s0 = 0.70, rho = 0.67,
         * q = 0.97 (sec:depth; Prop prop:cross).
         */
        public static Model calibration() {
            return new Model(0.70, 0.67, 0.97);
        }
    }
}
