package com.briefbench.reference;

/**
 * Independent Java re-implementation of the <em>crossover</em> analysis from
 * "Depth, Not Length" — the depth {@code d*} at which structured memory starts to
 * pay for itself over similarity memory. Cross-checks the Python reference
 * ({@code membench.theory.crossover} and {@code membench.theory.bounds}); the
 * accompanying {@code CrossoverTest} asserts the golden pack section A.3 values.
 *
 * <p>On pure recovery the structured store always looks better, so to be honest it
 * is charged for what it costs. Let {@code V} be the value of solving a task and
 * {@code W} the extra per-chain overhead of building and maintaining typed links.
 * Structured memory is the better choice when its net recovery gain clears that
 * overhead (eq:gd):
 * <pre>
 *   V * (q^d - s0^d rho^{d(d+1)/2}) &gt; W
 *     &lt;=&gt;  g(d) := P_struct(d) - P_sim(d) &gt; tau,   tau = W/V.
 * </pre>
 *
 * <p><b>An honesty note the theory sketch glosses over.</b> Proposition prop:cross
 * asserts {@code g} is increasing, which holds only in the {@code q = 1} limit. For
 * {@code q < 1} both terms vanish as {@code d -> inf}, so {@code g} rises, peaks,
 * then falls back toward zero: the win is an <em>interval</em>
 * {@code [d*, d_upper]}, not a half-line. We therefore solve for the crossover and
 * report the full win region, and additionally bracket it in closed form.
 *
 * <p>Taking logs of the advantage {@code P_struct/P_sim} gives an upward parabola in
 * {@code d} with roots at {@code d = 0} and the closed-form zero-overhead crossover
 * <pre>  d0 = 2*ln(q/s0)/ln(rho) - 1  </pre>
 * (bounds.py::tau_zero_crossover). So {@code g(d) &gt; 0} iff {@code d &gt; d0}, and at
 * {@code tau = 0} the smallest winning integer depth is exactly
 * {@code floor(d0) + 1}.
 *
 * <p>Paper anchors: sec:cross (the crossover), eq:gd (structured advantage),
 * prop:cross (calibrated d*=2 for 0.50 &lt; tau &lt;= 0.79; closed-form d*=1),
 * thm:hitting (return-on-tokens separation).
 */
public final class Crossover {

    private Crossover() {
        // static-only utility
    }

    /**
     * Net recovery gain {@code g(d) = P_struct(d) - P_sim(d) = q^d - s0^d rho^{d(d+1)/2}}
     * (eq:gd), defined for continuous {@code d} so the crossover can be located
     * between integers.
     */
    public static double g(double d, Recovery.Model model) {
        double pStruct = Math.pow(model.q(), d);
        double pSim = Math.pow(model.s0(), d) * Math.pow(model.rho(), Recovery.triangular(d));
        return pStruct - pSim;
    }

    /**
     * Overhead ratio {@code tau = W/V} (per-chain link cost over value-of-correct).
     *
     * @param maintenanceCost {@code W >= 0}
     * @param taskValue {@code V > 0}
     */
    public static double overheadRatio(double maintenanceCost, double taskValue) {
        if (!(taskValue > 0.0)) {
            throw new IllegalArgumentException("task_value V must be positive, got " + taskValue);
        }
        if (!(maintenanceCost >= 0.0)) {
            throw new IllegalArgumentException(
                    "maintenance_cost W must be non-negative, got " + maintenanceCost);
        }
        return maintenanceCost / taskValue;
    }

    /** Net value of structured over similarity memory: {@code V*g(d) - W}. */
    public static double netValueGain(
            double d, Recovery.Model model, double taskValue, double maintenanceCost) {
        return taskValue * g(d, model) - maintenanceCost;
    }

    /**
     * Continuous zero-overhead crossover {@code d0}, the nonzero root of the
     * log-advantage parabola, i.e. the depth where {@code P_struct(d) = P_sim(d)}
     * (bounds.py::tau_zero_crossover):
     * <pre>  d0 = 2*ln(q/s0)/ln(rho) - 1.  </pre>
     * For {@code d > d0} structured recovery exceeds similarity recovery; below it,
     * similarity wins. At the paper calibration {@code d0 = -2.629133632756974}, so
     * structure already wins at {@code d = 1}.
     *
     * @throws IllegalArgumentException if the model does not decay ({@code rho >= 1}),
     *     so the quadratic depth-ceiling argument does not apply
     */
    public static double tauZeroCrossover(Recovery.Model model) {
        if (!model.isDecaying()) {
            throw new IllegalArgumentException(
                    "crossover bounds require a decaying similarity model (rho < 1), got rho="
                            + model.rho());
        }
        double lnQ = Math.log(model.q());
        double lnS0 = Math.log(model.s0());
        double lnRho = Math.log(model.rho());
        return 2.0 * (lnQ - lnS0) / lnRho - 1.0;
    }
}
