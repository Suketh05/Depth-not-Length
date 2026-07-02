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

    /**
     * The crossover analysis at a given overhead ratio {@code tau} (mirror of
     * {@code crossover.py::CrossoverResult}). When no depth clears the overhead
     * {@code exists} is false and the depth fields are sentinels ({@code dStar = -1},
     * {@code dStarContinuous = NaN}, {@code winLo/winHi = -1}); {@code peakDepth} and
     * {@code gCurve} are always populated.
     *
     * @param tau overhead ratio the gain was tested against
     * @param exists whether structured memory's net gain ever clears {@code tau}
     * @param dStar smallest integer depth in the win region ({@code -1} if none)
     * @param dStarContinuous sub-integer crossover on the rising edge of {@code g}
     *     ({@code NaN} if none)
     * @param winLo inclusive lower end of the integer win region ({@code -1} if none)
     * @param winHi inclusive upper end of the integer win region ({@code -1} if none)
     * @param peakDepth integer depth maximising {@code g} over the scanned range
     * @param gCurve {@code g} evaluated at integer depths {@code 1..maxDepth}
     */
    public record CrossoverResult(
            double tau,
            boolean exists,
            int dStar,
            double dStarContinuous,
            int winLo,
            int winHi,
            int peakDepth,
            double[] gCurve) {}

    /**
     * Locate the crossover depth {@code d*} for overhead ratio {@code tau} using the
     * default scan range ({@code maxDepth = 64}, matching the Python reference).
     */
    public static CrossoverResult findCrossoverDepth(Recovery.Model model, double tau) {
        return findCrossoverDepth(model, tau, 64);
    }

    /**
     * Locate the crossover depth {@code d*} for overhead ratio {@code tau}
     * (crossover.py::find_crossover_depth). Scans integer depths {@code 1..maxDepth}
     * for the contiguous region where {@code g(d) > tau} (strict), takes its smallest
     * integer as {@code d*}, and refines a continuous crossover on the rising edge by
     * bisection (a dependency-free stand-in for the reference's Brent solver — the
     * root is bracketed, so both converge to the same value).
     *
     * @param maxDepth largest integer depth scanned ({@code >= 1})
     */
    public static CrossoverResult findCrossoverDepth(
            Recovery.Model model, double tau, int maxDepth) {
        if (tau < 0.0) {
            throw new IllegalArgumentException("tau must be non-negative, got " + tau);
        }
        if (maxDepth < 1) {
            throw new IllegalArgumentException("max_depth must be >= 1, got " + maxDepth);
        }

        double[] gCurve = new double[maxDepth];
        boolean[] above = new boolean[maxDepth];
        int peakIdx = 0;
        for (int i = 0; i < maxDepth; i++) {
            gCurve[i] = g(i + 1, model);
            above[i] = gCurve[i] > tau;
            if (gCurve[i] > gCurve[peakIdx]) {
                peakIdx = i;
            }
        }
        int peakDepth = peakIdx + 1;

        int firstIdx = -1;
        for (int i = 0; i < maxDepth; i++) {
            if (above[i]) {
                firstIdx = i;
                break;
            }
        }
        if (firstIdx < 0) {
            return new CrossoverResult(tau, false, -1, Double.NaN, -1, -1, peakDepth, gCurve);
        }

        int dStar = firstIdx + 1;
        int lastIdx = firstIdx;
        while (lastIdx + 1 < maxDepth && above[lastIdx + 1]) {
            lastIdx++;
        }
        int winHi = lastIdx + 1;

        // Continuous crossover on the rising edge: g(d) - tau changes sign in
        // (d* - 1, d*] unless the model already wins at d = 1.
        double lo = dStar - 1.0;
        double dContinuous;
        if (lo < 1e-9 || g(lo, model) - tau >= 0.0) {
            dContinuous = dStar;
        } else {
            dContinuous = bisectRoot(model, tau, lo, dStar);
        }

        return new CrossoverResult(
                tau, true, dStar, dContinuous, dStar, winHi, peakDepth, gCurve);
    }

    /**
     * Bisection root of {@code g(d) - tau} on {@code [lo, hi]}, which is assumed to
     * bracket a sign change (the caller guarantees {@code g(lo) - tau < 0} and
     * {@code g(hi) - tau >= 0}). Converges to full float64 precision, well inside the
     * {@code 1e-6} tolerance the reference asserts on the continuous root.
     */
    private static double bisectRoot(Recovery.Model model, double tau, double lo, double hi) {
        double fLo = g(lo, model) - tau;
        for (int iter = 0; iter < 200; iter++) {
            double mid = 0.5 * (lo + hi);
            double fMid = g(mid, model) - tau;
            if (fMid == 0.0 || (hi - lo) <= Math.ulp(mid)) {
                return mid;
            }
            if ((fLo < 0.0) != (fMid < 0.0)) {
                hi = mid;
            } else {
                lo = mid;
                fLo = fMid;
            }
        }
        return 0.5 * (lo + hi);
    }
}
