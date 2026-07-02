// Theory.cs -- Independent C# re-derivation of the depth-theory closed forms and the
// paper-table identities from "Depth, Not Length: A Benchmark and Theory of Memory,
// Retrieval, and Decision Compliance in LLM Coding Agents" (depth_not_length_FIXED.tex).
//
// This is the THIRD independent implementation of the theory (after the Python
// reference `src/membench/theory/` and the Java arm). The point of N independent
// derivations is that a bug in any single one surfaces as a cross-language
// disagreement on a shared, published number. Every value asserted in TheoryTests.cs
// is transcribed VERBATIM from the canonical golden pack (full float64 precision /
// exact fraction) and from the paper's published tables -- never a value produced by
// running this code.
//
// Paper anchors implemented / verified here:
//   eq:simceiling, Thm thm:sim  -> P_ret^sim(d) = s0^d * rho^{d(d+1)/2}  (super-geometric decay)
//   Thm thm:struct              -> P_ret^struct(d) = q^d                 (bounded traversal, no M_d penalty)
//   eq:gd, Prop prop:cross      -> g(d) = q^d - s0^d*rho^{d(d+1)/2}; crossover depth d*
//   eq:fano, Thm thm:fano       -> context-free floor -> 1 - 1/log2 M_d  (uniform survivors -> 1 - 1/K)
//   eq:factor, Cor cor:atten    -> P_comply = P_ret * kappa  =>  kappa = compliance/recall (tab:realret)
//   tab:scorecard               -> 22 W / 14 m / 5 l = 41 axes
namespace Membench;

/// <summary>
/// Per-hop survival parameters for the depth theory (paper §sec:perhop, ass:geometric).
/// <c>S0</c> = initial similarity retention s0 (0,1]; <c>Rho</c> = geometric decay ratio rho (0,1);
/// <c>Q</c> = per-edge traversal fidelity q (0,1]. The paper's recurring synthetic operating
/// point is s0=0.70, rho=0.67, q=0.97 (§sec:depth "Synthetic operating point", Prop prop:cross).
/// </summary>
public readonly record struct SurvivalModel(double S0, double Rho, double Q);

/// <summary>
/// Closed-form depth-theory functions and paper-table identity checks. Static, pure, and
/// side-effect free so the same inputs always yield the same IEEE-754 float64 output as the
/// Python/Java references. See file header for the paper anchor of each region.
/// </summary>
public static class Theory
{
    /// <summary>
    /// Paper synthetic operating point (§sec:depth "Synthetic operating point"; Prop prop:cross):
    /// s0 = 0.70, rho = 0.67, q = 0.97. All Section-A goldens are evaluated at this calibration.
    /// </summary>
    public static readonly SurvivalModel Calibration = new(0.70, 0.67, 0.97);

    // ---- domain guards (golden-pack cross-cutting note 4: d>=1; q,p in (0,1]; tau>=0) ----

    private static void RequireDepth(int d)
    {
        if (d < 1)
            throw new ArgumentOutOfRangeException(nameof(d), d, "depth d must be >= 1");
    }

    private static void RequireModel(SurvivalModel m)
    {
        if (m.S0 <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(m), m.S0, "s0 must be > 0");
        if (m.Rho <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(m), m.Rho, "rho must be > 0");
        RequireHalfOpenUnit(m.Q, "q");
    }

    private static void RequireHalfOpenUnit(double x, string name)
    {
        // (0, 1]
        if (x <= 0.0 || x > 1.0)
            throw new ArgumentOutOfRangeException(name, x, name + " must be in (0, 1]");
    }

    private static void RequireClosedUnit(double x, string name)
    {
        // [0, 1]
        if (x < 0.0 || x > 1.0)
            throw new ArgumentOutOfRangeException(name, x, name + " must be in [0, 1]");
    }

    private static void RequireTau(double tau)
    {
        if (tau < 0.0)
            throw new ArgumentOutOfRangeException(nameof(tau), tau, "tau must be >= 0");
    }

    // ---- A.1 similarity recovery (eq:simceiling / Thm thm:sim) ----

    /// <summary>
    /// Similarity retrieval recovery ceiling P_ret^sim(d) = s0^d * rho^{d(d+1)/2}
    /// (eq:simceiling, Thm thm:sim). The triangular exponent T = d(d+1)/2 = Sum_{k=1..d} k
    /// makes the decay super-geometric (log-slope -> -inf). At the paper calibration this
    /// gives 0.469, 0.14737387..., 0.031027225..., 0.004376633..., 0.000413630... for d=1..5.
    /// </summary>
    public static double SimilarityRecovery(int d, SurvivalModel model)
    {
        RequireDepth(d);
        RequireModel(model);
        double t = d * (d + 1) / 2.0; // triangular exponent Sum_{k=1..d} k
        return Math.Pow(model.S0, d) * Math.Pow(model.Rho, t);
    }

    /// <summary>
    /// Log-space form log P_ret^sim(d) = d*ln(s0) + (d(d+1)/2)*ln(rho) (eq:simceiling).
    /// Stays finite where <see cref="SimilarityRecovery"/> underflows to 0.0 in float64
    /// (golden pack A.1 underflow guard: s0=0.5, rho=0.3, d=60).
    /// </summary>
    public static double LogSimilarityRecovery(int d, SurvivalModel model)
    {
        RequireDepth(d);
        RequireModel(model);
        double t = d * (d + 1) / 2.0;
        return d * Math.Log(model.S0) + t * Math.Log(model.Rho);
    }

    // ---- A.2 structured recovery (Thm thm:struct) ----

    /// <summary>
    /// Bounded-traversal recovery P_ret^struct(d) = q^d (Thm thm:struct): a typed store that
    /// follows stored links pays only a steady per-edge cost q, with no similarity M_d penalty.
    /// q=0.97 gives 0.97, 0.9409, 0.912673, 0.88529281, 0.8587340257 for d=1..5.
    /// </summary>
    public static double StructuredRecovery(int d, double q)
    {
        RequireDepth(d);
        RequireHalfOpenUnit(q, nameof(q));
        return Math.Pow(q, d);
    }

    // === derivations appended below ===
}
