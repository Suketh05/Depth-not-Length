/*
 * membench native kernels — C/SIMD hot-path implementations.
 *
 * These mirror the NumPy / Python reference implementations in
 * `membench.retrieval._native`, `membench.theory.*`, `membench.stats.*` and
 * `membench.metrics.factorization` exactly; the reference is authoritative and
 * produces the reported numbers, while this library exists so that sweeping 20+
 * memory systems over thousands of tasks and multiple seeds stays tractable.
 * Correctness is enforced by a byte-for-byte parity test against the reference.
 *
 * Two families of kernel live here:
 *   1. the cosine top-k retrieval hot path (mb_cosine_topk);
 *   2. scalar closed-form kernels for the paper's depth theory and inference
 *      statistics (mb_similarity_recovery, mb_structured_recovery, mb_gap,
 *      mb_crossover_search, mb_wilson_low/high, mb_cohens_h). These are cheap on
 *      their own but are evaluated inside million-iteration crossover / bootstrap
 *      sweeps, so a native path keeps the theory figures reproducible fast. Every
 *      one cites the paper equation it evaluates in its contract below.
 *
 * The library is built as a shared object (libmembench_native.{dylib,so}) and
 * loaded via ctypes at runtime; if it is absent the Python layer transparently
 * falls back to NumPy, so the package never hard-depends on a compiled artifact.
 */
#ifndef MEMBENCH_NATIVE_H
#define MEMBENCH_NATIVE_H

#ifdef __cplusplus
extern "C" {
#endif

/* ABI version, bumped on any signature change. */
int mb_version(void);

/*
 * Cosine top-k over an L2-normalised corpus.
 *
 * `corpus` is a row-major (n, dim) float32 matrix whose rows are unit-norm;
 * `query` is a unit-norm (dim,) float32 vector. Because both are normalised the
 * dot product equals cosine similarity. Fills `out_idx`/`out_score` (length k)
 * with the indices and scores of the top-k rows in descending score order and
 * returns the number actually written (min(k, n)).
 */
int mb_cosine_topk(const float *corpus, int n, int dim, const float *query,
                   int k, int *out_idx, float *out_score);

/* --------------------------------------------------------------------------
 * Depth-theory and inference kernels.
 *
 * All are pure, side-effect-free scalar functions on IEEE-754 double, computed
 * with the same libm primitives (pow, asin, sqrt) and the same operation order
 * as their Python references, so the ctypes parity test agrees to < 1e-12.
 * ------------------------------------------------------------------------ */

/*
 * Similarity full-chain recovery P_sim(d) = s0^d * rho^(d(d+1)/2).
 *
 * Paper: eq:simceiling / Thm thm:sim ("the similarity ceiling"); the triangular
 * exponent d(d+1)/2 is the super-geometric decay of chain recovery. Mirrors
 * membench.theory.recovery.similarity_recovery. Domain: d >= 1, s0 > 0, rho > 0.
 * Evaluated as pow(s0, d) * pow(rho, d*(d+1)/2.0) — matching Python float**int,
 * which routes through libm pow after promoting the exponent to double.
 */
double mb_similarity_recovery(double s0, double rho, int d);

/*
 * Structured full-chain recovery P_struct(d) = q^d.
 *
 * Paper: Thm thm:struct (bounded-traversal recovery; steady per-hop cost q, no
 * M_d branching penalty). Mirrors membench.theory.recovery.structured_recovery.
 * Domain: d >= 1, q in (0, 1] (not enforced here — the Python layer validates).
 */
double mb_structured_recovery(double q, int d);

/*
 * Structured advantage / crossover gain g(d) = q^d - s0^d * rho^(d(d+1)/2).
 *
 * Paper: eq:gd ("the structured advantage", unimodal in d). Mirrors
 * membench.theory.crossover.g, which accepts a continuous d for sub-integer
 * crossover location, so d is a double here too.
 */
double mb_gap(double s0, double rho, double q, double d);

/*
 * Smallest integer depth d in [1, max_d] with g(d) > tau (strict), else 0.
 *
 * Paper: Prop prop:cross ("the crossover depth d*"); this is the integer d_star
 * of membench.theory.crossover.find_crossover_depth under overhead ratio
 * tau = W/V. Returns 0 when structure never clears the overhead on the scanned
 * range (the Python d_star is None there). Scans low-to-high and stops at the
 * first winning depth. Requires max_d >= 1, tau >= 0.
 */
int mb_crossover_search(double s0, double rho, double q, double tau, int max_d);

/*
 * Wilson score interval for a binomial proportion, lower and upper bound.
 *
 * center = (p + z^2/2n) / (1 + z^2/n),
 * margin = z*sqrt(p(1-p)/n + z^2/4n^2) / (1 + z^2/n),   p = successes/n,
 * low  = max(0, center - margin),  high = min(1, center + margin).
 *
 * Paper: Wilson (1927) score interval, the inference used for the use-factor
 * kappa (sec:realuse, tab:realret) and every rate CI. Mirrors
 * membench.metrics.factorization.wilson_interval. z (the standard-normal
 * quantile) is passed in so the kernel is free of a normal-CDF inverse;
 * successes/n may be fractional (pooled expected counts). Requires n > 0.
 */
double mb_wilson_low(double successes, double n, double z);
double mb_wilson_high(double successes, double n, double z);

/*
 * Cohen's h effect size for two proportions:
 *   h = 2*asin(sqrt(p1)) - 2*asin(sqrt(p2)).
 *
 * Paper: fig:cohensh / sec:worked (variance-stabilising arcsine contrast of two
 * compliance rates). Mirrors membench.stats.effect_size.cohens_h. Domain
 * p1, p2 in [0, 1] (validated in the Python layer).
 */
double mb_cohens_h(double p1, double p2);

#ifdef __cplusplus
}
#endif

#endif /* MEMBENCH_NATIVE_H */
