/*
 * membench native kernels — see membench_native.h.
 *
 * The inner product is vectorised with ARM NEON when available (Apple Silicon /
 * aarch64) and falls back to a scalar loop elsewhere; the compiler auto-vectorises
 * the scalar path at -O3 on x86 too. Top-k uses an insertion-maintained sorted
 * buffer, which is optimal for the small k (single-digit to low-tens) this
 * benchmark uses.
 *
 * The depth-theory and inference kernels (lower in the file) are scalar
 * closed-forms; each reproduces its Python reference bit-for-bit by using the
 * same libm primitives (pow/asin/sqrt) in the same association order.
 */
#include "membench_native.h"

#include <math.h>
#include <stddef.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

int mb_version(void) { return 1; }

static float mb_dot(const float *a, const float *b, int dim) {
#ifdef __ARM_NEON
  float32x4_t acc = vdupq_n_f32(0.0f);
  int i = 0;
  for (; i + 4 <= dim; i += 4) {
    acc = vmlaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
  }
  float s = vaddvq_f32(acc);
  for (; i < dim; ++i) {
    s += a[i] * b[i];
  }
  return s;
#else
  float s = 0.0f;
  for (int i = 0; i < dim; ++i) {
    s += a[i] * b[i];
  }
  return s;
#endif
}

int mb_cosine_topk(const float *corpus, int n, int dim, const float *query,
                   int k, int *out_idx, float *out_score) {
  if (k > n) {
    k = n;
  }
  if (k <= 0) {
    return 0;
  }
  for (int j = 0; j < k; ++j) {
    out_idx[j] = -1;
    out_score[j] = -1e30f;
  }
  for (int i = 0; i < n; ++i) {
    const float score = mb_dot(query, corpus + (size_t)i * dim, dim);
    if (score > out_score[k - 1]) {
      int pos = k - 1;
      while (pos > 0 && out_score[pos - 1] < score) {
        out_score[pos] = out_score[pos - 1];
        out_idx[pos] = out_idx[pos - 1];
        --pos;
      }
      out_score[pos] = score;
      out_idx[pos] = i;
    }
  }
  return k;
}

/* ==========================================================================
 * Depth-theory and inference kernels (see membench_native.h for contracts and
 * paper anchors). Each matches the Python reference to < 1e-12.
 * ======================================================================== */

/*
 * Triangular exponent T = d(d+1)/2 as a double. Python evaluates `d*(d+1)/2`
 * with an int d and true division, yielding an exact double for these small d;
 * we reproduce that promotion so pow() sees the identical exponent.
 */
static double mb_triangular(double d) { return d * (d + 1.0) / 2.0; }

/* P_sim(d) = s0^d * rho^(d(d+1)/2). Paper eq:simceiling. */
double mb_similarity_recovery(double s0, double rho, int d) {
  return pow(s0, (double)d) * pow(rho, mb_triangular((double)d));
}

/* P_struct(d) = q^d. Paper Thm thm:struct. */
double mb_structured_recovery(double q, int d) { return pow(q, (double)d); }

/*
 * g(d) = q^d - s0^d * rho^(d(d+1)/2). Paper eq:gd. Continuous d (matching
 * crossover.g, which locates a sub-integer crossover on the rising edge). The
 * two terms are formed then subtracted in the same order as the Python.
 */
double mb_gap(double s0, double rho, double q, double d) {
  const double p_struct = pow(q, d);
  const double p_sim = pow(s0, d) * pow(rho, mb_triangular(d));
  return p_struct - p_sim;
}

/*
 * Smallest integer depth d in [1, max_d] with g(d) > tau (strict); 0 if none.
 *
 * This is find_crossover_depth(...).d_star: it scans integer depths ascending
 * and returns the first winner, matching numpy's `first True of g_curve > tau`.
 * 0 encodes the Python `d_star is None` (structure never clears the overhead on
 * the scanned range). max_d < 1 is treated as an empty scan (returns 0); the
 * Python layer raises on that input before calling in.
 */
int mb_crossover_search(double s0, double rho, double q, double tau, int max_d) {
  for (int d = 1; d <= max_d; ++d) {
    if (mb_gap(s0, rho, q, (double)d) > tau) {
      return d;
    }
  }
  return 0;
}
