/*
 * membench native kernels — see membench_native.h.
 *
 * The inner product is vectorised with ARM NEON when available (Apple Silicon /
 * aarch64) and falls back to a scalar loop elsewhere; the compiler auto-vectorises
 * the scalar path at -O3 on x86 too. Top-k uses an insertion-maintained sorted
 * buffer, which is optimal for the small k (single-digit to low-tens) this
 * benchmark uses.
 */
#include "membench_native.h"

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
