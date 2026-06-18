/*
 * membench native kernels — C/SIMD hot-path implementations.
 *
 * These mirror the NumPy reference implementations in
 * `membench.retrieval._native` exactly; the reference is authoritative and
 * produces the reported numbers, while this library exists so that sweeping 20+
 * memory systems over thousands of tasks and multiple seeds stays tractable.
 * Correctness is enforced by a byte-for-byte parity test against the reference.
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

#ifdef __cplusplus
}
#endif

#endif /* MEMBENCH_NATIVE_H */
