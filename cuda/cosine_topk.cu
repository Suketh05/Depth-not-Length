// CUDA cosine-top-k kernel for membench (optional GPU accelerator).
//
// Mirrors the NumPy reference in membench.retrieval._native: given a row-major
// (n, dim) L2-normalised corpus and an L2-normalised query, compute per-row cosine
// (= dot product) on the GPU, then select the top-k on the host. Build with:
//   nvcc -O3 -shared -Xcompiler -fPIC cuda/cosine_topk.cu -o cuda/build/libmembench_cuda.so
// It is NOT required for the reported numbers (the CPU/NumPy path is authoritative);
// it exists so large sweeps can run on NVIDIA hardware. On Apple Silicon the Metal/CPU
// path is used instead (see docs/HARDWARE.md). The device a run used is recorded by
// membench.device.detect_device(), never asserted.
//
// This file is compiled in CI (compile-only where no GPU is attached); correctness is
// validated by the same byte-for-byte parity contract as the C and Rust kernels when a
// GPU runner is available.

#include <cuda_runtime.h>
#include <float.h>

extern "C" {

// One thread per corpus row: scores[i] = dot(query, corpus[i]).
__global__ void cosine_scores(const float *corpus, int n, int dim,
                              const float *query, float *scores) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float *row = corpus + (size_t)i * dim;
  float acc = 0.0f;
  for (int j = 0; j < dim; ++j) {
    acc += query[j] * row[j];
  }
  scores[i] = acc;
}

// Host entry point: compute scores on device, copy back, select top-k on host.
// Returns the number written (min(k, n)). out_idx/out_score must hold k elements.
int mb_cuda_cosine_topk(const float *h_corpus, int n, int dim, const float *h_query,
                        int k, int *out_idx, float *out_score) {
  if (k > n) k = n;
  if (k <= 0) return 0;

  float *d_corpus = nullptr, *d_query = nullptr, *d_scores = nullptr;
  cudaMalloc(&d_corpus, (size_t)n * dim * sizeof(float));
  cudaMalloc(&d_query, (size_t)dim * sizeof(float));
  cudaMalloc(&d_scores, (size_t)n * sizeof(float));
  cudaMemcpy(d_corpus, h_corpus, (size_t)n * dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_query, h_query, (size_t)dim * sizeof(float), cudaMemcpyHostToDevice);

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  cosine_scores<<<blocks, threads>>>(d_corpus, n, dim, d_query, d_scores);

  float *h_scores = (float *)malloc((size_t)n * sizeof(float));
  cudaMemcpy(h_scores, d_scores, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);

  // top-k via insertion into a descending buffer (k is small)
  for (int j = 0; j < k; ++j) { out_idx[j] = -1; out_score[j] = -FLT_MAX; }
  for (int i = 0; i < n; ++i) {
    const float s = h_scores[i];
    if (s > out_score[k - 1]) {
      int pos = k - 1;
      while (pos > 0 && out_score[pos - 1] < s) {
        out_score[pos] = out_score[pos - 1];
        out_idx[pos] = out_idx[pos - 1];
        --pos;
      }
      out_score[pos] = s;
      out_idx[pos] = i;
    }
  }

  free(h_scores);
  cudaFree(d_corpus);
  cudaFree(d_query);
  cudaFree(d_scores);
  return k;
}

}  // extern "C"
