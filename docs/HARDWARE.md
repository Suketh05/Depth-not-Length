# Compute paths & hardware provenance

BriefBench is built so the **reported numbers are reproducible on a laptop CPU**, while
remaining able to scale to large sweeps on native and GPU hardware. This document is
explicit about which path produces what, so no result is ever attributed to hardware it
did not run on.

## The reference path (authoritative)

All reported numbers come from the deterministic **Python + NumPy** pipeline with the
offline stub model and the hashing embedding. It needs no network, no API key, and no
GPU, and is bit-for-bit reproducible from a seed. Everything below is *acceleration of
this same computation*, validated to match it — never a substitute that could change a
number.

## Native acceleration (for scale)

Sweeping 20+ memory systems × datasets × depths × seeds makes the cosine-top-k inner
loop a real bottleneck. Three native kernels implement it identically to the NumPy
reference and are **byte-for-byte parity-tested** against it:

| Path | Where | Hardware | Built/validated |
|---|---|---|---|
| C + ARM NEON | `native/c` | CPU (SIMD) | locally on Apple M5 + Linux CI |
| Rust / PyO3 | `crates/membench-kernels` | CPU | CI (maturin) |
| CUDA | `cuda/cosine_topk.cu` | NVIDIA GPU | CI compile; parity on a GPU runner |

On Apple Silicon the **Metal/CPU** path is used (there is no CUDA); on NVIDIA hosts the
CUDA path is available. The Python layer auto-selects: it uses a native kernel when one
is importable/built and otherwise falls back to NumPy, so correctness never depends on a
compiled artifact.

## Orchestration

`orchestrator/` (Go) shards a sweep — each `(dataset, seed)` is an independent
`membench run` — across a worker pool sized to the CPU count (or hosts), shelling out to
the Python CLI so parallelism adds no new benchmark logic.

## Device provenance

`membench.device.detect_device()` records the device a run actually executed on (CUDA /
Apple Metal / CPU) for the run manifest. It **labels** the environment; it never asserts
a device. There are no fabricated "ran on N×A100" claims anywhere in this repository.
