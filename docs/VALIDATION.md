# Cross-language validation

Headline metrics are validated by **independent re-implementations in other languages**
— the standard way to gain confidence that a number is the computation you think it is,
not an artifact of one implementation. If a reference disagrees with the Python on the
same inputs, CI fails and the discrepancy is a real bug to fix in one of them.

| Computation | Python (authoritative) | Independent reference | Agreement checked on |
|---|---|---|---|
| nDCG (ranking quality) | `membench.metrics.retrieval.ndcg_at_k` | C# — `reference/csharp-ndcg` | exact values (xunit == pytest) |
| BM25 (lexical ranking) | `membench.retrieval.bm25` | Java (Okapi) — `reference/java-bm25` | ranking order |
| Friedman + Nemenyi CD | `membench.stats.multiple_comparisons` | R — `reference/r` | omnibus significance + best-arm rank + CD |
| Cosine top-k (kernel) | `membench.retrieval._native` (NumPy) | C / Rust / CUDA | byte-for-byte parity |
| Graph traversal | `membench.retrieval.graph.traversal` | SQL recursive CTE — `schema/graph.sql` | same reachable set/depths |

Within Python, statistical results are additionally cross-checked against SciPy and
statsmodels (Wilcoxon, Holm/BH, Cohen's kappa) in the test suite.

## Reproducing

```sh
scripts/reproduce.sh        # Linux/macOS
scripts/reproduce.ps1       # Windows (PowerShell 7+)
docker build -t briefbench . && docker run --rm briefbench run --dataset synthetic
```

All cross-language references build and run in the `native` CI workflow (C, Rust, CUDA
compile, Go, TypeScript, C#, Java, R, SQL, shell + PowerShell lint).
