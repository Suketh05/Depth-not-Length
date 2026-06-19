# BriefBench (`membench`)

Reference implementation for the paper *Depth, Not Length: Why Coding Agents Fail and
How Structured Memory Fixes It.*

Coding agents fail not because they run out of tokens but because **similarity-based
memory cannot recover decisions that sit several reasoning hops away** from the current
task. This benchmark measures that — and shows that a **structured memory that follows
typed links** (as in Brief's context graph) escapes the depth ceiling, on both accuracy
and token cost, with a provable, measurable crossover depth `d⋆`.

## The result, in one figure

At a fixed token budget, governing-decision recovery vs. causal depth `d`
(measured, offline, deterministic):

| arm | d=1 | d=2 | d=3 |
|---|---|---|---|
| BM25 / TF-IDF / dense / hybrid | 0.95–1.00 | 0.85–1.00 | 0.35–0.70 |
| RAPTOR (hierarchical) | 0.55 | 0.25 | 0.05 |
| **Brief graph (link-following)** | **1.00** | **1.00** | **1.00** |

Similarity is fine when the rule is shallow and **collapses as depth grows**; the
structured arm stays flat. The crossover appears near `d=2`, matching the theory's
predicted `d⋆`. This holds for **any** link-following system — it is a property of the
mechanism, not of Brief specifically.

## Reproduce

```sh
scripts/reproduce.sh                       # Linux/macOS: sync, build, test, run, render
docker build -t briefbench . && docker run --rm briefbench run --dataset synthetic
```

Or step by step:

```sh
uv sync --extra dev --extra viz
uv run membench run     --dataset synthetic --budget 150 --out results/rows.jsonl
uv run membench tables  --src results/rows.jsonl
uv run membench report  --src results/rows.jsonl --out results/report.md
uv run membench figures --src results/rows.jsonl --out-dir figures
uv run membench theory  --rho 0.5 --q 0.97        # predicted d*
```

Everything runs offline and deterministically (a stub model + a hashing embedding); real
LLM / embedding / competitor backends live behind optional extras.

## What's measured

- **22 memory arms** — controls (none / full-context / random), lexical (BM25, TF-IDF),
  dense / hybrid / rerank / HyDE / RAPTOR, the Brief typed-graph arm (+ hop/decay
  variants), and adapters onto Mem0 / Supermemory / Zep-Graphiti / GraphRAG / Letta /
  LangChain.
- **Datasets** — dcbench (decisions invisible in code), SWE-bench (cross-file
  constraint), LongMemEval (temporal supersession), and a controllable synthetic
  depth-crossover benchmark.
- **Metrics** — decision compliance, merge-ready correctness, retrieval P@k/MRR/nDCG/
  full-chain recovery, cost-to-correct + Return on Tokens, calibration.
- **Statistics** — BCa / cluster bootstrap CIs, Bayesian `P(Brief > competitor)`, paired
  permutation / Wilcoxon / McNemar, Holm / BH / Friedman / Nemenyi critical difference,
  cluster-robust logistic `compliance ~ arm*depth`, empirical `d⋆` with CI, causal
  mediation, stochastic dominance, decay-law model selection, power.

## Integrity (why the numbers are defensible)

- **Fairness lock** — identical model, retrieval budget, and code-search tool across
  every arm; memory architecture is the only variable (validated in `config/schema.py`).
- **Two-tier provenance** — numbers we measured under identical conditions are never
  mixed with vendor-reported claims (`competitors/`, guarded by `assert_single_tier`).
- **No rigging** — the benchmark tests the genuine failure mode (link-reachable,
  dissimilar facts); strong baselines; if a result ever doesn't favour Brief we report it.
- **Cross-language validation** — core computations re-implemented in C#, Java, R, SQL
  and checked for agreement (see `docs/VALIDATION.md`).

## Repository layout

```
src/membench/   types · theory/ · retrieval/ (+ graph/, external/) · agents/ (+ llm/)
                datasets/ · metrics/ · stats/ · analysis/ · config/ · cli · harness · device
native/c · crates/membench-kernels (Rust) · cuda/ · orchestrator/ (Go)   # accel + scale
tools/seed (TypeScript) · reference/ (C#, Java, R) · schema/ (SQL)        # tooling + validation
scripts/ (sh + ps1) · Dockerfile · docs/ (HARDWARE, VALIDATION, METHODOLOGY, THREATS)
```

See `docs/METHODOLOGY.md` for the design, `docs/THREATS.md` for threats to validity, and
`paper/CLAIMS.md` for the claim → code → table map. Cite via `CITATION.cff`.
