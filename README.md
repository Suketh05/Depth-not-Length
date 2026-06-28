# BriefBench (`membench`)

> Reference implementation for the paper
> **"Depth, Not Length: A Benchmark and Theory of Memory, Retrieval, and Decision
> Compliance in LLM Coding Agents."**

[![ci](https://github.com/Suketh05/Depth-not-Length/actions/workflows/ci.yml/badge.svg)](https://github.com/Suketh05/Depth-not-Length/actions/workflows/ci.yml)
[![license: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](.python-version)
[![ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](pyproject.toml)

Coding agents fail not because they run out of tokens but because **similarity-based
memory cannot recover the decisions that sit several causal hops away** from the task
they are touching — and once a decision *supersedes* an earlier one, similarity has no
way to prefer the current rule over the obsolete one. This benchmark measures that
failure, and shows that a **structured memory that follows typed links** (as in Brief's
decision graph) escapes the depth ceiling on accuracy *and* token cost, with a
**provable, measurable crossover depth `d⋆`** beyond which structure must win.

The thesis in one line: *similarity retrieval has a depth ceiling; a typed-link store
escapes it; the crossover is provable.*

---

## Contents

- [The result, in one table](#the-result-in-one-table)
- [The theory in brief](#the-theory-in-brief)
- [Results](#results)
- [Reproduce the paper](#reproduce-the-paper)
- [What's measured](#whats-measured)
- [Repository layout](#repository-layout)
- [Integrity & the fairness lock](#integrity--the-fairness-lock)
- [Citation](#citation)

---

## The result, in one table

Real-code retrieval and use, pooled dcbench + swebench, Claude — every number is the
**paper's reported result** (source of truth), from paper table `tab:realret`
([`results/METRICS.md`](results/METRICS.md) §1 ·
[`results/data/paper_realcode_pooled.csv`](results/data/paper_realcode_pooled.csv)):

| arm | recall `P_ret` | compliance `P_comply` | precision | use factor `κ` |
|---|---|---|---|---|
| **Brief graph (typed link-following)** | **0.667** | **0.469** | **0.385** | **0.703** |
| dense (best similarity baseline) | 0.635 | 0.406 | 0.323 | 0.639 |
| bm25 | 0.604 | 0.344 | 0.302 | 0.570 |
| none (no product context) | 0.219 | 0.073 | 0.052 | — |

The typed store ranks first on recall and compliance, and its use factor `κ = 0.703` is
the **only** arm to clear the **0.56–0.64** band that ceilings every similarity arm. The
same mechanism shows up on causal depth: on the synthetic suite Brief decays least
(compliance slope −0.075 vs. similarity −0.125 to −0.200), and the Brief−best-similarity
gap widens with depth (+0.025 / +0.025 / **+0.075** at d=1/2/3; `tab:slope`) — the
crossover the theory predicts near `d⋆ = 2`. The effect is a property of the *mechanism*
(link-following), not of Brief specifically: any typed-link store inherits it.

---

## The theory in brief

The paper factors decision compliance into a *retrieval* term and a *use* term, then
bounds each. (Equations quoted verbatim from the manuscript; see also the
claim→code→table map in [`paper/CLAIMS.md`](paper/CLAIMS.md).)

**Compliance factorization** — compliance is retrieval times a "use factor":

```
P_comply(d) = P_ret(d) · κ(d)
```

where `P_ret(d)` is the probability the governing decision is retrieved and
`κ(d) = P(comply | governing decision retrieved)`. Because `∂P_comply/∂P_ret = κ`, under
separability `P_comply ≤ κ` — a **hard use ceiling**. The "use ceiling" names the
empirical `κ ≈ 0.6` that bounds similarity retrievers on real code.

**Irreducible context-free floor** — a *governed* task (`H(Y|X) > 0`) cannot be solved
from the task surface alone; the Bayes error is a strictly positive floor:

```
P_err^cf ≥ 1 − E_X max_y P(y|X) > 0
```

**Per-hop survival** — with hop independence, `P_ret(d) = Π_{k=1}^d f_k`.

**The similarity ceiling** — under geometric similarity decay `s_k = s_0·ρ^k`, recovery
decays *super-geometrically* in depth:

```
P_ret^sim(d) = s_0^d · ρ^{d(d+1)/2}
```

The log-probability is concave with slope → −∞.

**Structured recovery** — a bounded traversal that follows stored links pays only a
steady per-edge cost:

```
P_ret^struct(d) = q^d        (q ∈ (0,1] per-edge fidelity, Θ(d) work)
```

**The crossover** — the structured advantage `g(d) = q^d − s_0^d·ρ^{d(d+1)/2}` is
unimodal; a crossover depth `d⋆` exists. Calibrated at `s_0 = 0.70, ρ = 0.67, q = 0.97`:
`g(1) ≈ 0.50`, `g(2) ≈ 0.79`. The closed-form crossover (smallest `d` with `g > 0`) is
`d⋆ = 1`; the headline **`d⋆ = 2`** is the calibrated value for an overhead margin
`τ ∈ (0.50, 0.79]` and is flagged as calibrated post-hoc.

**Scatter penalty** — spreading one decision across `σ` locations costs
`P_asm(σ) = p^σ`; a typed record keeps `σ → 1`.

**Supersession** — governance edges (`supersedes`) make "prefer the current decision"
a dereference, not a similarity guess.

Each result is marked in the paper as **proven**, **calibrated**, or **under-identified**.

---

## Results

> **The paper is the single source of truth.** Every number below is transcribed
> verbatim from the manuscript (`Claude (Sonnet)` answer model; "Brief" = the
> `brief_graph_3hop` typed-graph arm) and is attributed to its paper table label. These
> are the paper's **reported** results — not regenerated by this repo's offline harness.
> The full tables and the per-table CSVs live in [`results/METRICS.md`](results/METRICS.md),
> [`results/EVALS.md`](results/EVALS.md), and [`results/data/`](results/data) (see
> [`results/data/README.md`](results/data/README.md) for provenance). The paper tiers each
> competitor number as **reported / modeled / vendor-published** and never mixes the tiers.

**Real code (pooled dcbench + swebench, Claude)** — the typed store ranks first where it
matters (`tab:realret`):

| arm | recall `P_ret` | compliance `P_comply` | precision | use factor `κ` |
|---|---|---|---|---|
| **Brief** | **0.667** | **0.469** | **0.385** | **0.703** |
| dense | 0.635 | 0.406 | 0.323 | 0.639 |
| hybrid_rrf | 0.615 | 0.396 | 0.323 | 0.644 |
| tfidf | 0.604 | 0.385 | 0.344 | 0.637 |
| bm25 | 0.604 | 0.344 | 0.302 | 0.570 |
| rerank_ce | 0.583 | 0.354 | 0.312 | 0.607 |
| raptor | 0.573 | 0.323 | 0.281 | 0.564 |
| none | 0.219 | 0.073 | 0.052 | — |

Brief's `κ = 0.703` is the **only** arm to clear the **0.56–0.64** use-factor band that
ceilings every similarity arm.

**Synthetic suite (Claude), compliance by arm:** Brief **0.933**, tfidf 0.883,
bm25 0.875, dense 0.858, raptor 0.817, none 0.025.

**Depth slope (synthetic, Claude)** — Brief decays least: Brief −0.075 vs. similarity
−0.125 to −0.200; the Brief−best-similarity gap is +0.025 / +0.025 / **+0.075** at
d=1 / 2 / 3.

**Supersession (current-decision-ranked-first %, chance = 50%):** Brief **92.3%** vs.
the similarity band **64–69%** (bm25 64.1, tfidf 65.8, dense 68.7) — a 23–28 pt lead.

**Public benchmarks (unified harness):** LoCoMo LLM-judge **87.6** (next best RAPTOR
84.7); DMR fact-F1 **94.2** (next best Kluris 87.0); SWE-ContextBench resolution
**47.3%** (next best Unabyss 37.6%); HotpotQA recall@5 **0.783**, nDCG@10 **0.987**,
MRR **0.981** (all best-in-class); HotpotQA supporting-fact recall 0.503 — the one row
Brief loses (Zep 0.529).

**Token economics:** Brief is the cheaper session in **80.0%** of matchups
(2880 / 3600, 0 ties), 68.6%–86.7% per competitor; sessions collapse to 2–3 turns at
~60% fewer tokens.

**Unbiased scorecard over 41 axes** (`tab:scorecard`): **22 W / 14 m (near-tie) / 5 l** —
the 5 losses are listed in [`results/EVALS.md`](results/EVALS.md) and kept on purpose.

Each headline above is backed by a paper-sourced CSV under
[`results/data/`](results/data):
[`paper_realcode_pooled.csv`](results/data/paper_realcode_pooled.csv),
[`paper_synthetic_compliance.csv`](results/data/paper_synthetic_compliance.csv),
[`paper_depth_slope.csv`](results/data/paper_depth_slope.csv),
[`paper_supersession.csv`](results/data/paper_supersession.csv),
[`paper_public_benchmarks.csv`](results/data/paper_public_benchmarks.csv),
[`paper_token_economics.csv`](results/data/paper_token_economics.csv),
[`paper_scorecard.csv`](results/data/paper_scorecard.csv). The paper additionally tiers
each competitor number as **reported / modeled / vendor-published** and never mixes the
tiers; competitors' own published headlines are in
[`results/competitor_landscape.md`](results/competitor_landscape.md).

---

## Reproduce the paper

The manuscript reports a verified inventory of **69 tables**, **66 figure
environments** (~**80** `\includegraphics`, 78 unique), backed by **108 figure files**.
Everything below runs **offline and deterministically** by default (a stub model + a
hashing embedding); real LLM / embedding / competitor backends live behind optional
extras.

```sh
scripts/reproduce.sh                       # Linux/macOS: sync, build native, test, run, render
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

See [`scripts/reproduce.sh`](scripts/reproduce.sh) for the one-command path.

### CLI → paper artifact map

The `membench` CLI has eight commands (see [`src/membench/cli.py`](src/membench/cli.py)).
Defaults below are the real defaults in the code / [`configs/default.yaml`](configs/default.yaml).

| Command | What it produces | Representative paper artifact | Key flags (defaults) |
|---|---|---|---|
| `theory` | Predicted crossover depth `d⋆` from decay/structure params | Crossover prop. & phase diagram (`d⋆`) | `--rho 0.5 --q 0.97 --s0 0.9 --maintenance 0.2 --value 1.0` |
| `run` | Scored rows for one dataset × arms sweep | Synthetic depth-crossover & slope tables | `--dataset synthetic --budget 150 --per-depth 10 --seed 0 --out results/rows.jsonl` |
| `sweep` | Headline arms **+** the dcbench/swebench four-arm ablation | Real-code transfer + ablation cells | `--budget 150 --per-depth 10 --seed 0` |
| `sweep-config` | Config-driven sweep (per-dataset budget) + run manifest | Full multi-dataset run with provenance | `--config configs/default.yaml --out-dir results` |
| `tables` | Headline compliance tables to stdout | Synthetic / real-code headline tables | `--src results/rows.jsonl` |
| `report` | Markdown report (incl. vendor-claim landscape) | Results narrative + competitive landscape | `--src results/rows.jsonl --out results/report.md` |
| `figures` | Publication figures (e.g. depth-crossover) | Crossover / slope figures | `--src results/rows.jsonl --out-dir figures` |
| `frontier` | Accuracy–cost Pareto frontier + area-under-depth-curve | Cost-frontier figure, AUDC per arm | `--src results/rows.jsonl` |

The fairness lock lives in the config: budget is keyed **by dataset, never by arm**
(`synthetic: 150`, `dcbench: 250`, `swebench: 500`, `longmemeval: 8000` in
[`configs/default.yaml`](configs/default.yaml); the headline arms are `none, bm25, tfidf,
dense, hybrid_rrf, rerank_ce, raptor, brief_graph_3hop`).

Pre-rendered figures are indexed in [`results/FIGURES.md`](results/FIGURES.md).

> **Note on the manuscript PDF.** The full paper (≈91 pp.) is a preliminary working
> draft and, per its own availability statement, is *not redistributed* with this repo.
> Design and methodology documents live in [`docs/`](docs)
> ([`METHODOLOGY.md`](docs/METHODOLOGY.md), [`THREATS.md`](docs/THREATS.md),
> [`VALIDATION.md`](docs/VALIDATION.md), [`HARDWARE.md`](docs/HARDWARE.md)).

---

## What's measured

- **Memory arms.** Headline sweep (offline-runnable): `none`, `bm25`, `tfidf`, `dense`,
  `hybrid_rrf`, `rerank_ce`, `raptor`, and **`brief_graph_3hop`** (the typed decision
  graph with governance edges `constrains` / `supersedes` / `implements`, 3-hop
  traversal). A budget-matched `random_context` control and the `none` control anchor the
  four-arm ablation. The wider library also ships HyDE, ColBERT-lite, SPLADE-lite, MMR,
  PPR, RM3, BM25F/BM25+, LMDir, spreading-activation, and external adapters for Mem0,
  Supermemory, Zep, GraphRAG, LangChain, Letta, and Cognee
  ([`src/membench/retrieval/`](src/membench/retrieval), `external/` adapters need their
  own envs).
- **Datasets** ([`src/membench/datasets/`](src/membench/datasets)). A controllable
  **synthetic** depth-crossover suite (120 tasks, 40/depth, drift ρ≈0.67); **dcbench**
  (decisions invisible in code; 42 tasks, 41 weighted decision points); **swebench**
  (cross-file constraint from real SWE-bench issues; 54 tasks, 21/21/12 at d=1/2/3);
  **longmemeval** (temporal supersession); plus loaders for the public **LoCoMo**,
  **DMR**, **HotpotQA**, SWE-ContextBench, and DC-bench harnesses.
- **Metrics** ([`src/membench/metrics/`](src/membench/metrics)). Decision compliance,
  merge-ready correctness, retrieval recall / precision / P@k / MRR / nDCG / full-chain
  recovery, the **use factor κ**, cost-to-correct, and **Return on Tokens**, plus
  calibration.
- **Statistics** ([`src/membench/stats/`](src/membench/stats)). BCa / cluster bootstrap
  CIs, Bayesian `P(Brief > competitor)`, paired permutation / Wilcoxon / McNemar,
  Holm / BH / Friedman + Nemenyi critical difference, cluster-robust logistic
  `compliance ~ arm*depth`, empirical `d⋆` with CI, causal mediation, stochastic
  dominance, decay-law model selection, and power.

---

## Repository layout

```
src/membench/   types · theory/ · retrieval/ (+ graph/, external/) · agents/
                datasets/ · metrics/ · stats/ · analysis/ · config/ · cli · harness · device
native/c · crates/ (Rust kernels) · cuda/ · orchestrator/ (Go)            # accel + scale
tools/ (TypeScript seed) · reference/ (C#, Java, R) · schema/ (SQL)        # tooling + validation
competitors/ (vendor-claim registry) · configs/ · scripts/ · Dockerfile
results/ (METRICS, EVALS, FIGURES, data/, figures/) · paper/ (CLAIMS, measured/) · docs/
```

Architecture & methodology pointers: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the
design, [`docs/THREATS.md`](docs/THREATS.md) for threats to validity,
[`docs/VALIDATION.md`](docs/VALIDATION.md) for the cross-language re-implementations, and
[`paper/CLAIMS.md`](paper/CLAIMS.md) for the claim → code → artifact map.

---

## Integrity & the fairness lock

- **Fairness lock** — identical model, retrieval budget, and code-search tool across
  every arm; the memory architecture is the *only* variable. This is enforced
  structurally: the budget is keyed by dataset (never by arm) and the schema validator
  rejects a config that omits a swept dataset's budget
  ([`src/membench/config/schema.py`](src/membench/config/schema.py),
  [`configs/default.yaml`](configs/default.yaml)).
- **Tiered provenance** — the paper's reported numbers are never mixed with vendor
  claims. Competitors' own published numbers are cited separately
  ([`results/competitor_landscape.md`](results/competitor_landscape.md),
  [`competitors/registry.yaml`](competitors/registry.yaml)); the paper tiers each cell as
  reported / modeled / vendor-published and the CSVs under
  [`results/data/`](results/data) preserve that distinction.
- **No rigging** — the benchmark targets the genuine failure mode (link-reachable but
  dissimilar facts) against strong baselines; losing cuts are reported, not hidden
  (5 losses in the paper's 41-axis scorecard, listed in
  [`results/EVALS.md`](results/EVALS.md)).
- **Scope caveat** — the harness supplies a common pre-extracted decision corpus, so it
  measures retrieval and *use* of typed links, not extraction; the paper states this is
  not an end-to-end product win.
- **Cross-language validation** — core computations are re-implemented in C#, Java, R,
  and SQL and checked for agreement ([`docs/VALIDATION.md`](docs/VALIDATION.md)).

---

## Citation

Please cite the paper and the benchmark. See [`CITATION.cff`](CITATION.cff) for the
machine-readable record.

```bibtex
@misc{varanasi2026depth,
  title        = {Depth, Not Length: A Benchmark and Theory of Memory, Retrieval,
                  and Decision Compliance in LLM Coding Agents},
  author       = {Varanasi, Kasyap and Dillon, Drew and Ankeshwar, Suketh and
                  Bussa, Sai Santosh and Sistla, Gayatri and Muthuraman, Vaikunth},
  year         = {2026},
  note         = {Preliminary working draft. Independent personal research;
                  affiliations are employers/schools, not sponsors.},
  howpublished = {\url{https://github.com/Suketh05/Depth-not-Length}}
}
```

Licensed under [Apache-2.0](LICENSE).
