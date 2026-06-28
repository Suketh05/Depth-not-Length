# Reproduce the paper

This guide maps the artifact inventory of *Depth, Not Length: A Benchmark and Theory of
Memory, Retrieval, and Decision Compliance in LLM Coding Agents* to the exact `membench`
commands that regenerate the corresponding repository artifact.

Read it alongside [`../paper/CLAIMS.md`](../paper/CLAIMS.md) (claim -> code -> table map)
and [`../README.md`](../README.md) (quick start).

---

## Two separate provenances — read this first

This repository contains **two distinct families of numbers**, and the paper keeps them
apart on purpose. So does this guide.

1. **Repo-measured** — produced by the offline, deterministic `membench` harness on this
   machine, or by the committed measured run. These live in the result files:
   - [`../results/METRICS.md`](../results/METRICS.md) /
     [`../results/data/metrics_matrix.csv`](../results/data/metrics_matrix.csv) — the
     10-bucket metric matrix (387 cells; measured=349, modeled=13, vendor=25).
   - [`../results/EVALS.md`](../results/EVALS.md) — 440 evals across 9 families.
   - [`../results/FIGURES.md`](../results/FIGURES.md) — the 43-figure index.
   - [`../paper/measured/summary.md`](../paper/measured/summary.md) /
     [`../paper/measured/dossier.md`](../paper/measured/dossier.md) — the real
     Claude-Sonnet run (`claude-sonnet-4-6`, 1,824 attempts, n=40 per cell).

2. **Paper-reported** — the numbers printed in the LaTeX manuscript. These are the
   headline figures quoted in the table below under "paper-reported".

> [!IMPORTANT]
> **The committed repo results and the paper-reported numbers are separate provenance and
> may differ run-to-run.** The deterministic offline harness uses a stub model and a
> hashing embedding provider; the paper's measured run uses real model backends; and the
> paper additionally reports *modeled* and *vendor* cells. Do not expect a freshly run
> `results/rows.jsonl` to reproduce a paper-reported headline cell-for-cell. Where the
> harness intentionally compresses a result (the offline stub cannot comply with a
> decision that lacks an extractable code identifier), the repo files say so — see the
> "Honest boundary" in [`../paper/measured/summary.md`](../paper/measured/summary.md).

---

## Verified artifact inventory (the paper)

Cross-checked against the manuscript at `depth_not_length_FIXED.tex` and the digest:

| Artifact kind | Count |
|---|---|
| `\begin{table}` environments | **69** (many are appendix `T001`–`T103` dumps) |
| `\begin{figure}` environments | **66** (many pair two side-by-side images) |
| `\includegraphics` includes | **~80** (80 includes; 78 unique filenames) |
| Figure files on disk (`figures/`) | **108** |

The repo's own figure pipeline emits a smaller, curated set: see
[`../results/FIGURES.md`](../results/FIGURES.md) (43 measured visualizations) and the
rendered artifacts under `results/figures/` and `results/paper_artifacts/`.

---

## The `membench` command surface

All flags below are read directly from [`../src/membench/cli.py`](../src/membench/cli.py)
and the defaults in [`../configs/default.yaml`](../configs/default.yaml). Everything runs
offline and deterministically by default.

| Command | What it does | Key flags (defaults) |
|---|---|---|
| `membench theory` | Predict the crossover depth `d*` from parameters | `--rho 0.5` `--q 0.97` `--s0 0.9` `--maintenance 0.2` `--value 1.0` |
| `membench run` | Run one `dataset × arms` sweep, write scored rows, print the depth-crossover table | `--dataset synthetic` `--budget 150` `--per-depth 10` `--seed 0` `--out results/rows.jsonl` |
| `membench sweep` | Run the full sweep (headline arms + dcbench/swebench ablation) | `--budget 150` `--per-depth 10` `--seed 0` `--out results/rows.jsonl` |
| `membench sweep-config` | Config-driven sweep (per-dataset budget) + a run manifest | `--config configs/default.yaml` `--out-dir results` `--created-at <iso>` |
| `membench tables` | Print headline compliance per `(dataset, arm)` | `--src results/rows.jsonl` |
| `membench report` | Write the full Markdown report (summary, headline, crossover, ablation, competitor matrix) | `--src results/rows.jsonl` `--out results/report.md` |
| `membench figures` | Render the depth-crossover figure | `--src results/rows.jsonl` `--out-dir figures` |

The fairness lock lives in the config: `budget_tokens_by_dataset` keys the token budget by
**dataset, never by arm** (`synthetic: 150`, `dcbench: 250`, `swebench: 500`,
`longmemeval: 8000`), so every arm sees the same budget for a given dataset. The default
arm set is `[none, bm25, tfidf, dense, hybrid_rrf, rerank_ce, raptor, brief_graph_3hop]`.

### One command for everything

[`../scripts/reproduce.sh`](../scripts/reproduce.sh) chains the whole pipeline
(`uv sync` -> build the native kernel -> offline test gate -> `membench run` on synthetic
-> `tables` -> `report` -> `figures`):

```sh
scripts/reproduce.sh
```

---

## Paper figure/table -> command map

For each representative paper artifact: the command that regenerates the corresponding
repo artifact, the **repo-measured** value (with its source file), and the
**paper-reported** value (from the manuscript). The two are listed separately; they are
not the same measurement and are not expected to match exactly.

### 1. Depth crossover

Paper artifacts: `fig:crossover` (`F006`/`F009`), `tab:slope` (compliance by depth),
`sec:cross` / `sec:resmech-depth`.

```sh
uv run membench run --dataset synthetic --budget 150 --out results/rows.jsonl
# the command prints the depth-crossover (full-chain recovery) table per arm
uv run membench figures --src results/rows.jsonl --out-dir figures   # -> figures/depth_crossover.png
uv run membench theory  --rho 0.5 --q 0.97                           # predicted d*
```

- **Repo-measured** (synthetic compliance by depth d1/d2/d3, from
  [`../paper/measured/dossier.md`](../paper/measured/dossier.md) Eval 16):
  `brief_graph_3hop` 0.97 / 1.00 / 1.00; `dense` 0.93 / 0.88 / 0.68; `raptor`
  0.47 / 0.23 / 0.05.
- **Paper-reported** (`tab:slope`, compliance by depth, synthetic/Claude): Brief
  0.950 / 0.975 / 0.875 (slope −0.075, least decay); dense 0.925 / 0.900 / 0.750
  (slope −0.175); `Δ(Brief − best-sim)` = +0.025 / +0.025 / +0.075 at d=1/2/3.
- **Crossover depth**: paper-reported predicted `d*=2` for margin `τ∈(0.50, 0.79]`
  (closed-form `d*=1`, the reported `d*=2` is flagged calibrated post-hoc). The
  `membench theory` command computes this `d*` from the decay/structure parameters.

### 2. Compliance by arm

Paper artifacts: `fig:bar_syn` (`F001`), `tab:synthall` (synthetic, all metrics).

```sh
uv run membench tables --src results/rows.jsonl   # headline compliance per (dataset, arm)
uv run membench sweep  --out results/rows.jsonl   # full sweep + four-arm ablation summary
```

- **Repo-measured** (synthetic compliance by arm,
  [`../paper/measured/dossier.md`](../paper/measured/dossier.md) Eval 1):
  `brief_graph_3hop` 0.99, `bm25` 0.89, `tfidf` 0.88, `dense` 0.82, `hybrid_rrf` 0.86,
  `rerank_ce` 0.67, `raptor` 0.25, `none` 0.00.
- **Paper-reported** (`tab:synthall`, arm-level mean over 120 tasks): Brief 0.933,
  tfidf 0.883, bm25 0.875, dense 0.858, raptor 0.817, none 0.025.

### 3. Return on Tokens (RoT)

Paper artifacts: `fig:rotbar` (`F096`), `sec:resmech-econ`, `sec:tokecon`. RoT_i =
compliance_i / tokens_i.

```sh
uv run membench report --src results/rows.jsonl --out results/report.md
# the report's executive summary carries the Return-on-Tokens headline
```

- **Repo-measured** (RoT = compliance per 1k tokens, all data,
  [`../paper/measured/dossier.md`](../paper/measured/dossier.md) Eval 51):
  `brief_graph_3hop` 0.507, `tfidf` 0.473, `bm25` 0.461, `hybrid_rrf` 0.445,
  `dense` 0.438, `rerank_ce` 0.378, `raptor` 0.198.
- **Paper-reported** (`fig:rotbar`, return-on-tokens by arm): typed 0.572, tfidf 0.532,
  bm25 0.521, hybrid_rrf 0.504, dense 0.496, rerank_ce 0.419, raptor 0.212, none 0.063.
- **Paper-reported** token-economics headline (`sec:tokecon`): Brief is cheaper in
  **80.0%** (2880/3600) of matchups, sessions collapse to two or three turns at
  ~60% fewer tokens.

### 4. Competitor matrix

Paper artifacts: `tab:stdbench` (unified harness), `tab:landscape` /
`fig:competitor` (`F052`), `F106_stdbench.png`. In the repo, the report's competitor
matrix is a two-tier object: measured arms vs. cited vendor claims
(`load_vendor_claims()` from `competitors/`).

```sh
uv run membench report --src results/rows.jsonl --out results/report.md
# emits "## Competitor matrix" with a measured tier and a vendor-reported tier
```

- **Repo-measured / vendor-reported (cited)**: the committed landscape lives in
  [`../results/data/competitor_landscape.csv`](../results/data/competitor_landscape.csv)
  and [`../results/competitor_landscape.md`](../results/competitor_landscape.md). It is
  tagged by `status` per row, e.g. Mem0 LoCoMo LLM-judge 66.9 (baseline 52.9,
  peer-reviewed); Zep Deep Memory Retrieval 94.8 (baseline 93.4, peer-reviewed);
  GraphRAG comprehensiveness win-rate 77.5 vs 27.0 (peer-reviewed). These are vendors'
  published numbers, cited — **never mixed** with the measured arm tier.
- **Paper-reported** (`tab:stdbench`, all competitors on one unified harness): LoCoMo
  LLM-judge Brief 87.6 (next: RAPTOR 84.7); DMR fact-F1 Brief 94.2 (next: Kluris 87.0);
  SWE-ContextBench resolution Brief 47.3% (next: Unabyss 37.6%). `tab:landscape` lists
  competitors' *published* headlines for orientation only (Mem0 66.9, Zep 94.8,
  GraphRAG 77.5, Supermemory 59.7, MemGPT 93.4).

### 5. Mediation

Paper artifacts: `fig:mediation` (`F038`), `sec:resmech-mediation` — the compliance gain
mediated through recall (Brief vs dense, synthetic).

```sh
uv run membench run --dataset synthetic --budget 150 --out results/rows.jsonl
# scored rows in results/rows.jsonl carry per-row recall + compliance; the mediation/
# causal-mediation analysis is one of the statistics listed in ../README.md
```

- **Repo-measured** inputs (synthetic compliance, Brief vs dense, from
  [`../paper/measured/dossier.md`](../paper/measured/dossier.md) Eval 1): Brief 0.99,
  dense 0.82.
- **Paper-reported** (`fig:mediation`): total effect **+0.075** (Brief 0.933 vs
  dense 0.858), routed through recall.

> The mediation, forest, posterior, and reliability statistics are computed by the
> `stats`/`analysis` modules (see the statistics list in [`../README.md`](../README.md));
> the curated CLI (`theory`/`run`/`sweep`/`tables`/`report`/`figures`) renders the
> headline depth-crossover figure, while the full 43-figure measured set is catalogued in
> [`../results/FIGURES.md`](../results/FIGURES.md) and committed under `results/figures/`.

### 6. Critical-difference diagram

Paper artifacts: `fig:cd` (`F023`), `sec:resagg` (Friedman + Nemenyi, synthetic).

```sh
uv run membench run --dataset synthetic --budget 150 --out results/rows.jsonl
# rows.jsonl feeds the Friedman/Nemenyi critical-difference analysis (stats module)
```

- **Repo-measured** (synthetic d=3, from
  [`../paper/measured/dossier.md`](../paper/measured/dossier.md)): Friedman omnibus
  `chi2=137.9, p=1.37e-26` (Eval 39); mean rank (low=better) `brief_graph_3hop` 2.55,
  `bm25` 3.75, `tfidf` 3.75, `dense` 3.85; Nemenyi critical difference (α=0.05) = 1.66
  (Eval 40).
- **Paper-reported** (`fig:cd`): k=8 (7 df), critical value 14.07, typed mean rank
  ≈2.55. The repo's BCa 95% CI at synthetic d=3 (Eval 41) puts `brief_graph_3hop` at
  1.00 [1.00, 1.00] vs `dense` 0.68 [0.50, 0.78].

---

## Notes on faithful reproduction

- **Determinism**: the default offline path is deterministic (stub model + hashing
  embedding), so repeated `membench run --seed 0` calls reproduce each other. The
  paper's *measured* run used real backends (`claude-sonnet-4-6`; GPT-5.1 on synthetic
  for cross-model) and is not bit-reproducible without those backends and the exact model
  snapshots — the paper flags its friendly model names as non-reproducible identifiers.
- **Budgets**: when reproducing a specific dataset, match its budget from
  [`../configs/default.yaml`](../configs/default.yaml) (synthetic 150, dcbench 250,
  swebench 500), or run `membench sweep-config --config configs/default.yaml` to apply
  all per-dataset budgets at once and emit a `manifest.json`.
- **Provenance discipline**: when citing a number, carry its tier. Repo-measured cells in
  [`../results/METRICS.md`](../results/METRICS.md) are tagged `measured` / `modeled` /
  `vendor`; the two-tier separation is enforced in code (`assert_single_tier`, per
  [`../README.md`](../README.md)). Do not present a repo-measured value and a
  paper-reported value as the same measurement.
