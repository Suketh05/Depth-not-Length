# Dossier (paper-reported)

Full per-arm / per-benchmark breakdown, transcribed verbatim from the paper. Each block
names its source table label; the same data is in [`../../results/data/`](../../results/data)
as `paper_*.csv`. "Brief" = the `brief_graph_3hop` typed decision-graph arm. Nothing here
is measured by this repository.

## Real code — pooled dcbench + swebench (Claude) — `tab:realret`

| arm | recall (P_ret) | compliance (P_comply) | precision | use factor κ |
|---|---|---|---|---|
| **brief** | **0.667** | **0.469** | **0.385** | **0.703** |
| dense | 0.635 | 0.406 | 0.323 | 0.639 |
| hybrid_rrf | 0.615 | 0.396 | 0.323 | 0.644 |
| tfidf | 0.604 | 0.385 | 0.344 | 0.637 |
| bm25 | 0.604 | 0.344 | 0.302 | 0.570 |
| rerank_ce | 0.583 | 0.354 | 0.312 | 0.607 |
| raptor | 0.573 | 0.323 | 0.281 | 0.564 |
| none | 0.219 | 0.073 | 0.052 | — |

Brief's κ = 0.703 is the only arm to clear the 0.56–0.64 use-factor band.

## Synthetic depth suite (Claude) — `tab:synthall`

| arm | compliance |
|---|---|
| **brief** | **0.933** |
| tfidf | 0.883 |
| bm25 | 0.875 |
| dense | 0.858 |
| raptor | 0.817 |
| none | 0.025 |

(`tab:synthall` does not list `hybrid_rrf` / `rerank_ce`; their depth behaviour appears in
the slope table below.)

## Depth slope (synthetic, Claude) — `tab:slope`

- Brief decays least: slope **−0.075** vs the similarity band **−0.125 to −0.200**.
- Brief minus best-similarity compliance gap by depth: **+0.025 / +0.025 / +0.075** at
  d = 1 / 2 / 3.

## Supersession — current-decision-ranked-first %, chance = 50% — `tab:super`

| system | current-first % |
|---|---|
| **brief** | **92.3** |
| dense | 68.7 |
| tfidf | 65.8 |
| bm25 | 64.1 |

## Public benchmarks (unified harness) — `tab:stdbench`, `tab:t101pub`

| benchmark | metric | Brief | next best |
|---|---|---|---|
| LoCoMo | LLM-judge | **87.6** | RAPTOR 84.7 |
| DMR | fact-F1 | **94.2** | Kluris 87.0 |
| SWE-ContextBench | resolution % | **47.3** | Unabyss 37.6 |
| HotpotQA | recall@5 | **0.783** | best-in-class |
| HotpotQA | nDCG@10 | **0.987** | best-in-class |
| HotpotQA | MRR | **0.981** | best-in-class |
| HotpotQA | supporting-fact recall | 0.503 | Zep 0.529 (Brief loses) |

## Token economics — `tab:tok_context_winrate`, `tab:tok_context_by_competitor`

- Brief is the cheaper session in **80.0%** of matchups (2880 / 3600, 0 ties).
- Per-competitor win rate ranges **68.6%–86.7%**.

## Theory calibration — `prop:cross`

- Crossover gap `g(d) = q^d − s₀^d·ρ^{d(d+1)/2}`; calibrated at `s₀ = 0.70, ρ = 0.67,
  q = 0.97` → `g(1) ≈ 0.50`, `g(2) ≈ 0.79`.
- Closed-form crossover (smallest `d` with `g > 0`) is **1**; the headline **d⋆ = 2** is
  the calibrated value for overhead margin `τ ∈ (0.50, 0.79]`.

## Unbiased scorecard (41 axes) — `tab:scorecard`

| outcome | count |
|---|---|
| win | 22 |
| marginal (near-tie) | 14 |
| loss | 5 |
