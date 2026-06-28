# Results — headline numbers, verbatim

This page reproduces the project's headline numbers **verbatim, with strict attribution**.
Every digit on this page is **copied** from a source file — none is computed, rounded, or
re-derived here. Two kinds of numbers live below and they are **kept deliberately separate**:

- **(A) Repo-measured** — transcribed from the committed result files in
  [`results/`](../results/) (our offline harness data, plus clearly-tagged `vendor`/`modeled`
  cells where the source file marks them as such). Each table cites its source file.
- **(B) Paper-reported** — transcribed from the paper digest of
  `depth_not_length_FIXED.tex` ("Depth, Not Length"). These are labeled **paper-reported**
  throughout.

> **Provenance note (read this first).** The repo-measured numbers and the paper-reported
> numbers come from **distinct runs with distinct provenance** (different harness snapshots,
> different pools, and in some cases modeled vs. published figures). They are **intentionally
> not merged or reconciled** on this page. Where a "same-sounding" quantity appears in both
> sections (e.g. pooled real-code compliance), the values differ because the underlying cuts
> differ — see the call-outs. Treat each section on its own terms.

Provenance tiers used by the source files: `measured` = our harness data · `vendor` =
published competitor numbers, cited · `modeled` = derived (method stated in the source note;
not a measurement). Modeled and vendor cells are tagged inline and are **never** presented as
measured.

---

## A. Repo-measured

All numbers in this section are copied from committed files under [`results/`](../results/).
Source file is cited per table.

### A.1 Head-to-head summary

Source: [`results/EVALS.md`](../results/EVALS.md)

> **440 evals** across 9 families, on Claude Sonnet + GPT-5.1.
> **Brief wins 312/440 head-to-head cuts.**

Per-family win counts (verbatim from the family headers in `results/EVALS.md`):

| family | evals | Brief wins |
|---|---|---|
| F1_product_navigator | 120 | 116 |
| F2_depth_crossover | 72 | 24 |
| F3_vs_competitor | 120 | 81 |
| F4_bayesian_superiority | 28 | 19 |
| F4_cliffs_delta | 24 | 15 |
| F4_cohens_h | 24 | 15 |
| F4_bca_ci | 12 | 7 |
| F5_return_on_tokens | 32 | 28 |
| F6_ablation | 8 | 7 |

### A.2 Compliance by dataset and arm (Claude, d=all)

Source: [`results/METRICS.md`](../results/METRICS.md) — bucket `2_task_outcome` (all `measured`)

| metric | Brief | dense (best baseline) | BM25 | none (no context) | random_context |
|---|---|---|---|---|---|
| compliance@synthetic | 0.9917 | 0.825 | 0.8917 | 0.0 | _(blank in source)_ |
| compliance@dcbench | 0.3571 | 0.3452 | 0.3651 | 0.1627 | 0.2222 |
| compliance@swebench | 0.3333 | 0.3519 | 0.3148 | 0.2037 | 0.3333 |
| compliance@all | 0.7037 | 0.6134 | 0.6451 | 0.0826 | 0.2847 |

### A.3 Recall by dataset and arm (Claude, d=all)

Source: [`results/METRICS.md`](../results/METRICS.md) — bucket `1_retrieval_quality` (all `measured`)

| metric | Brief | dense (best baseline) | BM25 | none (no context) | random_context |
|---|---|---|---|---|---|
| recall@synthetic | 1.0 | 0.8417 | 0.9 | 0.0 | _(blank in source)_ |
| recall@dcbench | 0.7698 | 0.7817 | 0.7421 | 0.0 | 0.1984 |
| recall@swebench | 0.5556 | 0.5926 | 0.5185 | 0.0 | 0.3704 |
| recall@all | 0.8441 | 0.7677 | 0.7739 | 0.0 | 0.2951 |

### A.4 Return-on-tokens by dataset and arm (Claude, d=all)

Source: [`results/METRICS.md`](../results/METRICS.md) — bucket `3_token_economics` (all `measured`)

| metric | Brief | dense (best baseline) | BM25 | none (no context) | random_context |
|---|---|---|---|---|---|
| return_on_tokens@synthetic | 0.8451 | 0.6989 | 0.7566 | 0.0 | _(blank in source)_ |
| return_on_tokens@dcbench | 0.2709 | 0.26 | 0.2751 | 0.151 | 0.1654 |
| return_on_tokens@swebench | 0.2012 | 0.2286 | 0.1904 | 0.1362 | 0.2076 |
| return_on_tokens@all | 0.5725 | 0.496 | 0.5214 | 0.0634 | 0.1891 |

### A.5 Compliance by depth — synthetic (Claude)

Source: [`results/METRICS.md`](../results/METRICS.md) — bucket `4_depth_robustness` (all `measured`)

| depth | Brief | dense (best baseline) | BM25 | none (no context) |
|---|---|---|---|---|
| compliance_synthetic_d1 | 0.975 | 0.925 | 0.975 | 0.0 |
| compliance_synthetic_d2 | 1.0 | 0.875 | 1.0 | 0.0 |
| compliance_synthetic_d3 | 1.0 | 0.675 | 0.7 | 0.0 |

### A.6 Cross-model compliance, depth-3 (synthetic)

Source: [`results/METRICS.md`](../results/METRICS.md) — bucket `7_cross_model` (all `measured`)

| arm | Claude (d3) | GPT (d3) |
|---|---|---|
| Brief | 1.0 | 1.0 |
| dense (best baseline) | 0.675 | 0.6 |
| BM25 | 0.7 | _(not listed at d3 in source)_ |

### A.7 Statistical confidence, depth-3 (synthetic, Claude)

Source: [`results/METRICS.md`](../results/METRICS.md) — bucket `6_statistical_confidence` (all `measured`)

| quantity | value |
|---|---|
| brief_ci_low_synthetic_d3 | 1.0 |
| brief_ci_high_synthetic_d3 | 1.0 |
| P_brief_gt_dense_synthetic | 1.0 |
| cohens_h_vs_dense_synthetic | 1.2132 |
| P_brief_gt_bm25_synthetic | 1.0 |
| cohens_h_vs_bm25_synthetic | 1.1593 |

### A.8 Competitor head-to-head (vendor + modeled cells, clearly tagged)

Source: [`results/METRICS.md`](../results/METRICS.md) — bucket `10_competitor_head_to_head`

| metric | system | value | tier | note |
|---|---|---|---|---|
| Mem0_headline_score | Mem0 | 66.9 | vendor | arXiv:2504.19413 |
| Zep_headline_score | Zep | 94.8 | vendor | arXiv:2501.13956 |
| GraphRAG_headline_score | GraphRAG | 77.5 | vendor | arXiv:2404.16130 |
| Supermemory_headline_score | Supermemory | 59.7 | vendor | supermemory.ai (unverified) |
| Brief_compliance_ours | Brief | 0.7037 | **measured** | claude all-data |
| Mem0_modeled_compliance_ours | Mem0 | 0.22 | **modeled** | modeled from published lift; not a measured run |
| Zep_modeled_compliance_ours | Zep | 0.265 | **modeled** | modeled from published lift; not a measured run |
| GraphRAG_modeled_compliance_ours | GraphRAG | 0.58 | **modeled** | modeled from published lift; not a measured run |
| Supermemory_modeled_compliance_ours | Supermemory | 0.333 | **modeled** | modeled from published lift; not a measured run |

### A.9 SWE-ContextBench — repo's leaderboard row (Brief is `modeled`, not measured)

Source: [`results/swe_context_bench.md`](../results/swe_context_bench.md)

| system | task resolution % | token cost | tier |
|---|---|---|---|
| Oracle Summary Learning (upper bound) | 34.34 | 6.66 | vendor |
| Supermemory | 30.30 | 5.04 | vendor (self-reported) |
| OpenViking | 29.20 | 4.20 | vendor |
| **Brief (modeled)** | **27** (range 24–31) | **~4.3** | **modeled** |
| Mem0 | 24.24 | 4.72 | vendor |

> **Cross-section call-out:** the repo's SWE-ContextBench Brief row is **modeled at 27%
> (range 24–31)**. The paper reports a **47.3%** SWE-ContextBench resolution for Brief on a
> unified harness (see §B). These are **different provenance** (repo `modeled` proxy vs.
> paper-reported on a different harness) and are not the same measurement.

---

## B. Paper-reported

All numbers in this section are **paper-reported** — transcribed verbatim from the digest of
the paper LaTeX `depth_not_length_FIXED.tex` ("Depth, Not Length: A Benchmark and Theory of
Memory, Retrieval, and Decision Compliance in LLM Coding Agents"). They are **not** repo
harness measurements and are kept separate from §A on purpose.

### B.1 The ~10 headline numbers (paper-reported)

| # | quantity | value (paper-reported) | paper context |
|---|---|---|---|
| 1 | LoCoMo LLM-judge accuracy (%) | **87.6** | best on unified harness (next: RAPTOR 84.7) |
| 2 | DMR fact F1 (%) | **94.2** | best on unified harness (next: Kluris 87.0) |
| 3 | SWE-ContextBench resolution (%) | **47.3%** | best on unified harness (next: Unabyss 37.6) |
| 4 | Supersession, current-ranked-first (%) | **92.3%** | typed store vs. similarity 64–69% band (chance 50%) |
| 5 | Token session-token win rate | **80.0%** | 2880 / 3600 matchups, 0 ties |
| 6 | Depth crossover | **d\*=2** | margin τ∈(0.50, 0.79]; closed-form crossover d\*=1 (calibrated post-hoc) |
| 7 | Real-code use factor κ (typed store) | **0.703** | only arm clearing the 0.56–0.64 similarity κ band |
| 8 | Pooled real-code recall (typed store) | **0.667** | #1 on real code (0.032 above dense 0.635) |
| 9 | Pooled real-code compliance (typed store) | **0.469** | #1 on real code |
| 10 | Synthetic compliance (typed store) | **0.933** | vs bm25 0.875 / tfidf 0.883 / dense 0.858 / raptor 0.817 / none 0.025 |

Additional paper-reported headline figures (same source):

- HotpotQA (paper-reported, unified harness): recall@5 **0.783**, nDCG@10 **0.987**,
  MRR **0.981** (all best-in-class); supporting-fact recall **0.503** (loses to Zep 0.529).
- Cross-model replication (paper-reported, synthetic, GPT-5.1): same arm ordering as Claude,
  Pearson **r≈0.999** (n=8), per-arm deltas ≤0.02, Δ(3)=**+0.075** reproduced.
- Unbiased scorecard (paper-reported, over 41 axes): **22 W / 14 m / 5 l**.
- Synthetic depth slope (paper-reported, Claude): Brief **−0.075** (least decay) vs similarity
  −0.125 to −0.200; Δ(Brief − best-sim) **+0.025 / +0.025 / +0.075** at d=1/2/3.
- Context-free floor (paper-reported): `none` compliance **0.000–0.025** synthetic,
  **0.073** pooled real-code.

### B.2 Public benchmarks, best vs. next (paper-reported)

Transcribed from the paper's `F106_stdbench` / `tab:stdbench` figure caption.

| benchmark | Brief (paper-reported) | next-best competitor (paper-reported) |
|---|---|---|
| LoCoMo LLM-judge acc (%) | 87.6 | RAPTOR 84.7 |
| DMR fact F1 (%) | 94.2 | Kluris 87.0 |
| SWE-ContextBench resolution (%) | 47.3% | Unabyss 37.6 |

### B.3 Pooled real-code (paper-reported), sorted by recall

Transcribed from the paper's `tab:realret` (pooled dcbench+swebench, Claude).

| arm | recall P_ret | compliance P_comply | precision | use factor κ |
|---|---|---|---|---|
| **Brief** | **0.667** | **0.469** | **0.385** | **0.703** |
| dense | 0.635 | 0.406 | 0.323 | 0.639 |
| hybrid_rrf | 0.615 | 0.396 | 0.323 | 0.644 |
| tfidf | 0.604 | 0.385 | 0.344 | 0.637 |
| bm25 | 0.604 | 0.344 | 0.302 | 0.570 |
| rerank_ce | 0.583 | 0.354 | 0.312 | 0.607 |
| raptor | 0.573 | 0.323 | 0.281 | 0.564 |
| none | 0.219 | 0.073 | 0.052 | — |

> **Cross-section call-out:** the paper's pooled real-code compliance (Brief **0.469**) pools
> **dcbench + swebench only**. The repo-measured `compliance@all` (Brief **0.7037**, §A.2)
> pools **synthetic + dcbench + swebench**. Different pools → different numbers; do not equate
> them.

---

## Sources

- Repo-measured (§A): [`results/METRICS.md`](../results/METRICS.md),
  [`results/EVALS.md`](../results/EVALS.md),
  [`results/swe_context_bench.md`](../results/swe_context_bench.md),
  [`results/competitor_landscape.md`](../results/competitor_landscape.md).
- Paper-reported (§B): paper LaTeX `depth_not_length_FIXED.tex` ("Depth, Not Length"),
  headline-results inventory. These figures are not reproduced from the repo harness.

_No number on this page was invented, estimated, or recomputed — each digit is copied verbatim
from the cited source._
</content>
</invoke>
