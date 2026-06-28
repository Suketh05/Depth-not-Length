# Competitive landscape — the paper's reported numbers

> **Source of truth.** All numbers here are transcribed VERBATIM from the paper
> *"Depth, Not Length."* The competitors' figures are their **own published headline
> numbers** as cited by the paper (paper table **`tab:landscape`**) — orientation only,
> **NOT a controlled head-to-head** (each is a different benchmark/metric and not directly
> comparable cell-to-cell). CSV: [`data/competitor_landscape.csv`](data/competitor_landscape.csv).

## Competitors' published headline numbers (`tab:landscape`)

| system | metric (as published) | score | vs base | benchmark | source | status |
|---|---|---|---|---|---|---|
| Mem0 | LoCoMo LLM-judge | 66.9 | 52.9 | LoCoMo | [arXiv:2504.19413](https://arxiv.org/abs/2504.19413) | peer-reviewed |
| Zep | Deep Memory Retrieval | 94.8 | 93.4 | DMR | [arXiv:2501.13956](https://arxiv.org/abs/2501.13956) | peer-reviewed |
| GraphRAG | QFS comprehensiveness | 77.5 | 27.0 | QFS comprehensiveness | [arXiv:2404.16130](https://arxiv.org/abs/2404.16130) | peer-reviewed |
| Supermemory | LoCoMo P@1 (self-rep.) | 59.7 | 34.4 | LoCoMo | [supermemory.ai](https://supermemory.ai/) | self-reported (unverified) |
| MemGPT | Deep Memory Retrieval | 93.4 | — | DMR | [arXiv:2310.08560](https://arxiv.org/abs/2310.08560) | peer-reviewed |

The takeaway is the industry-wide pattern the paper notes: a structured context/memory
layer beats the no-context / vanilla baseline everywhere.

## Brief on the same competitors, unified harness (`tab:stdbench`)

When the paper runs every system through one unified harness, Brief leads where it matters
(see [`results/METRICS.md`](METRICS.md) §5 and
[`data/paper_public_benchmarks.csv`](data/paper_public_benchmarks.csv)):

- LoCoMo LLM-judge: **Brief 87.6** (next best RAPTOR 84.7)
- DMR fact F1: **Brief 94.2** (next best Kluris 87.0)
- SWE-ContextBench resolution: **Brief 47.3%** (next best Unabyss 37.6%)
- Real-code pooled use factor κ: **Brief 0.703** (next best Unabyss 0.652)

> Note (from the paper): where the same vendor appears with a different number across
> tables — e.g. Mem0 66.9 on LoCoMo here vs. 24.2 on SWE-ContextBench — the gap is a
> change of benchmark and metric, not an inconsistency. Supermemory's figures are
> self-reported and not independently verified; the rest are peer-reviewed.
