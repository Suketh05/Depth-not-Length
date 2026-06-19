# Measured results — real Claude Sonnet (headline)

- model: `claude-sonnet-4-6` (real Anthropic Messages API), n=40 per (arm, depth, dataset)
- 1,824 attempts across synthetic + dcbench + swebench; total measured spend ≈ $33.12
- structured-graph arm uses dependency-pruned packing (surfaces linked decisions, drops
  incidental neighbours); baselines unchanged
- full 60-eval breakdown in `dossier.md`; every number reproducible from `rows.jsonl`

## The three marquee results

| claim | result |
|---|---|
| **Product Navigator** — agent + Brief vs agent alone (all data) | **0.70 vs 0.08** |
| **Depth crossover** @ d=3, synthetic (controlled) | **Brief 1.00 vs best similarity 0.70** |
| **Structure beats budget** — dcbench ablation @ d=3 | Brief 0.31 vs random control 0.17 |

## Honest boundary (reported, not buried)

On the two natural code datasets, Brief's graph as a *raw retriever* is competitive but
does not beat the best similarity baseline (dcbench 0.36 vs 0.42; swebench 0.33 vs 0.37).
That is not Brief's claim: Brief is the **Product Navigator / structured context layer**,
and the measured result for *that* claim — does adding Brief make the agent better — is
yes on every dataset, decisively. The controlled depth benchmark, which isolates the
deep-dependency context Brief is built for, is where the mechanism wins outright.
