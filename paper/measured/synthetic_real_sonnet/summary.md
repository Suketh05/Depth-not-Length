# Synthetic depth-crossover — measured on Claude Sonnet

- model: `claude-sonnet-4-6` (real Anthropic Messages API, offline=false)
- date: 2026-06-18 | code: membench @ `5a21a80` (+ live-run timeout fix in this PR)
- n = 40 tasks per (arm, depth); 960 attempts total; 0 errors after patch
- measured cost (token ledger): **$15.08**
- offline=false, deterministic dataset (seed=0); scoring = governing-decision compliance


## Depth-crossover (compliance, n=40/cell)

| arm | d=1 | d=2 | d=3 |
|---|---|---|---|
| brief_graph_3hop | 0.97 | 0.97 | 1.00 |
| bm25 | 0.97 | 1.00 | 0.70 |
| tfidf | 0.95 | 1.00 | 0.70 |
| dense | 0.93 | 0.88 | 0.68 |
| hybrid_rrf | 1.00 | 0.97 | 0.60 |
| rerank_ce | 0.93 | 0.70 | 0.38 |
| raptor | 0.47 | 0.23 | 0.05 |
| none | 0.00 | 0.00 | 0.00 |

## Significance at d=3 (brief_graph_3hop vs best similarity, unpaired)

- brief_graph_3hop = **1.00** [1.00, 1.00]  vs  best similarity `bm25` = 0.70 [0.53, 0.80] (95% BCa CI, non-overlapping)
- Bayesian P(brief > bm25) = **1.000**  ·  Cohen's h = **+1.16** (large)
- d=1 / d=2: brief is statistically tied with the best similarity baseline (the honest crossover — no advantage when shallow).

_Paired tests (McNemar/Friedman) require `task_id` on each scored row, which `ScoredRow` does not yet persist; the unpaired Bayesian/CI/effect-size battery above is valid and conclusive at this effect size._

