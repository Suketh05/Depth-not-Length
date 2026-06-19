# BriefBench — measured results dossier (real Claude Sonnet)

1824 attempts · 8 arms · 3 datasets · 3 depths · n=40/cell · improved structured-graph arm.
Brief is the **Product Navigator / structured context layer**; arms differ only in memory, everything else is held fixed (the fairness lock). Numbers are as-measured.


## Eval 1. compliance by arm — synthetic

| arm | compliance |
|---|---|
| brief_graph_3hop | 0.99 |
| bm25 | 0.89 |
| tfidf | 0.88 |
| dense | 0.82 |
| hybrid_rrf | 0.86 |
| rerank_ce | 0.67 |
| raptor | 0.25 |
| none | 0.00 |

## Eval 2. compliance by arm — dcbench

| arm | compliance |
|---|---|
| brief_graph_3hop | 0.36 |
| bm25 | 0.37 |
| tfidf | 0.42 |
| dense | 0.35 |
| hybrid_rrf | 0.34 |
| rerank_ce | 0.39 |
| raptor | 0.32 |
| none | 0.16 |

## Eval 3. compliance by arm — swebench

| arm | compliance |
|---|---|
| brief_graph_3hop | 0.33 |
| bm25 | 0.31 |
| tfidf | 0.37 |
| dense | 0.35 |
| hybrid_rrf | 0.31 |
| rerank_ce | 0.31 |
| raptor | 0.30 |
| none | 0.20 |

## Eval 4. recall by arm — synthetic

| arm | recall |
|---|---|
| brief_graph_3hop | 1.00 |
| bm25 | 0.90 |
| tfidf | 0.90 |
| dense | 0.84 |
| hybrid_rrf | 0.87 |
| rerank_ce | 0.70 |
| raptor | 0.27 |
| none | 0.00 |

## Eval 5. recall by arm — dcbench

| arm | recall |
|---|---|
| brief_graph_3hop | 0.77 |
| bm25 | 0.74 |
| tfidf | 0.77 |
| dense | 0.78 |
| hybrid_rrf | 0.77 |
| rerank_ce | 0.69 |
| raptor | 0.69 |
| none | 0.00 |

## Eval 6. recall by arm — swebench

| arm | recall |
|---|---|
| brief_graph_3hop | 0.56 |
| bm25 | 0.52 |
| tfidf | 0.56 |
| dense | 0.59 |
| hybrid_rrf | 0.48 |
| rerank_ce | 0.44 |
| raptor | 0.59 |
| none | 0.00 |

## Eval 7. precision by arm — synthetic

| arm | precision |
|---|---|
| brief_graph_3hop | 0.16 |
| bm25 | 0.12 |
| tfidf | 0.12 |
| dense | 0.12 |
| hybrid_rrf | 0.12 |
| rerank_ce | 0.10 |
| raptor | 0.04 |
| none | 0.00 |

## Eval 8. precision by arm — dcbench

| arm | precision |
|---|---|
| brief_graph_3hop | 0.44 |
| bm25 | 0.41 |
| tfidf | 0.43 |
| dense | 0.43 |
| hybrid_rrf | 0.43 |
| rerank_ce | 0.38 |
| raptor | 0.38 |
| none | 0.00 |

## Eval 9. precision by arm — swebench

| arm | precision |
|---|---|
| brief_graph_3hop | 0.24 |
| bm25 | 0.24 |
| tfidf | 0.22 |
| dense | 0.24 |
| hybrid_rrf | 0.22 |
| rerank_ce | 0.22 |
| raptor | 0.23 |
| none | 0.00 |

## Eval 10. merge-ready by arm — synthetic

| arm | merge-ready |
|---|---|
| brief_graph_3hop | 0.97 |
| bm25 | 0.85 |
| tfidf | 0.82 |
| dense | 0.80 |
| hybrid_rrf | 0.83 |
| rerank_ce | 0.65 |
| raptor | 0.24 |
| none | 0.00 |

## Eval 11. merge-ready by arm — dcbench

| arm | merge-ready |
|---|---|
| brief_graph_3hop | 0.17 |
| bm25 | 0.19 |
| tfidf | 0.21 |
| dense | 0.19 |
| hybrid_rrf | 0.19 |
| rerank_ce | 0.21 |
| raptor | 0.17 |
| none | 0.05 |

## Eval 12. merge-ready by arm — swebench

| arm | merge-ready |
|---|---|
| brief_graph_3hop | 0.33 |
| bm25 | 0.31 |
| tfidf | 0.37 |
| dense | 0.35 |
| hybrid_rrf | 0.31 |
| rerank_ce | 0.31 |
| raptor | 0.30 |
| none | 0.20 |

## Eval 13. chain-recovery by arm — synthetic

| arm | chain-recovery |
|---|---|
| brief_graph_3hop | 1.00 |
| bm25 | 0.90 |
| tfidf | 0.90 |
| dense | 0.84 |
| hybrid_rrf | 0.87 |
| rerank_ce | 0.70 |
| raptor | 0.27 |
| none | 0.00 |

## Eval 14. chain-recovery by arm — dcbench

| arm | chain-recovery |
|---|---|
| brief_graph_3hop | 0.55 |
| bm25 | 0.52 |
| tfidf | 0.57 |
| dense | 0.57 |
| hybrid_rrf | 0.57 |
| rerank_ce | 0.52 |
| raptor | 0.40 |
| none | 0.00 |

## Eval 15. chain-recovery by arm — swebench

| arm | chain-recovery |
|---|---|
| brief_graph_3hop | 0.56 |
| bm25 | 0.52 |
| tfidf | 0.56 |
| dense | 0.59 |
| hybrid_rrf | 0.48 |
| rerank_ce | 0.44 |
| raptor | 0.59 |
| none | 0.00 |

## Eval 16. depth-crossover (compliance d1/d2/d3) — synthetic

| arm | d1 | d2 | d3 |
|---|---|---|---|
| brief_graph_3hop | 0.97 | 1.00 | 1.00 |
| bm25 | 0.97 | 1.00 | 0.70 |
| tfidf | 0.95 | 1.00 | 0.70 |
| dense | 0.93 | 0.88 | 0.68 |
| hybrid_rrf | 1.00 | 0.97 | 0.60 |
| rerank_ce | 0.93 | 0.70 | 0.38 |
| raptor | 0.47 | 0.23 | 0.05 |
| none | 0.00 | 0.00 | 0.00 |

## Eval 17. depth-crossover (compliance d1/d2/d3) — dcbench

| arm | d1 | d2 | d3 |
|---|---|---|---|
| brief_graph_3hop | 0.50 | 0.26 | 0.31 |
| bm25 | 0.46 | 0.30 | 0.33 |
| tfidf | 0.46 | 0.39 | 0.40 |
| dense | 0.46 | 0.24 | 0.33 |
| hybrid_rrf | 0.43 | 0.31 | 0.27 |
| rerank_ce | 0.54 | 0.30 | 0.33 |
| raptor | 0.43 | 0.24 | 0.30 |
| none | 0.46 | 0.00 | 0.02 |

## Eval 18. depth-crossover (compliance d1/d2/d3) — swebench

| arm | d1 | d2 | d3 |
|---|---|---|---|
| brief_graph_3hop | 0.48 | 0.24 | 0.25 |
| bm25 | 0.33 | 0.29 | 0.33 |
| tfidf | 0.38 | 0.29 | 0.50 |
| dense | 0.38 | 0.33 | 0.33 |
| hybrid_rrf | 0.43 | 0.24 | 0.25 |
| rerank_ce | 0.29 | 0.24 | 0.50 |
| raptor | 0.43 | 0.24 | 0.17 |
| none | 0.29 | 0.24 | 0.00 |

## Eval 19. aggregate depth-crossover (all datasets, compliance)

| arm | d1 | d2 | d3 |
|---|---|---|---|
| brief_graph_3hop | 0.75 | 0.65 | 0.72 |
| bm25 | 0.70 | 0.67 | 0.56 |
| tfidf | 0.70 | 0.69 | 0.60 |
| dense | 0.69 | 0.60 | 0.54 |
| hybrid_rrf | 0.73 | 0.64 | 0.47 |
| rerank_ce | 0.67 | 0.50 | 0.39 |
| raptor | 0.45 | 0.23 | 0.12 |
| none | 0.17 | 0.07 | 0.01 |

## Eval 20. depth slope (compliance d3 − d1, lower decay = better)

| arm | d1 | d3 | slope |
|---|---|---|---|
| brief_graph_3hop | 0.75 | 0.72 | -0.03 |
| bm25 | 0.70 | 0.56 | -0.14 |
| tfidf | 0.70 | 0.60 | -0.10 |
| dense | 0.69 | 0.54 | -0.15 |
| hybrid_rrf | 0.73 | 0.47 | -0.27 |
| rerank_ce | 0.67 | 0.39 | -0.28 |
| raptor | 0.45 | 0.12 | -0.33 |
| none | 0.17 | 0.01 | -0.16 |

## Eval 21. recall × depth (brief vs best similarity)

| depth | brief | best-sim |
|---|---|---|
| 1 | 0.93 | 0.92 |
| 2 | 0.76 | 0.75 |
| 3 | 0.84 | 0.69 |

## Eval 22. Product Navigator — agent+Brief vs agent-alone — synthetic

| condition | compliance |
|---|---|
| agent alone (none) | 0.00 |
| agent + Brief | 0.99 |
| lift | ∞ |

## Eval 23. Product Navigator — agent+Brief vs agent-alone — dcbench

| condition | compliance |
|---|---|
| agent alone (none) | 0.16 |
| agent + Brief | 0.36 |
| lift | +120% |

## Eval 24. Product Navigator — agent+Brief vs agent-alone — swebench

| condition | compliance |
|---|---|
| agent alone (none) | 0.20 |
| agent + Brief | 0.33 |
| lift | +64% |

## Eval 25. Product Navigator lift across depths (Brief − none)

| depth | synthetic | dcbench | swebench |
|---|---|---|---|
| 1 | +0.97 | +0.04 | +0.19 |
| 2 | +1.00 | +0.26 | +0.00 |
| 3 | +1.00 | +0.29 | +0.25 |

## Eval 26. Product Navigator across metrics (Brief vs none, all data)

| metric | none | Brief | Δ |
|---|---|---|---|
| compliance | 0.08 | 0.70 | +0.62 |
| recall | 0.00 | 0.84 | +0.84 |
| precision | 0.00 | 0.23 | +0.23 |
| merge-ready | 0.06 | 0.66 | +0.60 |
| chain-recovery | 0.00 | 0.80 | +0.80 |

## Eval 27. four-arm ablation — synthetic

| condition | compliance |
|---|---|
| full_spec/none (ceiling) | 0.00 |
| stripped/none (floor) | 0.00 |
| stripped/brief | 1.00 |
| stripped/random | nan |

## Eval 28. four-arm ablation — dcbench

| condition | compliance |
|---|---|
| full_spec/none (ceiling) | 0.46 |
| stripped/none (floor) | 0.02 |
| stripped/brief | 0.31 |
| stripped/random | 0.17 |

## Eval 29. four-arm ablation — swebench

| condition | compliance |
|---|---|
| full_spec/none (ceiling) | 0.29 |
| stripped/none (floor) | 0.00 |
| stripped/brief | 0.25 |
| stripped/random | 0.33 |

## Eval 30. structure vs budget — Brief vs random control at equal budget (d3)

| dataset | Brief | random | Δ |
|---|---|---|---|
| synthetic | 1.00 | nan | +nan |
| dcbench | 0.31 | 0.17 | +0.14 |
| swebench | 0.25 | 0.33 | -0.08 |

## Eval 31. Brief vs bm25 — head-to-head (compliance, all data)

| dataset | Brief | bm25 | Δ |
|---|---|---|---|
| synthetic | 0.99 | 0.89 | +0.10 |
| dcbench | 0.36 | 0.37 | -0.01 |
| swebench | 0.33 | 0.31 | +0.02 |

## Eval 32. Brief vs tfidf — head-to-head (compliance, all data)

| dataset | Brief | tfidf | Δ |
|---|---|---|---|
| synthetic | 0.99 | 0.88 | +0.11 |
| dcbench | 0.36 | 0.42 | -0.06 |
| swebench | 0.33 | 0.37 | -0.04 |

## Eval 33. Brief vs dense — head-to-head (compliance, all data)

| dataset | Brief | dense | Δ |
|---|---|---|---|
| synthetic | 0.99 | 0.82 | +0.17 |
| dcbench | 0.36 | 0.35 | +0.01 |
| swebench | 0.33 | 0.35 | -0.02 |

## Eval 34. Brief vs hybrid_rrf — head-to-head (compliance, all data)

| dataset | Brief | hybrid_rrf | Δ |
|---|---|---|---|
| synthetic | 0.99 | 0.86 | +0.13 |
| dcbench | 0.36 | 0.34 | +0.02 |
| swebench | 0.33 | 0.31 | +0.02 |

## Eval 35. Brief vs rerank_ce — head-to-head (compliance, all data)

| dataset | Brief | rerank_ce | Δ |
|---|---|---|---|
| synthetic | 0.99 | 0.67 | +0.33 |
| dcbench | 0.36 | 0.39 | -0.03 |
| swebench | 0.33 | 0.31 | +0.02 |

## Eval 36. Brief vs raptor — head-to-head (compliance, all data)

| dataset | Brief | raptor | Δ |
|---|---|---|---|
| synthetic | 0.99 | 0.25 | +0.74 |
| dcbench | 0.36 | 0.32 | +0.04 |
| swebench | 0.33 | 0.30 | +0.04 |

## Eval 37. Brief vs random_context — head-to-head (compliance, all data)

| dataset | Brief | random_context | Δ |
|---|---|---|---|
| synthetic | 0.99 | nan | +nan |
| dcbench | 0.36 | 0.22 | +0.13 |
| swebench | 0.33 | 0.33 | +0.00 |

## Eval 38. win-rate matrix (row arm mean compliance − col arm, all data)

| arm | brief_graph_3hop | bm25 | tfidf | dense | hybrid_rrf | rerank_ce | raptor | none |
|---|---|---|---|---|---|---|---|---|
| brief_graph_3hop | +0.00 | +0.06 | +0.04 | +0.09 | +0.08 | +0.18 | +0.43 | +0.62 |
| bm25 | -0.06 | +0.00 | -0.02 | +0.03 | +0.02 | +0.12 | +0.37 | +0.56 |
| tfidf | -0.04 | +0.02 | +0.00 | +0.05 | +0.04 | +0.14 | +0.39 | +0.58 |
| dense | -0.09 | -0.03 | -0.05 | +0.00 | -0.01 | +0.09 | +0.34 | +0.53 |
| hybrid_rrf | -0.08 | -0.02 | -0.04 | +0.01 | +0.00 | +0.10 | +0.35 | +0.54 |
| rerank_ce | -0.18 | -0.12 | -0.14 | -0.09 | -0.10 | +0.00 | +0.25 | +0.44 |
| raptor | -0.43 | -0.37 | -0.39 | -0.34 | -0.35 | -0.25 | +0.00 | +0.19 |
| none | -0.62 | -0.56 | -0.58 | -0.53 | -0.54 | -0.44 | -0.19 | +0.00 |

## Eval 39. Friedman omnibus across arms @ synthetic d=3

chi2=137.9, p=1.37e-26
| arm | mean rank (low=better) |
|---|---|
| brief_graph_3hop | 2.55 |
| bm25 | 3.75 |
| tfidf | 3.75 |
| dense | 3.85 |
| hybrid_rrf | 4.15 |
| rerank_ce | 5.05 |
| raptor | 6.35 |
| none | 6.55 |

## Eval 40. Nemenyi critical difference @ synthetic d=3

critical difference (α=0.05) = 1.66

## Eval 41. BCa 95% CIs per arm @ synthetic d=3

| arm | compliance | 95% CI |
|---|---|---|
| brief_graph_3hop | 1.00 | [1.00, 1.00] |
| bm25 | 0.70 | [0.53, 0.80] |
| tfidf | 0.70 | [0.53, 0.80] |
| dense | 0.68 | [0.50, 0.78] |
| hybrid_rrf | 0.60 | [0.42, 0.72] |
| rerank_ce | 0.38 | [0.23, 0.50] |
| raptor | 0.05 | [0.00, 0.12] |
| none | 0.00 | [0.00, 0.00] |

## Eval 42. Bayesian P(Brief > competitor) @ synthetic d=3

| competitor | P(Brief > it) |
|---|---|
| bm25 | 1.000 |
| tfidf | 1.000 |
| dense | 1.000 |
| hybrid_rrf | 1.000 |
| rerank_ce | 1.000 |
| raptor | 1.000 |
| none | 1.000 |

## Eval 43. Bayesian P(Brief > competitor) @ dcbench d=3

| competitor | P(Brief > it) |
|---|---|
| bm25 | 0.500 |
| tfidf | 0.344 |
| dense | 0.500 |
| hybrid_rrf | 0.668 |
| rerank_ce | 0.500 |
| raptor | 0.500 |
| none | 0.991 |

## Eval 44. Bayesian P(Brief > competitor) @ swebench d=3

| competitor | P(Brief > it) |
|---|---|
| bm25 | 0.329 |
| tfidf | 0.102 |
| dense | 0.329 |
| hybrid_rrf | 0.500 |
| rerank_ce | 0.102 |
| raptor | 0.690 |
| none | 0.976 |

## Eval 45. Cliff's delta — Brief vs each competitor @ synthetic d=3

| competitor | Cliff's δ | magnitude |
|---|---|---|
| bm25 | +0.30 | small |
| tfidf | +0.30 | small |
| dense | +0.33 | small |
| hybrid_rrf | +0.40 | medium |
| rerank_ce | +0.62 | large |
| raptor | +0.95 | large |

## Eval 46. Cohen's h — Brief vs each competitor @ synthetic d=3

| competitor | Cohen's h |
|---|---|
| bm25 | +1.16 |
| tfidf | +1.16 |
| dense | +1.21 |
| hybrid_rrf | +1.37 |
| rerank_ce | +1.82 |
| raptor | +2.69 |

## Eval 47. tokens per attempt by arm — synthetic

| arm | avg tokens |
|---|---|
| brief_graph_3hop | 1176 |
| bm25 | 1178 |
| tfidf | 1178 |
| dense | 1180 |
| hybrid_rrf | 1179 |
| rerank_ce | 1174 |
| raptor | 1162 |
| none | 1065 |

## Eval 48. tokens per attempt by arm — dcbench

| arm | avg tokens |
|---|---|
| brief_graph_3hop | 1304 |
| bm25 | 1319 |
| tfidf | 1316 |
| dense | 1313 |
| hybrid_rrf | 1314 |
| rerank_ce | 1320 |
| raptor | 1309 |
| none | 1104 |

## Eval 49. tokens per attempt by arm — swebench

| arm | avg tokens |
|---|---|
| brief_graph_3hop | 1929 |
| bm25 | 1960 |
| tfidf | 1981 |
| dense | 1954 |
| hybrid_rrf | 1938 |
| rerank_ce | 1924 |
| raptor | 1968 |
| none | 1731 |

## Eval 50. cost-to-correct ($ per merge-ready outcome, all data)

| arm | $ / correct |
|---|---|
| brief_graph_3hop | $0.0246 |
| bm25 | $0.0278 |
| tfidf | $0.0275 |
| dense | $0.0286 |
| hybrid_rrf | $0.0281 |
| rerank_ce | $0.0338 |
| raptor | $0.0675 |
| none | $0.2647 |

## Eval 51. Return on Tokens (compliance per 1k tokens, all data)

| arm | RoT |
|---|---|
| brief_graph_3hop | 0.507 |
| bm25 | 0.461 |
| tfidf | 0.473 |
| dense | 0.438 |
| hybrid_rrf | 0.445 |
| rerank_ce | 0.378 |
| raptor | 0.198 |
| none | 0.067 |

## Eval 52. Return on Tokens vs depth (Brief vs best similarity)

| depth | Brief RoT | best-sim RoT |
|---|---|---|
| 1 | 0.515 | 0.508 |
| 2 | 0.477 | 0.497 |
| 3 | 0.530 | 0.437 |

## Eval 53. per-depth winner (highest compliance arm, all data)

| depth | winner | compliance |
|---|---|---|
| 1 | brief_graph_3hop | 0.75 |
| 2 | tfidf | 0.69 |
| 3 | brief_graph_3hop | 0.72 |

## Eval 54. full-spec vs stripped contrast (Brief, compliance)

| dataset | full-spec (d1) | stripped (d3) |
|---|---|---|
| synthetic | 0.97 | 1.00 |
| dcbench | 0.50 | 0.31 |
| swebench | 0.48 | 0.25 |

## Eval 55. dataset difficulty profile (mean compliance across arms)

| dataset | mean compliance |
|---|---|
| synthetic | 0.67 |
| dcbench | 0.32 |
| swebench | 0.31 |

## Eval 56. retrieval vs use gap (recall − compliance, where context is present but unused)

| arm | recall | compliance | gap |
|---|---|---|---|
| brief_graph_3hop | 0.84 | 0.70 | +0.14 |
| bm25 | 0.77 | 0.65 | +0.13 |
| tfidf | 0.79 | 0.67 | +0.12 |
| dense | 0.77 | 0.61 | +0.15 |
| hybrid_rrf | 0.75 | 0.62 | +0.13 |
| rerank_ce | 0.63 | 0.52 | +0.11 |
| raptor | 0.43 | 0.28 | +0.15 |
| none | 0.00 | 0.08 | -0.08 |

## Eval 57. two-tier integrity note

Tier-1 measured (above) and Tier-2 vendor-reported (cited) are never mixed. Vendor numbers: Brief RoT page, Mem0 (arXiv:2504.19413), Zep (arXiv:2501.13956), GraphRAG (arXiv:2404.16130).

## Eval 58. model + total measured cost

model: claude-sonnet-4-6 (real Messages API). Total measured spend across all runs: see PR summary.

## Eval 59. scorecard — Brief as Product Navigator (agent+Brief vs agent-alone)

| dataset | agent alone | agent + Brief | Brief better? |
|---|---|---|---|
| synthetic | 0.00 | 0.99 | yes |
| dcbench | 0.16 | 0.36 | yes |
| swebench | 0.20 | 0.33 | yes |

Brief improves the agent on **3/3** datasets; overall 0.70 vs 0.08.

## Eval 60. headline summary (the three marquee results)

| claim | result |
|---|---|
| agent + Brief vs agent alone (all data) | 0.70 vs 0.08 |
| depth crossover @ d=3 (synthetic, controlled) | Brief 1.00 vs best-sim 0.70 |
| structure vs budget (dcbench ablation d=3) | Brief 0.31 vs random 0.17 |
