# Paper artifacts — Batch 3 (INDEX): robustness suite

tables 48–52, figures 40–45. Distractor, corpus-scaling, noise-ratio, budget×noise.

| kind | n | family | angle | convention | tier | file |
|---|---|---|---|---|---|---|
| table | 48 | F | Distractor robustness: recall@150 vs K injected decoys (synthetic d=3) | RAGBench noise robustness | offline-recall | T048_distractor_recall.md |
| figure | 40 | F | Recall vs decoy count | degradation curve | offline-recall | F040_distractor_curve.png |
| table | 49 | F | Noise-retention score: recall(K=40)/recall(K=0) | robustness ratio | offline-recall | T049_noise_retention.md |
| figure | 41 | F | Noise-retention by arm | retention bar | offline-recall | F041_retention_bar.png |
| table | 50 | F | Corpus-scaling: recall@250 vs added corpus size | scaling robustness | offline-recall | T050_corpus_scaling.md |
| figure | 42 | F | Recall vs corpus size | scaling curve | offline-recall | F042_corpus_scaling_curve.png |
| table | 51 | F | Noise-ratio sensitivity: recall vs distractor:signal ratio | noise-ratio table | offline-recall | T051_noise_ratio.md |
| figure | 43 | F | Recall over budget × decoys — brief_graph_3hop | 2D robustness heatmap | offline-recall | F043_budget_distractor_brief_graph_3hop.png |
| figure | 44 | F | Recall over budget × decoys — dense | 2D robustness heatmap | offline-recall | F044_budget_distractor_dense.png |
| table | 52 | F | Robustness summary: recall under clean vs heavy noise (K=40, b=150) | robustness leaderboard | offline-recall | T052_robustness_summary.md |
| figure | 45 | F | Clean vs heavy-noise recall by arm | grouped robustness bar | offline-recall | F045_clean_vs_noise.png |
