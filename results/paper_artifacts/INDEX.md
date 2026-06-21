# Paper artifacts — Batch 1 (INDEX)

32 tables + 30 figures. Each row = one distinct angle.

| kind | n | family | angle | convention | tier | file |
|---|---|---|---|---|---|---|
| table | 1 | A | Compliance leaderboard — synthetic (Claude) | Mem0/HELM main table | measured | T001_leaderboard_compliance_synthetic.md |
| table | 2 | A | Compliance leaderboard — dcbench (Claude) | Mem0/HELM main table | measured | T002_leaderboard_compliance_dcbench.md |
| table | 3 | A | Compliance leaderboard — swebench (Claude) | Mem0/HELM main table | measured | T003_leaderboard_compliance_swebench.md |
| table | 4 | A | Compliance leaderboard — all datasets (Claude) | HELM aggregate | measured | T004_leaderboard_compliance_all.md |
| table | 5 | A | Compliance leaderboard — synthetic (GPT-5.1) | cross-model leaderboard | measured | T005_leaderboard_compliance_gpt.md |
| table | 6 | A | Recall leaderboard — synthetic | BEIR-style retrieval table | measured | T006_leaderboard_recall_synthetic.md |
| table | 7 | A | Precision leaderboard — synthetic | BEIR-style retrieval table | measured | T007_leaderboard_precision_synthetic.md |
| table | 8 | A | Recall leaderboard — dcbench | BEIR-style retrieval table | measured | T008_leaderboard_recall_dcbench.md |
| table | 9 | A | Precision leaderboard — dcbench | BEIR-style retrieval table | measured | T009_leaderboard_precision_dcbench.md |
| table | 10 | A | Recall leaderboard — swebench | BEIR-style retrieval table | measured | T010_leaderboard_recall_swebench.md |
| table | 11 | A | Precision leaderboard — swebench | BEIR-style retrieval table | measured | T011_leaderboard_precision_swebench.md |
| table | 12 | A | Merge-ready (correct) leaderboard — all data | outcome table | measured | T012_leaderboard_mergeready.md |
| table | 13 | A | Chain-recovery leaderboard — all data | outcome table | measured | T013_leaderboard_chainrec.md |
| figure | 1 | A | Compliance bar — synthetic | grouped bar | measured | F001_bar_compliance_synthetic.png |
| figure | 2 | A | Compliance bar — dcbench | grouped bar | measured | F002_bar_compliance_dcbench.png |
| figure | 3 | A | Compliance bar — swebench | grouped bar | measured | F003_bar_compliance_swebench.png |
| figure | 4 | A | Multi-metric radar Brief vs dense — claude synthetic | radar/spider (HELM) | measured | F004_radar_claude.png |
| figure | 5 | A | Multi-metric radar Brief vs dense — gpt synthetic | radar/spider (HELM) | measured | F005_radar_gpt.png |
| table | 14 | B | Depth-crossover compliance — synthetic | depth curve table | measured | T014_depth_crossover_synthetic.md |
| table | 15 | B | Depth-crossover compliance — dcbench | depth curve table | measured | T015_depth_crossover_dcbench.md |
| table | 16 | B | Depth-crossover compliance — swebench | depth curve table | measured | T016_depth_crossover_swebench.md |
| table | 17 | B | Depth-crossover compliance — GPT synthetic | cross-model depth | measured | T017_depth_crossover_gpt.md |
| table | 18 | B | Depth slope (d3-d1, less decay better) | slope table | measured | T018_depth_slope.md |
| figure | 6 | B | compliance vs depth — synthetic | depth-crossover curve | measured | F006_crossover_compliance_synthetic.png |
| figure | 7 | B | compliance vs depth — dcbench | depth-crossover curve | measured | F007_crossover_compliance_dcbench.png |
| figure | 8 | B | compliance vs depth — swebench | depth-crossover curve | measured | F008_crossover_compliance_swebench.png |
| figure | 9 | B | recall vs depth — synthetic | depth-crossover curve | measured | F009_crossover_recall_synthetic.png |
| figure | 10 | B | recall vs depth — dcbench | depth-crossover curve | measured | F010_crossover_recall_dcbench.png |
| figure | 11 | B | recall vs depth — swebench | depth-crossover curve | measured | F011_crossover_recall_swebench.png |
| figure | 12 | B | precision vs depth — synthetic | depth-crossover curve | measured | F012_crossover_precision_synthetic.png |
| figure | 13 | B | precision vs depth — dcbench | depth-crossover curve | measured | F013_crossover_precision_dcbench.png |
| figure | 14 | B | precision vs depth — swebench | depth-crossover curve | measured | F014_crossover_precision_swebench.png |
| figure | 15 | B | Compliance vs depth — GPT synthetic | cross-model crossover | measured | F015_crossover_gpt.png |
| figure | 16 | B | Depth-robustness slope bar | slope bar | measured | F016_depth_slope_bar.png |
| table | 19 | D | Product-Navigator lift (compliance): Brief vs none | intervention table | measured | T019_pn_lift_compliance.md |
| figure | 17 | D | Product-Navigator compliance | with/without bar | measured | F017_pn_compliance.png |
| table | 20 | D | Product-Navigator lift (recall): Brief vs none | intervention table | measured | T020_pn_lift_recall.md |
| figure | 18 | D | Product-Navigator recall | with/without bar | measured | F018_pn_recall.png |
| table | 21 | D | Product-Navigator lift (merge-ready): Brief vs none | intervention table | measured | T021_pn_lift_merge-ready.md |
| figure | 19 | D | Product-Navigator merge-ready | with/without bar | measured | F019_pn_merge-ready.png |
| table | 22 | G | Tokens per query by arm (all data) | efficiency table | measured | T022_tokens_per_arm.md |
| table | 23 | G | Return on Tokens by arm (compliance/1k tok) | RoT table | measured | T023_rot_per_arm.md |
| table | 24 | G | Cost-to-correct ($/merge-ready) | cost table | measured | T024_cost_to_correct.md |
| figure | 20 | G | Accuracy-vs-cost Pareto frontier | Pareto scatter | measured | F020_pareto.png |
| figure | 21 | G | Return on Tokens vs depth | RoT curve | measured | F021_rot_vs_depth.png |
| figure | 22 | G | Token distribution by arm | box plot | measured | F022_tokens_box.png |
| table | 25 | H | BCa 95% CI per arm — synthetic d=3 | CI table | measured | T025_bca_ci.md |
| table | 26 | H | Bayesian P(Brief>competitor) — synthetic d=3 | posterior table | measured | T026_bayes_superiority.md |
| table | 27 | H | Effect size — Brief vs each baseline (synthetic d=3) | effect-size table | measured | T027_effect_sizes.md |
| table | 28 | H | Friedman omnibus + mean ranks — synthetic d=3 | ranking table (Demšar) | measured | T028_friedman_ranks.md |
| table | 29 | H | Power & required-n — Brief vs baselines (synthetic d=3) | power table (Card 2020) | measured | T029_power.md |
| figure | 23 | H | Nemenyi critical-difference diagram | critical-difference (Demšar 2006) | measured | F023_critical_difference.png |
| figure | 24 | H | Forest plot of arm CIs | forest plot | measured | F024_forest.png |
| figure | 25 | H | Bayesian posterior densities | posterior density | measured | F025_posteriors.png |
| table | 30 | J | Claude vs GPT-5.1 — synthetic by metric | cross-model table | measured | T030_cross_model.md |
| figure | 26 | J | Cross-model agreement scatter | agreement scatter | measured | F026_cross_model_scatter.png |
| table | 31 | L | Failure-mode taxonomy (Claude all data) | failure-mode table | measured | T031_failure_taxonomy.md |
| figure | 27 | L | Failure composition (stacked) | stacked bar | measured | F027_failure_stacked.png |
| table | 32 | O | Pairwise advantage matrix (row-col compliance, all data) | win matrix | measured | T032_win_matrix.md |
| figure | 28 | O | Pairwise win-rate heatmap | win-rate heatmap | measured | F028_winrate_heatmap.png |
| figure | 29 | O | Metric correlation heatmap | correlation heatmap | measured | F029_metric_corr.png |
| figure | 30 | O | Compliance ECDF | ECDF | measured | F030_ecdf.png |
