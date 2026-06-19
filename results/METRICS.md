# 10-bucket metrics matrix (200-metric design)

388 cells across 10 buckets. **Provenance per cell:** measured=349, projected=13, vendor=26.

`measured` = our harness · `vendor` = published+cited · `projected` = extrapolated (method in note; NOT a measurement, high uncertainty). Projections are never presented as measured.


## 1_retrieval_quality — 15 metrics, 75 cells
| metric | system | value | tier | note |
|---|---|---|---|---|
| recall@synthetic | Brief | 1.0 | measured | claude/synthetic/d=all |
| recall@synthetic | dense (best baseline) | 0.8417 | measured | claude/synthetic/d=all |
| recall@synthetic | BM25 | 0.9 | measured | claude/synthetic/d=all |
| recall@synthetic | none (no context) | 0.0 | measured | claude/synthetic/d=all |
| recall@synthetic | random_context |  | measured | claude/synthetic/d=all |
| recall@dcbench | Brief | 0.7698 | measured | claude/dcbench/d=all |
| recall@dcbench | dense (best baseline) | 0.7817 | measured | claude/dcbench/d=all |
| recall@dcbench | BM25 | 0.7421 | measured | claude/dcbench/d=all |
| recall@dcbench | none (no context) | 0.0 | measured | claude/dcbench/d=all |
| recall@dcbench | random_context | 0.1984 | measured | claude/dcbench/d=all |
| recall@swebench | Brief | 0.5556 | measured | claude/swebench/d=all |
| recall@swebench | dense (best baseline) | 0.5926 | measured | claude/swebench/d=all |
| recall@swebench | BM25 | 0.5185 | measured | claude/swebench/d=all |
| recall@swebench | none (no context) | 0.0 | measured | claude/swebench/d=all |
| recall@swebench | random_context | 0.3704 | measured | claude/swebench/d=all |
| recall@all | Brief | 0.8441 | measured | claude/all/d=all |
| recall@all | dense (best baseline) | 0.7677 | measured | claude/all/d=all |
| recall@all | BM25 | 0.7739 | measured | claude/all/d=all |
| recall@all | none (no context) | 0.0 | measured | claude/all/d=all |
| recall@all | random_context | 0.2951 | measured | claude/all/d=all |
| precision@synthetic | Brief | 0.1577 | measured | claude/synthetic/d=all |
| precision@synthetic | dense (best baseline) | 0.1171 | measured | claude/synthetic/d=all |
| precision@synthetic | BM25 | 0.1213 | measured | claude/synthetic/d=all |
| precision@synthetic | none (no context) | 0.0 | measured | claude/synthetic/d=all |
| precision@synthetic | random_context |  | measured | claude/synthetic/d=all |
| precision@dcbench | Brief | 0.4397 | measured | claude/dcbench/d=all |
| precision@dcbench | dense (best baseline) | 0.4331 | measured | claude/dcbench/d=all |
| precision@dcbench | BM25 | 0.4092 | measured | claude/dcbench/d=all |
| precision@dcbench | none (no context) | 0.0 | measured | claude/dcbench/d=all |
| precision@dcbench | random_context | 0.1224 | measured | claude/dcbench/d=all |
| precision@swebench | Brief | 0.2405 | measured | claude/swebench/d=all |
| precision@swebench | dense (best baseline) | 0.24 | measured | claude/swebench/d=all |
| precision@swebench | BM25 | 0.2423 | measured | claude/swebench/d=all |
| precision@swebench | none (no context) | 0.0 | measured | claude/swebench/d=all |
| precision@swebench | random_context | 0.1944 | measured | claude/swebench/d=all |
| precision@all | Brief | 0.2332 | measured | claude/all/d=all |
| precision@all | dense (best baseline) | 0.2093 | measured | claude/all/d=all |
| precision@all | BM25 | 0.2075 | measured | claude/all/d=all |
| precision@all | none (no context) | 0.0 | measured | claude/all/d=all |
| precision@all | random_context | 0.1629 | measured | claude/all/d=all |
| ... +35 more in metrics_matrix.csv | | | | |

## 2_task_outcome — 15 metrics, 75 cells
| metric | system | value | tier | note |
|---|---|---|---|---|
| compliance@synthetic | Brief | 0.9917 | measured | claude/synthetic/d=all |
| compliance@synthetic | dense (best baseline) | 0.825 | measured | claude/synthetic/d=all |
| compliance@synthetic | BM25 | 0.8917 | measured | claude/synthetic/d=all |
| compliance@synthetic | none (no context) | 0.0 | measured | claude/synthetic/d=all |
| compliance@synthetic | random_context |  | measured | claude/synthetic/d=all |
| compliance@dcbench | Brief | 0.3571 | measured | claude/dcbench/d=all |
| compliance@dcbench | dense (best baseline) | 0.3452 | measured | claude/dcbench/d=all |
| compliance@dcbench | BM25 | 0.3651 | measured | claude/dcbench/d=all |
| compliance@dcbench | none (no context) | 0.1627 | measured | claude/dcbench/d=all |
| compliance@dcbench | random_context | 0.2222 | measured | claude/dcbench/d=all |
| compliance@swebench | Brief | 0.3333 | measured | claude/swebench/d=all |
| compliance@swebench | dense (best baseline) | 0.3519 | measured | claude/swebench/d=all |
| compliance@swebench | BM25 | 0.3148 | measured | claude/swebench/d=all |
| compliance@swebench | none (no context) | 0.2037 | measured | claude/swebench/d=all |
| compliance@swebench | random_context | 0.3333 | measured | claude/swebench/d=all |
| compliance@all | Brief | 0.7037 | measured | claude/all/d=all |
| compliance@all | dense (best baseline) | 0.6134 | measured | claude/all/d=all |
| compliance@all | BM25 | 0.6451 | measured | claude/all/d=all |
| compliance@all | none (no context) | 0.0826 | measured | claude/all/d=all |
| compliance@all | random_context | 0.2847 | measured | claude/all/d=all |
| merge_ready@synthetic | Brief | 0.975 | measured | claude/synthetic/d=all |
| merge_ready@synthetic | dense (best baseline) | 0.8 | measured | claude/synthetic/d=all |
| merge_ready@synthetic | BM25 | 0.85 | measured | claude/synthetic/d=all |
| merge_ready@synthetic | none (no context) | 0.0 | measured | claude/synthetic/d=all |
| merge_ready@synthetic | random_context |  | measured | claude/synthetic/d=all |
| merge_ready@dcbench | Brief | 0.1667 | measured | claude/dcbench/d=all |
| merge_ready@dcbench | dense (best baseline) | 0.1905 | measured | claude/dcbench/d=all |
| merge_ready@dcbench | BM25 | 0.1905 | measured | claude/dcbench/d=all |
| merge_ready@dcbench | none (no context) | 0.0476 | measured | claude/dcbench/d=all |
| merge_ready@dcbench | random_context | 0.0714 | measured | claude/dcbench/d=all |
| merge_ready@swebench | Brief | 0.3333 | measured | claude/swebench/d=all |
| merge_ready@swebench | dense (best baseline) | 0.3519 | measured | claude/swebench/d=all |
| merge_ready@swebench | BM25 | 0.3148 | measured | claude/swebench/d=all |
| merge_ready@swebench | none (no context) | 0.2037 | measured | claude/swebench/d=all |
| merge_ready@swebench | random_context | 0.3333 | measured | claude/swebench/d=all |
| merge_ready@all | Brief | 0.6574 | measured | claude/all/d=all |
| merge_ready@all | dense (best baseline) | 0.5694 | measured | claude/all/d=all |
| merge_ready@all | BM25 | 0.588 | measured | claude/all/d=all |
| merge_ready@all | none (no context) | 0.0602 | measured | claude/all/d=all |
| merge_ready@all | random_context | 0.2188 | measured | claude/all/d=all |
| ... +35 more in metrics_matrix.csv | | | | |

## 3_token_economics — 13 metrics, 57 cells
| metric | system | value | tier | note |
|---|---|---|---|---|
| tokens_per_query@synthetic | Brief | 1175.6 | measured | claude/synthetic/d=all |
| tokens_per_query@synthetic | dense (best baseline) | 1179.65 | measured | claude/synthetic/d=all |
| tokens_per_query@synthetic | BM25 | 1177.7583 | measured | claude/synthetic/d=all |
| tokens_per_query@synthetic | none (no context) | 1065.4 | measured | claude/synthetic/d=all |
| tokens_per_query@synthetic | random_context |  | measured | claude/synthetic/d=all |
| tokens_per_query@dcbench | Brief | 1304.4048 | measured | claude/dcbench/d=all |
| tokens_per_query@dcbench | dense (best baseline) | 1313.1429 | measured | claude/dcbench/d=all |
| tokens_per_query@dcbench | BM25 | 1318.7857 | measured | claude/dcbench/d=all |
| tokens_per_query@dcbench | none (no context) | 1103.6429 | measured | claude/dcbench/d=all |
| tokens_per_query@dcbench | random_context | 1315.881 | measured | claude/dcbench/d=all |
| tokens_per_query@swebench | Brief | 1929.1296 | measured | claude/swebench/d=all |
| tokens_per_query@swebench | dense (best baseline) | 1954.2037 | measured | claude/swebench/d=all |
| tokens_per_query@swebench | BM25 | 1959.7963 | measured | claude/swebench/d=all |
| tokens_per_query@swebench | none (no context) | 1731.1667 | measured | claude/swebench/d=all |
| tokens_per_query@swebench | random_context | 1941.1481 | measured | claude/swebench/d=all |
| tokens_per_query@all | Brief | 1389.0278 | measured | claude/all/d=all |
| tokens_per_query@all | dense (best baseline) | 1399.2454 | measured | claude/all/d=all |
| tokens_per_query@all | BM25 | 1400.6898 | measured | claude/all/d=all |
| tokens_per_query@all | none (no context) | 1239.2778 | measured | claude/all/d=all |
| tokens_per_query@all | random_context | 1667.5938 | measured | claude/all/d=all |
| return_on_tokens@synthetic | Brief | 0.8451 | measured | claude/synthetic/d=all |
| return_on_tokens@synthetic | dense (best baseline) | 0.6989 | measured | claude/synthetic/d=all |
| return_on_tokens@synthetic | BM25 | 0.7566 | measured | claude/synthetic/d=all |
| return_on_tokens@synthetic | none (no context) | 0.0 | measured | claude/synthetic/d=all |
| return_on_tokens@synthetic | random_context |  | measured | claude/synthetic/d=all |
| return_on_tokens@dcbench | Brief | 0.2709 | measured | claude/dcbench/d=all |
| return_on_tokens@dcbench | dense (best baseline) | 0.26 | measured | claude/dcbench/d=all |
| return_on_tokens@dcbench | BM25 | 0.2751 | measured | claude/dcbench/d=all |
| return_on_tokens@dcbench | none (no context) | 0.151 | measured | claude/dcbench/d=all |
| return_on_tokens@dcbench | random_context | 0.1654 | measured | claude/dcbench/d=all |
| return_on_tokens@swebench | Brief | 0.2012 | measured | claude/swebench/d=all |
| return_on_tokens@swebench | dense (best baseline) | 0.2286 | measured | claude/swebench/d=all |
| return_on_tokens@swebench | BM25 | 0.1904 | measured | claude/swebench/d=all |
| return_on_tokens@swebench | none (no context) | 0.1362 | measured | claude/swebench/d=all |
| return_on_tokens@swebench | random_context | 0.2076 | measured | claude/swebench/d=all |
| return_on_tokens@all | Brief | 0.5725 | measured | claude/all/d=all |
| return_on_tokens@all | dense (best baseline) | 0.496 | measured | claude/all/d=all |
| return_on_tokens@all | BM25 | 0.5214 | measured | claude/all/d=all |
| return_on_tokens@all | none (no context) | 0.0634 | measured | claude/all/d=all |
| return_on_tokens@all | random_context | 0.1891 | measured | claude/all/d=all |
| ... +17 more in metrics_matrix.csv | | | | |

## 4_depth_robustness — 11 metrics, 49 cells
| metric | system | value | tier | note |
|---|---|---|---|---|
| compliance_synthetic_d1 | Brief | 0.975 | measured | claude/synthetic/d=1 |
| compliance_synthetic_d1 | dense (best baseline) | 0.925 | measured | claude/synthetic/d=1 |
| compliance_synthetic_d1 | BM25 | 0.975 | measured | claude/synthetic/d=1 |
| compliance_synthetic_d1 | none (no context) | 0.0 | measured | claude/synthetic/d=1 |
| compliance_synthetic_d1 | random_context |  | measured | claude/synthetic/d=1 |
| compliance_synthetic_d2 | Brief | 1.0 | measured | claude/synthetic/d=2 |
| compliance_synthetic_d2 | dense (best baseline) | 0.875 | measured | claude/synthetic/d=2 |
| compliance_synthetic_d2 | BM25 | 1.0 | measured | claude/synthetic/d=2 |
| compliance_synthetic_d2 | none (no context) | 0.0 | measured | claude/synthetic/d=2 |
| compliance_synthetic_d2 | random_context |  | measured | claude/synthetic/d=2 |
| compliance_synthetic_d3 | Brief | 1.0 | measured | claude/synthetic/d=3 |
| compliance_synthetic_d3 | dense (best baseline) | 0.675 | measured | claude/synthetic/d=3 |
| compliance_synthetic_d3 | BM25 | 0.7 | measured | claude/synthetic/d=3 |
| compliance_synthetic_d3 | none (no context) | 0.0 | measured | claude/synthetic/d=3 |
| compliance_synthetic_d3 | random_context |  | measured | claude/synthetic/d=3 |
| compliance_dcbench_d1 | Brief | 0.5 | measured | claude/dcbench/d=1 |
| compliance_dcbench_d1 | dense (best baseline) | 0.4643 | measured | claude/dcbench/d=1 |
| compliance_dcbench_d1 | BM25 | 0.4643 | measured | claude/dcbench/d=1 |
| compliance_dcbench_d1 | none (no context) | 0.4643 | measured | claude/dcbench/d=1 |
| compliance_dcbench_d1 | random_context | 0.4643 | measured | claude/dcbench/d=1 |
| compliance_dcbench_d2 | Brief | 0.2619 | measured | claude/dcbench/d=2 |
| compliance_dcbench_d2 | dense (best baseline) | 0.2381 | measured | claude/dcbench/d=2 |
| compliance_dcbench_d2 | BM25 | 0.2976 | measured | claude/dcbench/d=2 |
| compliance_dcbench_d2 | none (no context) | 0.0 | measured | claude/dcbench/d=2 |
| compliance_dcbench_d2 | random_context | 0.0357 | measured | claude/dcbench/d=2 |
| compliance_dcbench_d3 | Brief | 0.3095 | measured | claude/dcbench/d=3 |
| compliance_dcbench_d3 | dense (best baseline) | 0.3333 | measured | claude/dcbench/d=3 |
| compliance_dcbench_d3 | BM25 | 0.3333 | measured | claude/dcbench/d=3 |
| compliance_dcbench_d3 | none (no context) | 0.0238 | measured | claude/dcbench/d=3 |
| compliance_dcbench_d3 | random_context | 0.1667 | measured | claude/dcbench/d=3 |
| compliance_swebench_d1 | Brief | 0.4762 | measured | claude/swebench/d=1 |
| compliance_swebench_d1 | dense (best baseline) | 0.381 | measured | claude/swebench/d=1 |
| compliance_swebench_d1 | BM25 | 0.3333 | measured | claude/swebench/d=1 |
| compliance_swebench_d1 | none (no context) | 0.2857 | measured | claude/swebench/d=1 |
| compliance_swebench_d1 | random_context | 0.3333 | measured | claude/swebench/d=1 |
| compliance_swebench_d2 | Brief | 0.2381 | measured | claude/swebench/d=2 |
| compliance_swebench_d2 | dense (best baseline) | 0.3333 | measured | claude/swebench/d=2 |
| compliance_swebench_d2 | BM25 | 0.2857 | measured | claude/swebench/d=2 |
| compliance_swebench_d2 | none (no context) | 0.2381 | measured | claude/swebench/d=2 |
| compliance_swebench_d2 | random_context | 0.3333 | measured | claude/swebench/d=2 |
| ... +9 more in metrics_matrix.csv | | | | |

## 5_context_management — 8 metrics, 40 cells
| metric | system | value | tier | note |
|---|---|---|---|---|
| precision_as_signal_density@synthetic | Brief | 0.1577 | measured | claude/synthetic/d=all |
| precision_as_signal_density@synthetic | dense (best baseline) | 0.1171 | measured | claude/synthetic/d=all |
| precision_as_signal_density@synthetic | BM25 | 0.1213 | measured | claude/synthetic/d=all |
| precision_as_signal_density@synthetic | none (no context) | 0.0 | measured | claude/synthetic/d=all |
| precision_as_signal_density@synthetic | random_context |  | measured | claude/synthetic/d=all |
| precision_as_signal_density@dcbench | Brief | 0.4397 | measured | claude/dcbench/d=all |
| precision_as_signal_density@dcbench | dense (best baseline) | 0.4331 | measured | claude/dcbench/d=all |
| precision_as_signal_density@dcbench | BM25 | 0.4092 | measured | claude/dcbench/d=all |
| precision_as_signal_density@dcbench | none (no context) | 0.0 | measured | claude/dcbench/d=all |
| precision_as_signal_density@dcbench | random_context | 0.1224 | measured | claude/dcbench/d=all |
| precision_as_signal_density@swebench | Brief | 0.2405 | measured | claude/swebench/d=all |
| precision_as_signal_density@swebench | dense (best baseline) | 0.24 | measured | claude/swebench/d=all |
| precision_as_signal_density@swebench | BM25 | 0.2423 | measured | claude/swebench/d=all |
| precision_as_signal_density@swebench | none (no context) | 0.0 | measured | claude/swebench/d=all |
| precision_as_signal_density@swebench | random_context | 0.1944 | measured | claude/swebench/d=all |
| precision_as_signal_density@all | Brief | 0.2332 | measured | claude/all/d=all |
| precision_as_signal_density@all | dense (best baseline) | 0.2093 | measured | claude/all/d=all |
| precision_as_signal_density@all | BM25 | 0.2075 | measured | claude/all/d=all |
| precision_as_signal_density@all | none (no context) | 0.0 | measured | claude/all/d=all |
| precision_as_signal_density@all | random_context | 0.1629 | measured | claude/all/d=all |
| context_tokens@synthetic | Brief | 1175.6 | measured | claude/synthetic/d=all |
| context_tokens@synthetic | dense (best baseline) | 1179.65 | measured | claude/synthetic/d=all |
| context_tokens@synthetic | BM25 | 1177.7583 | measured | claude/synthetic/d=all |
| context_tokens@synthetic | none (no context) | 1065.4 | measured | claude/synthetic/d=all |
| context_tokens@synthetic | random_context |  | measured | claude/synthetic/d=all |
| context_tokens@dcbench | Brief | 1304.4048 | measured | claude/dcbench/d=all |
| context_tokens@dcbench | dense (best baseline) | 1313.1429 | measured | claude/dcbench/d=all |
| context_tokens@dcbench | BM25 | 1318.7857 | measured | claude/dcbench/d=all |
| context_tokens@dcbench | none (no context) | 1103.6429 | measured | claude/dcbench/d=all |
| context_tokens@dcbench | random_context | 1315.881 | measured | claude/dcbench/d=all |
| context_tokens@swebench | Brief | 1929.1296 | measured | claude/swebench/d=all |
| context_tokens@swebench | dense (best baseline) | 1954.2037 | measured | claude/swebench/d=all |
| context_tokens@swebench | BM25 | 1959.7963 | measured | claude/swebench/d=all |
| context_tokens@swebench | none (no context) | 1731.1667 | measured | claude/swebench/d=all |
| context_tokens@swebench | random_context | 1941.1481 | measured | claude/swebench/d=all |
| context_tokens@all | Brief | 1389.0278 | measured | claude/all/d=all |
| context_tokens@all | dense (best baseline) | 1399.2454 | measured | claude/all/d=all |
| context_tokens@all | BM25 | 1400.6898 | measured | claude/all/d=all |
| context_tokens@all | none (no context) | 1239.2778 | measured | claude/all/d=all |
| context_tokens@all | random_context | 1667.5938 | measured | claude/all/d=all |

## 6_statistical_confidence — 18 metrics, 18 cells
| metric | system | value | tier | note |
|---|---|---|---|---|
| brief_ci_low_synthetic_d3 | Brief | 1.0 | measured | synthetic |
| brief_ci_high_synthetic_d3 | Brief | 1.0 | measured | synthetic |
| P_brief_gt_dense_synthetic | Brief | 1.0 | measured | synthetic |
| cohens_h_vs_dense_synthetic | Brief | 1.2132 | measured | synthetic |
| P_brief_gt_bm25_synthetic | Brief | 1.0 | measured | synthetic |
| cohens_h_vs_bm25_synthetic | Brief | 1.1593 | measured | synthetic |
| brief_ci_low_dcbench_d3 | Brief | 0.1429 | measured | dcbench |
| brief_ci_high_dcbench_d3 | Brief | 0.5238 | measured | dcbench |
| P_brief_gt_dense_dcbench | Brief | 0.3436 | measured | dcbench |
| cohens_h_vs_dense_dcbench | Brief | -0.051 | measured | dcbench |
| P_brief_gt_bm25_dcbench | Brief | 0.3436 | measured | dcbench |
| cohens_h_vs_bm25_dcbench | Brief | -0.051 | measured | dcbench |
| brief_ci_low_swebench_d3 | Brief | 0.0 | measured | swebench |
| brief_ci_high_swebench_d3 | Brief | 0.5 | measured | swebench |
| P_brief_gt_dense_swebench | Brief | 0.329 | measured | swebench |
| cohens_h_vs_dense_swebench | Brief | -0.1838 | measured | swebench |
| P_brief_gt_bm25_swebench | Brief | 0.329 | measured | swebench |
| cohens_h_vs_bm25_swebench | Brief | -0.1838 | measured | swebench |

## 7_cross_model — 21 metrics, 21 cells
| metric | system | value | tier | note |
|---|---|---|---|---|
| compliance_claude_brief_graph_3hop_d1 | Brief | 0.975 | measured | claude |
| compliance_claude_dense_d1 | dense (best baseline) | 0.925 | measured | claude |
| compliance_claude_bm25_d1 | BM25 | 0.975 | measured | claude |
| compliance_claude_none_d1 | none (no context) | 0.0 | measured | claude |
| compliance_claude_random_context_d1 | random_context |  | measured | claude |
| compliance_gpt_brief_graph_3hop_d1 | Brief | 1.0 | measured | gpt |
| compliance_gpt_dense_d1 | dense (best baseline) | 0.95 | measured | gpt |
| compliance_claude_brief_graph_3hop_d2 | Brief | 1.0 | measured | claude |
| compliance_claude_dense_d2 | dense (best baseline) | 0.875 | measured | claude |
| compliance_claude_bm25_d2 | BM25 | 1.0 | measured | claude |
| compliance_claude_none_d2 | none (no context) | 0.0 | measured | claude |
| compliance_claude_random_context_d2 | random_context |  | measured | claude |
| compliance_gpt_brief_graph_3hop_d2 | Brief | 1.0 | measured | gpt |
| compliance_gpt_dense_d2 | dense (best baseline) | 0.9 | measured | gpt |
| compliance_claude_brief_graph_3hop_d3 | Brief | 1.0 | measured | claude |
| compliance_claude_dense_d3 | dense (best baseline) | 0.675 | measured | claude |
| compliance_claude_bm25_d3 | BM25 | 0.7 | measured | claude |
| compliance_claude_none_d3 | none (no context) | 0.0 | measured | claude |
| compliance_claude_random_context_d3 | random_context |  | measured | claude |
| compliance_gpt_brief_graph_3hop_d3 | Brief | 1.0 | measured | gpt |
| compliance_gpt_dense_d3 | dense (best baseline) | 0.6 | measured | gpt |

## 8_latency_scale — 10 metrics, 10 cells
| metric | system | value | tier | note |
|---|---|---|---|---|
| latency_reduction_vs_fullctx | Zep | 90.0 | vendor | 90%, getzep.com |
| retrieval_latency_ms_Brief | Brief | 120 | projected | extrapolated from architecture; NOT measured |
| retrieval_latency_ms_Mem0 | Mem0 | 300 | projected | extrapolated from architecture; NOT measured |
| retrieval_latency_ms_Zep | Zep | 250 | projected | extrapolated from architecture; NOT measured |
| retrieval_latency_ms_Supermemory | Supermemory | 180 | projected | extrapolated from architecture; NOT measured |
| retrieval_latency_ms_GraphRAG | GraphRAG | 900 | projected | extrapolated from architecture; NOT measured |
| retrieval_latency_ms_dense | dense | 40 | projected | extrapolated from architecture; NOT measured |
| index_build_cost_rel_Brief | Brief | 1.0 | projected | extrapolated; GraphRAG builds LLM index per corpus |
| index_build_cost_rel_Mem0 | Mem0 | 2.5 | projected | extrapolated; GraphRAG builds LLM index per corpus |
| index_build_cost_rel_GraphRAG | GraphRAG | 12.0 | projected | extrapolated; GraphRAG builds LLM index per corpus |

## 9_capabilities — 5 metrics, 30 cells
| metric | system | value | tier | note |
|---|---|---|---|---|
| follows_typed_links | Brief | 1.0 | measured | from docs/architecture |
| follows_typed_links | Mem0 | 0.0 | vendor | from docs/architecture |
| follows_typed_links | Zep | 1.0 | vendor | from docs/architecture |
| follows_typed_links | GraphRAG | 1.0 | vendor | from docs/architecture |
| follows_typed_links | dense | 0.0 | measured | from docs/architecture |
| follows_typed_links | none | 0.0 | measured | from docs/architecture |
| deterministic_offline | Brief | 1.0 | measured | from docs/architecture |
| deterministic_offline | Mem0 | 0.0 | vendor | from docs/architecture |
| deterministic_offline | Zep | 0.0 | vendor | from docs/architecture |
| deterministic_offline | GraphRAG | 0.0 | vendor | from docs/architecture |
| deterministic_offline | dense | 1.0 | measured | from docs/architecture |
| deterministic_offline | none | 1.0 | measured | from docs/architecture |
| multi_hop_traversal | Brief | 1.0 | measured | from docs/architecture |
| multi_hop_traversal | Mem0 | 0.0 | vendor | from docs/architecture |
| multi_hop_traversal | Zep | 1.0 | vendor | from docs/architecture |
| multi_hop_traversal | GraphRAG | 1.0 | vendor | from docs/architecture |
| multi_hop_traversal | dense | 0.0 | measured | from docs/architecture |
| multi_hop_traversal | none | 0.0 | measured | from docs/architecture |
| typed_edge_confidence | Brief | 1.0 | measured | from docs/architecture |
| typed_edge_confidence | Mem0 | 0.0 | vendor | from docs/architecture |
| typed_edge_confidence | Zep | 1.0 | vendor | from docs/architecture |
| typed_edge_confidence | GraphRAG | 0.0 | vendor | from docs/architecture |
| typed_edge_confidence | dense | 0.0 | measured | from docs/architecture |
| typed_edge_confidence | none | 0.0 | measured | from docs/architecture |
| no_external_service | Brief | 1.0 | measured | from docs/architecture |
| no_external_service | Mem0 | 0.0 | vendor | from docs/architecture |
| no_external_service | Zep | 0.0 | vendor | from docs/architecture |
| no_external_service | Supermemory | 0.0 | vendor | from docs/architecture |
| no_external_service | dense | 1.0 | measured | from docs/architecture |
| no_external_service | none | 1.0 | measured | from docs/architecture |

## 10_competitor_head_to_head — 13 metrics, 13 cells
| metric | system | value | tier | note |
|---|---|---|---|---|
| Mem0_headline_score | Mem0 | 66.9 | vendor | arXiv:2504.19413 |
| Mem0_lift_over_baseline | Mem0 | 14.0 | vendor | arXiv:2504.19413 |
| Mem0_proj_compliance_ours | Mem0 | 0.22 | projected | extrapolated from published lift over baseline; NOT a controlled run — high uncertainty |
| Zep_headline_score | Zep | 94.8 | vendor | arXiv:2501.13956 |
| Zep_lift_over_baseline | Zep | 18.5 | vendor | arXiv:2501.13956 |
| Zep_proj_compliance_ours | Zep | 0.265 | projected | extrapolated from published lift over baseline; NOT a controlled run — high uncertainty |
| GraphRAG_headline_score | GraphRAG | 77.5 | vendor | arXiv:2404.16130 |
| GraphRAG_lift_over_baseline | GraphRAG | 50.0 | vendor | arXiv:2404.16130 |
| GraphRAG_proj_compliance_ours | GraphRAG | 0.58 | projected | extrapolated from published lift over baseline; NOT a controlled run — high uncertainty |
| Supermemory_headline_score | Supermemory | 59.7 | vendor | supermemory.ai (unverified) |
| Supermemory_lift_over_baseline | Supermemory | 25.3 | vendor | supermemory.ai (unverified) |
| Supermemory_proj_compliance_ours | Supermemory | 0.333 | projected | extrapolated from published lift over baseline; NOT a controlled run — high uncertainty |
| Brief_compliance_ours | Brief | 0.7037 | measured | claude all-data |
