<!-- Noise-retention score: recall(K=40)/recall(K=0) | robustness ratio | offline-recall -->

**Table 49. Noise-retention score: recall(K=40)/recall(K=0)**

| arm | recall@K0 | recall@K40 | retention |
|---|---|---|---|
| brief_graph_3hop | 1.00 | 1.00 | 1.00 |
| dense | 0.68 | 0.68 | 1.00 |
| bm25 | 0.70 | 0.70 | 1.00 |
| tfidf | 0.70 | 0.70 | 1.00 |
| hybrid_rrf | 0.60 | 0.60 | 1.00 |
| rerank_ce | 0.38 | 0.42 | 1.13 |
| random_context | 0.25 | 0.15 | 0.60 |
