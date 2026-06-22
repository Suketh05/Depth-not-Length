<!-- HotpotQA multi-hop retrieval (n=80 2-hop questions) | HotpotQA (established multi-hop) | measured-offline -->

**Table 101. HotpotQA multi-hop retrieval (n=80 2-hop questions)**

| arm | support-fact recall (both@top3) | recall@5 | nDCG@10 | MRR |
|---|---|---|---|---|
| brief_graph_3hop | 0.34 | 0.74 | 0.611 | 0.604 |
| dense | 0.15 | 0.59 | 0.643 | 0.589 |
| bm25 | 0.42 | 0.76 | 0.817 | 0.849 |
| tfidf | 0.39 | 0.76 | 0.790 | 0.806 |
| hybrid_rrf | 0.26 | 0.72 | 0.741 | 0.737 |
