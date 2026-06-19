# results — comprehensive eval matrix

**440 evals** across 9 families, on Claude Sonnet + GPT-5.1. Spine: agent+Brief vs agent-alone (Product Navigator). Every value computed from `data/all_rows_*.jsonl`; cuts where Brief does not win are included (honest).

**Brief wins 312/440 head-to-head cuts.**


## F1_product_navigator — 120 evals (Brief wins 116)

| id | model | dataset | depth | metric | comparison | brief | base | Δ | n |
|---|---|---|---|---|---|---|---|---|---|
| 1 | claude | synthetic | 1 | compliance | brief_vs_none | 0.975 | 0.0 | 0.975 | 40 |
| 2 | claude | synthetic | 1 | recall | brief_vs_none | 1.0 | 0.0 | 1.0 | 40 |
| 3 | claude | synthetic | 1 | precision | brief_vs_none | 0.1518 | 0.0 | 0.1518 | 40 |
| 4 | claude | synthetic | 1 | merge_ready | brief_vs_none | 0.975 | 0.0 | 0.975 | 40 |
| 5 | claude | synthetic | 1 | chain_recovery | brief_vs_none | 1.0 | 0.0 | 1.0 | 40 |
| 6 | claude | synthetic | 1 | return_on_tokens | brief_vs_none | 0.8326 | 0.0 | 0.8326 | 40 |
| 7 | claude | synthetic | 2 | compliance | brief_vs_none | 1.0 | 0.0 | 1.0 | 40 |
| 8 | claude | synthetic | 2 | recall | brief_vs_none | 1.0 | 0.0 | 1.0 | 40 |
| 9 | claude | synthetic | 2 | precision | brief_vs_none | 0.1607 | 0.0 | 0.1607 | 40 |
| 10 | claude | synthetic | 2 | merge_ready | brief_vs_none | 0.975 | 0.0 | 0.975 | 40 |
| 11 | claude | synthetic | 2 | chain_recovery | brief_vs_none | 1.0 | 0.0 | 1.0 | 40 |
| 12 | claude | synthetic | 2 | return_on_tokens | brief_vs_none | 0.847 | 0.0 | 0.847 | 40 |
| 13 | claude | synthetic | 3 | compliance | brief_vs_none | 1.0 | 0.0 | 1.0 | 40 |
| 14 | claude | synthetic | 3 | recall | brief_vs_none | 1.0 | 0.0 | 1.0 | 40 |
| 15 | claude | synthetic | 3 | precision | brief_vs_none | 0.1605 | 0.0 | 0.1605 | 40 |
| 16 | claude | synthetic | 3 | merge_ready | brief_vs_none | 0.975 | 0.0 | 0.975 | 40 |
| 17 | claude | synthetic | 3 | chain_recovery | brief_vs_none | 1.0 | 0.0 | 1.0 | 40 |
| 18 | claude | synthetic | 3 | return_on_tokens | brief_vs_none | 0.8557 | 0.0 | 0.8557 | 40 |
| 19 | claude | synthetic | ALL | compliance | brief_vs_none | 0.9917 | 0.0 | 0.9917 | 120 |
| 20 | claude | synthetic | ALL | recall | brief_vs_none | 1.0 | 0.0 | 1.0 | 120 |
| 21 | claude | synthetic | ALL | precision | brief_vs_none | 0.1577 | 0.0 | 0.1577 | 120 |
| 22 | claude | synthetic | ALL | merge_ready | brief_vs_none | 0.975 | 0.0 | 0.975 | 120 |
| 23 | claude | synthetic | ALL | chain_recovery | brief_vs_none | 1.0 | 0.0 | 1.0 | 120 |
| 24 | claude | synthetic | ALL | return_on_tokens | brief_vs_none | 0.8451 | 0.0 | 0.8451 | 120 |
| 25 | claude | dcbench | 1 | compliance | brief_vs_none | 0.5 | 0.4643 | 0.0357 | 14 |
| 26 | claude | dcbench | 1 | recall | brief_vs_none | 1.0 | 0.0 | 1.0 | 14 |
| 27 | claude | dcbench | 1 | precision | brief_vs_none | 0.6786 | 0.0 | 0.6786 | 14 |
| 28 | claude | dcbench | 1 | merge_ready | brief_vs_none | 0.2143 | 0.1429 | 0.0714 | 14 |
| 29 | claude | dcbench | 1 | chain_recovery | brief_vs_none | 1.0 | 0.0 | 1.0 | 14 |
| 30 | claude | dcbench | 1 | return_on_tokens | brief_vs_none | 0.3653 | 0.4312 | -0.0659 | 14 |
| 31 | claude | dcbench | 2 | compliance | brief_vs_none | 0.2619 | 0.0 | 0.2619 | 14 |
| 32 | claude | dcbench | 2 | recall | brief_vs_none | 0.6429 | 0.0 | 0.6429 | 14 |
| 33 | claude | dcbench | 2 | precision | brief_vs_none | 0.4167 | 0.0 | 0.4167 | 14 |
| 34 | claude | dcbench | 2 | merge_ready | brief_vs_none | 0.1429 | 0.0 | 0.1429 | 14 |
| 35 | claude | dcbench | 2 | chain_recovery | brief_vs_none | 0.2857 | 0.0 | 0.2857 | 14 |
| 36 | claude | dcbench | 2 | return_on_tokens | brief_vs_none | 0.2026 | 0.0 | 0.2026 | 14 |
| 37 | claude | dcbench | 3 | compliance | brief_vs_none | 0.3095 | 0.0238 | 0.2857 | 14 |
| 38 | claude | dcbench | 3 | recall | brief_vs_none | 0.6667 | 0.0 | 0.6667 | 14 |
| 39 | claude | dcbench | 3 | precision | brief_vs_none | 0.2238 | 0.0 | 0.2238 | 14 |
| 40 | claude | dcbench | 3 | merge_ready | brief_vs_none | 0.1429 | 0.0 | 0.1429 | 14 |
| 41 | claude | dcbench | 3 | chain_recovery | brief_vs_none | 0.3571 | 0.0 | 0.3571 | 14 |
| 42 | claude | dcbench | 3 | return_on_tokens | brief_vs_none | 0.2449 | 0.0217 | 0.2232 | 14 |
| 43 | claude | dcbench | ALL | compliance | brief_vs_none | 0.3571 | 0.1627 | 0.1944 | 42 |
| 44 | claude | dcbench | ALL | recall | brief_vs_none | 0.7698 | 0.0 | 0.7698 | 42 |
| 45 | claude | dcbench | ALL | precision | brief_vs_none | 0.4397 | 0.0 | 0.4397 | 42 |
| 46 | claude | dcbench | ALL | merge_ready | brief_vs_none | 0.1667 | 0.0476 | 0.119 | 42 |
| 47 | claude | dcbench | ALL | chain_recovery | brief_vs_none | 0.5476 | 0.0 | 0.5476 | 42 |
| 48 | claude | dcbench | ALL | return_on_tokens | brief_vs_none | 0.2709 | 0.151 | 0.1199 | 42 |
| 49 | claude | swebench | 1 | compliance | brief_vs_none | 0.4762 | 0.2857 | 0.1905 | 21 |
| 50 | claude | swebench | 1 | recall | brief_vs_none | 0.7619 | 0.0 | 0.7619 | 21 |
| 51 | claude | swebench | 1 | precision | brief_vs_none | 0.4079 | 0.0 | 0.4079 | 21 |
| 52 | claude | swebench | 1 | merge_ready | brief_vs_none | 0.4762 | 0.2857 | 0.1905 | 21 |
| 53 | claude | swebench | 1 | chain_recovery | brief_vs_none | 0.7619 | 0.0 | 0.7619 | 21 |
| 54 | claude | swebench | 1 | return_on_tokens | brief_vs_none | 0.2786 | 0.1777 | 0.1008 | 21 |
| 55 | claude | swebench | 2 | compliance | brief_vs_none | 0.2381 | 0.2381 | 0.0 | 21 |
| 56 | claude | swebench | 2 | recall | brief_vs_none | 0.381 | 0.0 | 0.381 | 21 |
| 57 | claude | swebench | 2 | precision | brief_vs_none | 0.1563 | 0.0 | 0.1563 | 21 |
| 58 | claude | swebench | 2 | merge_ready | brief_vs_none | 0.2381 | 0.2381 | 0.0 | 21 |
| 59 | claude | swebench | 2 | chain_recovery | brief_vs_none | 0.381 | 0.0 | 0.381 | 21 |
| 60 | claude | swebench | 2 | return_on_tokens | brief_vs_none | 0.1573 | 0.1725 | -0.0152 | 21 |
| ... | _+60 more in evals.csv_ | | | | | | | | |

## F2_depth_crossover — 72 evals (Brief wins 24)

| id | model | dataset | depth | metric | comparison | brief | base | Δ | n |
|---|---|---|---|---|---|---|---|---|---|
| 121 | claude | synthetic | 1 | compliance | brief_vs_best_similarity | 0.975 | 1.0 | -0.025 | 40 |
| 122 | claude | synthetic | 2 | compliance | brief_vs_best_similarity | 1.0 | 1.0 | 0.0 | 40 |
| 123 | claude | synthetic | 3 | compliance | brief_vs_best_similarity | 1.0 | 0.7 | 0.3 | 40 |
| 124 | claude | synthetic | 1 | recall | brief_vs_best_similarity | 1.0 | 1.0 | 0.0 | 40 |
| 125 | claude | synthetic | 2 | recall | brief_vs_best_similarity | 1.0 | 1.0 | 0.0 | 40 |
| 126 | claude | synthetic | 3 | recall | brief_vs_best_similarity | 1.0 | 0.7 | 0.3 | 40 |
| 127 | claude | synthetic | 1 | precision | brief_vs_best_similarity | 0.1518 | 0.1429 | 0.0089 | 40 |
| 128 | claude | synthetic | 2 | precision | brief_vs_best_similarity | 0.1607 | 0.1371 | 0.0237 | 40 |
| 129 | claude | synthetic | 3 | precision | brief_vs_best_similarity | 0.1605 | 0.1018 | 0.0587 | 40 |
| 130 | claude | synthetic | 1 | merge_ready | brief_vs_best_similarity | 0.975 | 0.975 | 0.0 | 40 |
| 131 | claude | synthetic | 2 | merge_ready | brief_vs_best_similarity | 0.975 | 0.95 | 0.025 | 40 |
| 132 | claude | synthetic | 3 | merge_ready | brief_vs_best_similarity | 0.975 | 0.7 | 0.275 | 40 |
| 133 | claude | synthetic | 1 | chain_recovery | brief_vs_best_similarity | 1.0 | 1.0 | 0.0 | 40 |
| 134 | claude | synthetic | 2 | chain_recovery | brief_vs_best_similarity | 1.0 | 1.0 | 0.0 | 40 |
| 135 | claude | synthetic | 3 | chain_recovery | brief_vs_best_similarity | 1.0 | 0.7 | 0.3 | 40 |
| 136 | claude | synthetic | 1 | return_on_tokens | brief_vs_best_similarity | 0.8326 | 0.8469 | -0.0143 | 40 |
| 137 | claude | synthetic | 2 | return_on_tokens | brief_vs_best_similarity | 0.847 | 0.8504 | -0.0035 | 40 |
| 138 | claude | synthetic | 3 | return_on_tokens | brief_vs_best_similarity | 0.8557 | 0.5934 | 0.2623 | 40 |
| 139 | claude | dcbench | 1 | compliance | brief_vs_best_similarity | 0.5 | 0.5357 | -0.0357 | 14 |
| 140 | claude | dcbench | 2 | compliance | brief_vs_best_similarity | 0.2619 | 0.3929 | -0.131 | 14 |
| 141 | claude | dcbench | 3 | compliance | brief_vs_best_similarity | 0.3095 | 0.4048 | -0.0952 | 14 |
| 142 | claude | dcbench | 1 | recall | brief_vs_best_similarity | 1.0 | 1.0 | 0.0 | 14 |
| 143 | claude | dcbench | 2 | recall | brief_vs_best_similarity | 0.6429 | 0.6667 | -0.0238 | 14 |
| 144 | claude | dcbench | 3 | recall | brief_vs_best_similarity | 0.6667 | 0.7024 | -0.0357 | 14 |
| 145 | claude | dcbench | 1 | precision | brief_vs_best_similarity | 0.6786 | 0.6786 | 0.0 | 14 |
| 146 | claude | dcbench | 2 | precision | brief_vs_best_similarity | 0.4167 | 0.4405 | -0.0238 | 14 |
| 147 | claude | dcbench | 3 | precision | brief_vs_best_similarity | 0.2238 | 0.2041 | 0.0197 | 14 |
| 148 | claude | dcbench | 1 | merge_ready | brief_vs_best_similarity | 0.2143 | 0.2143 | 0.0 | 14 |
| 149 | claude | dcbench | 2 | merge_ready | brief_vs_best_similarity | 0.1429 | 0.2143 | -0.0714 | 14 |
| 150 | claude | dcbench | 3 | merge_ready | brief_vs_best_similarity | 0.1429 | 0.2143 | -0.0714 | 14 |
| 151 | claude | dcbench | 1 | chain_recovery | brief_vs_best_similarity | 1.0 | 1.0 | 0.0 | 14 |
| 152 | claude | dcbench | 2 | chain_recovery | brief_vs_best_similarity | 0.2857 | 0.3571 | -0.0714 | 14 |
| 153 | claude | dcbench | 3 | chain_recovery | brief_vs_best_similarity | 0.3571 | 0.4286 | -0.0714 | 14 |
| 154 | claude | dcbench | 1 | return_on_tokens | brief_vs_best_similarity | 0.3653 | 0.3921 | -0.0268 | 14 |
| 155 | claude | dcbench | 2 | return_on_tokens | brief_vs_best_similarity | 0.2026 | 0.3016 | -0.099 | 14 |
| 156 | claude | dcbench | 3 | return_on_tokens | brief_vs_best_similarity | 0.2449 | 0.3139 | -0.069 | 14 |
| 157 | claude | swebench | 1 | compliance | brief_vs_best_similarity | 0.4762 | 0.4286 | 0.0476 | 21 |
| 158 | claude | swebench | 2 | compliance | brief_vs_best_similarity | 0.2381 | 0.3333 | -0.0952 | 21 |
| 159 | claude | swebench | 3 | compliance | brief_vs_best_similarity | 0.25 | 0.5 | -0.25 | 12 |
| 160 | claude | swebench | 1 | recall | brief_vs_best_similarity | 0.7619 | 0.7619 | 0.0 | 21 |
| 161 | claude | swebench | 2 | recall | brief_vs_best_similarity | 0.381 | 0.381 | 0.0 | 21 |
| 162 | claude | swebench | 3 | recall | brief_vs_best_similarity | 0.5 | 0.75 | -0.25 | 12 |
| 163 | claude | swebench | 1 | precision | brief_vs_best_similarity | 0.4079 | 0.4143 | -0.0063 | 21 |
| 164 | claude | swebench | 2 | precision | brief_vs_best_similarity | 0.1563 | 0.1548 | 0.0016 | 21 |
| 165 | claude | swebench | 3 | precision | brief_vs_best_similarity | 0.0945 | 0.1903 | -0.0957 | 12 |
| 166 | claude | swebench | 1 | merge_ready | brief_vs_best_similarity | 0.4762 | 0.4286 | 0.0476 | 21 |
| 167 | claude | swebench | 2 | merge_ready | brief_vs_best_similarity | 0.2381 | 0.3333 | -0.0952 | 21 |
| 168 | claude | swebench | 3 | merge_ready | brief_vs_best_similarity | 0.25 | 0.5 | -0.25 | 12 |
| 169 | claude | swebench | 1 | chain_recovery | brief_vs_best_similarity | 0.7619 | 0.7619 | 0.0 | 21 |
| 170 | claude | swebench | 2 | chain_recovery | brief_vs_best_similarity | 0.381 | 0.381 | 0.0 | 21 |
| 171 | claude | swebench | 3 | chain_recovery | brief_vs_best_similarity | 0.5 | 0.75 | -0.25 | 12 |
| 172 | claude | swebench | 1 | return_on_tokens | brief_vs_best_similarity | 0.2786 | 0.2699 | 0.0086 | 21 |
| 173 | claude | swebench | 2 | return_on_tokens | brief_vs_best_similarity | 0.1573 | 0.225 | -0.0676 | 21 |
| 174 | claude | swebench | 3 | return_on_tokens | brief_vs_best_similarity | 0.1428 | 0.2915 | -0.1487 | 12 |
| 175 | gpt | synthetic | 1 | compliance | brief_vs_best_similarity | 1.0 | 1.0 | 0.0 | 20 |
| 176 | gpt | synthetic | 2 | compliance | brief_vs_best_similarity | 1.0 | 1.0 | 0.0 | 20 |
| 177 | gpt | synthetic | 3 | compliance | brief_vs_best_similarity | 1.0 | 0.7 | 0.3 | 20 |
| 178 | gpt | synthetic | 1 | recall | brief_vs_best_similarity | 1.0 | 1.0 | 0.0 | 20 |
| 179 | gpt | synthetic | 2 | recall | brief_vs_best_similarity | 1.0 | 1.0 | 0.0 | 20 |
| 180 | gpt | synthetic | 3 | recall | brief_vs_best_similarity | 1.0 | 0.7 | 0.3 | 20 |
| ... | _+12 more in evals.csv_ | | | | | | | | |

## F3_vs_competitor — 120 evals (Brief wins 81)

| id | model | dataset | depth | metric | comparison | brief | base | Δ | n |
|---|---|---|---|---|---|---|---|---|---|
| 193 | claude | synthetic | 1 | compliance | brief_vs_bm25 | 0.975 | 0.975 | 0.0 | 40 |
| 194 | claude | synthetic | 2 | compliance | brief_vs_bm25 | 1.0 | 1.0 | 0.0 | 40 |
| 195 | claude | synthetic | 3 | compliance | brief_vs_bm25 | 1.0 | 0.7 | 0.3 | 40 |
| 196 | claude | synthetic | ALL | compliance | brief_vs_bm25 | 0.9917 | 0.8917 | 0.1 | 120 |
| 197 | claude | synthetic | 1 | compliance | brief_vs_tfidf | 0.975 | 0.95 | 0.025 | 40 |
| 198 | claude | synthetic | 2 | compliance | brief_vs_tfidf | 1.0 | 1.0 | 0.0 | 40 |
| 199 | claude | synthetic | 3 | compliance | brief_vs_tfidf | 1.0 | 0.7 | 0.3 | 40 |
| 200 | claude | synthetic | ALL | compliance | brief_vs_tfidf | 0.9917 | 0.8833 | 0.1083 | 120 |
| 201 | claude | synthetic | 1 | compliance | brief_vs_dense | 0.975 | 0.925 | 0.05 | 40 |
| 202 | claude | synthetic | 2 | compliance | brief_vs_dense | 1.0 | 0.875 | 0.125 | 40 |
| 203 | claude | synthetic | 3 | compliance | brief_vs_dense | 1.0 | 0.675 | 0.325 | 40 |
| 204 | claude | synthetic | ALL | compliance | brief_vs_dense | 0.9917 | 0.825 | 0.1667 | 120 |
| 205 | claude | synthetic | 1 | compliance | brief_vs_hybrid_rrf | 0.975 | 1.0 | -0.025 | 40 |
| 206 | claude | synthetic | 2 | compliance | brief_vs_hybrid_rrf | 1.0 | 0.975 | 0.025 | 40 |
| 207 | claude | synthetic | 3 | compliance | brief_vs_hybrid_rrf | 1.0 | 0.6 | 0.4 | 40 |
| 208 | claude | synthetic | ALL | compliance | brief_vs_hybrid_rrf | 0.9917 | 0.8583 | 0.1333 | 120 |
| 209 | claude | synthetic | 1 | compliance | brief_vs_rerank_ce | 0.975 | 0.925 | 0.05 | 40 |
| 210 | claude | synthetic | 2 | compliance | brief_vs_rerank_ce | 1.0 | 0.7 | 0.3 | 40 |
| 211 | claude | synthetic | 3 | compliance | brief_vs_rerank_ce | 1.0 | 0.375 | 0.625 | 40 |
| 212 | claude | synthetic | ALL | compliance | brief_vs_rerank_ce | 0.9917 | 0.6667 | 0.325 | 120 |
| 213 | claude | synthetic | 1 | compliance | brief_vs_raptor | 0.975 | 0.475 | 0.5 | 40 |
| 214 | claude | synthetic | 2 | compliance | brief_vs_raptor | 1.0 | 0.225 | 0.775 | 40 |
| 215 | claude | synthetic | 3 | compliance | brief_vs_raptor | 1.0 | 0.05 | 0.95 | 40 |
| 216 | claude | synthetic | ALL | compliance | brief_vs_raptor | 0.9917 | 0.25 | 0.7417 | 120 |
| 217 | claude | synthetic | 1 | compliance | brief_vs_none | 0.975 | 0.0 | 0.975 | 40 |
| 218 | claude | synthetic | 2 | compliance | brief_vs_none | 1.0 | 0.0 | 1.0 | 40 |
| 219 | claude | synthetic | 3 | compliance | brief_vs_none | 1.0 | 0.0 | 1.0 | 40 |
| 220 | claude | synthetic | ALL | compliance | brief_vs_none | 0.9917 | 0.0 | 0.9917 | 120 |
| 221 | claude | dcbench | 1 | compliance | brief_vs_bm25 | 0.5 | 0.4643 | 0.0357 | 14 |
| 222 | claude | dcbench | 2 | compliance | brief_vs_bm25 | 0.2619 | 0.2976 | -0.0357 | 14 |
| 223 | claude | dcbench | 3 | compliance | brief_vs_bm25 | 0.3095 | 0.3333 | -0.0238 | 14 |
| 224 | claude | dcbench | ALL | compliance | brief_vs_bm25 | 0.3571 | 0.3651 | -0.0079 | 42 |
| 225 | claude | dcbench | 1 | compliance | brief_vs_tfidf | 0.5 | 0.4643 | 0.0357 | 14 |
| 226 | claude | dcbench | 2 | compliance | brief_vs_tfidf | 0.2619 | 0.3929 | -0.131 | 14 |
| 227 | claude | dcbench | 3 | compliance | brief_vs_tfidf | 0.3095 | 0.4048 | -0.0952 | 14 |
| 228 | claude | dcbench | ALL | compliance | brief_vs_tfidf | 0.3571 | 0.4206 | -0.0635 | 42 |
| 229 | claude | dcbench | 1 | compliance | brief_vs_dense | 0.5 | 0.4643 | 0.0357 | 14 |
| 230 | claude | dcbench | 2 | compliance | brief_vs_dense | 0.2619 | 0.2381 | 0.0238 | 14 |
| 231 | claude | dcbench | 3 | compliance | brief_vs_dense | 0.3095 | 0.3333 | -0.0238 | 14 |
| 232 | claude | dcbench | ALL | compliance | brief_vs_dense | 0.3571 | 0.3452 | 0.0119 | 42 |
| 233 | claude | dcbench | 1 | compliance | brief_vs_hybrid_rrf | 0.5 | 0.4286 | 0.0714 | 14 |
| 234 | claude | dcbench | 2 | compliance | brief_vs_hybrid_rrf | 0.2619 | 0.3095 | -0.0476 | 14 |
| 235 | claude | dcbench | 3 | compliance | brief_vs_hybrid_rrf | 0.3095 | 0.2738 | 0.0357 | 14 |
| 236 | claude | dcbench | ALL | compliance | brief_vs_hybrid_rrf | 0.3571 | 0.3373 | 0.0198 | 42 |
| 237 | claude | dcbench | 1 | compliance | brief_vs_rerank_ce | 0.5 | 0.5357 | -0.0357 | 14 |
| 238 | claude | dcbench | 2 | compliance | brief_vs_rerank_ce | 0.2619 | 0.2976 | -0.0357 | 14 |
| 239 | claude | dcbench | 3 | compliance | brief_vs_rerank_ce | 0.3095 | 0.3333 | -0.0238 | 14 |
| 240 | claude | dcbench | ALL | compliance | brief_vs_rerank_ce | 0.3571 | 0.3889 | -0.0317 | 42 |
| 241 | claude | dcbench | 1 | compliance | brief_vs_raptor | 0.5 | 0.4286 | 0.0714 | 14 |
| 242 | claude | dcbench | 2 | compliance | brief_vs_raptor | 0.2619 | 0.2381 | 0.0238 | 14 |
| 243 | claude | dcbench | 3 | compliance | brief_vs_raptor | 0.3095 | 0.2976 | 0.0119 | 14 |
| 244 | claude | dcbench | ALL | compliance | brief_vs_raptor | 0.3571 | 0.3214 | 0.0357 | 42 |
| 245 | claude | dcbench | 1 | compliance | brief_vs_random_context | 0.5 | 0.4643 | 0.0357 | 14 |
| 246 | claude | dcbench | 2 | compliance | brief_vs_random_context | 0.2619 | 0.0357 | 0.2262 | 14 |
| 247 | claude | dcbench | 3 | compliance | brief_vs_random_context | 0.3095 | 0.1667 | 0.1429 | 14 |
| 248 | claude | dcbench | ALL | compliance | brief_vs_random_context | 0.3571 | 0.2222 | 0.1349 | 42 |
| 249 | claude | dcbench | 1 | compliance | brief_vs_none | 0.5 | 0.4643 | 0.0357 | 14 |
| 250 | claude | dcbench | 2 | compliance | brief_vs_none | 0.2619 | 0.0 | 0.2619 | 14 |
| 251 | claude | dcbench | 3 | compliance | brief_vs_none | 0.3095 | 0.0238 | 0.2857 | 14 |
| 252 | claude | dcbench | ALL | compliance | brief_vs_none | 0.3571 | 0.1627 | 0.1944 | 42 |
| ... | _+60 more in evals.csv_ | | | | | | | | |

## F4_bayesian_superiority — 28 evals (Brief wins 19)

| id | model | dataset | depth | metric | comparison | brief | base | Δ | n |
|---|---|---|---|---|---|---|---|---|---|
| 313 | claude | synthetic | 3 | P(brief>comp) | brief_vs_bm25 | 1.0 | 0.5 | 0.5 | 40 |
| 314 | claude | synthetic | 3 | P(brief>comp) | brief_vs_tfidf | 1.0 | 0.5 | 0.5 | 40 |
| 315 | claude | synthetic | 3 | P(brief>comp) | brief_vs_dense | 1.0 | 0.5 | 0.5 | 40 |
| 316 | claude | synthetic | 3 | P(brief>comp) | brief_vs_hybrid_rrf | 1.0 | 0.5 | 0.5 | 40 |
| 317 | claude | synthetic | 3 | P(brief>comp) | brief_vs_rerank_ce | 1.0 | 0.5 | 0.5 | 40 |
| 318 | claude | synthetic | 3 | P(brief>comp) | brief_vs_raptor | 1.0 | 0.5 | 0.5 | 40 |
| 319 | claude | synthetic | 3 | P(brief>comp) | brief_vs_none | 1.0 | 0.5 | 0.5 | 40 |
| 335 | claude | dcbench | 3 | P(brief>comp) | brief_vs_bm25 | 0.3436 | 0.5 | -0.1564 | 14 |
| 336 | claude | dcbench | 3 | P(brief>comp) | brief_vs_tfidf | 0.2158 | 0.5 | -0.2842 | 14 |
| 337 | claude | dcbench | 3 | P(brief>comp) | brief_vs_dense | 0.3436 | 0.5 | -0.1564 | 14 |
| 338 | claude | dcbench | 3 | P(brief>comp) | brief_vs_hybrid_rrf | 0.5004 | 0.5 | 0.0004 | 14 |
| 339 | claude | dcbench | 3 | P(brief>comp) | brief_vs_rerank_ce | 0.3436 | 0.5 | -0.1564 | 14 |
| 340 | claude | dcbench | 3 | P(brief>comp) | brief_vs_raptor | 0.5004 | 0.5 | 0.0004 | 14 |
| 341 | claude | dcbench | 3 | P(brief>comp) | brief_vs_none | 0.9906 | 0.5 | 0.4906 | 14 |
| 357 | claude | swebench | 3 | P(brief>comp) | brief_vs_bm25 | 0.329 | 0.5 | -0.171 | 12 |
| 358 | claude | swebench | 3 | P(brief>comp) | brief_vs_tfidf | 0.1023 | 0.5 | -0.3977 | 12 |
| 359 | claude | swebench | 3 | P(brief>comp) | brief_vs_dense | 0.329 | 0.5 | -0.171 | 12 |
| 360 | claude | swebench | 3 | P(brief>comp) | brief_vs_hybrid_rrf | 0.4999 | 0.5 | -0.0001 | 12 |
| 361 | claude | swebench | 3 | P(brief>comp) | brief_vs_rerank_ce | 0.1023 | 0.5 | -0.3977 | 12 |
| 362 | claude | swebench | 3 | P(brief>comp) | brief_vs_raptor | 0.6901 | 0.5 | 0.1901 | 12 |
| 363 | claude | swebench | 3 | P(brief>comp) | brief_vs_none | 0.9755 | 0.5 | 0.4755 | 12 |
| 379 | gpt | synthetic | 3 | P(brief>comp) | brief_vs_bm25 | 0.9983 | 0.5 | 0.4983 | 20 |
| 380 | gpt | synthetic | 3 | P(brief>comp) | brief_vs_tfidf | 0.9983 | 0.5 | 0.4983 | 20 |
| 381 | gpt | synthetic | 3 | P(brief>comp) | brief_vs_dense | 0.9997 | 0.5 | 0.4997 | 20 |
| 382 | gpt | synthetic | 3 | P(brief>comp) | brief_vs_hybrid_rrf | 0.9997 | 0.5 | 0.4997 | 20 |
| 383 | gpt | synthetic | 3 | P(brief>comp) | brief_vs_rerank_ce | 1.0 | 0.5 | 0.5 | 20 |
| 384 | gpt | synthetic | 3 | P(brief>comp) | brief_vs_raptor | 1.0 | 0.5 | 0.5 | 20 |
| 385 | gpt | synthetic | 3 | P(brief>comp) | brief_vs_none | 1.0 | 0.5 | 0.5 | 20 |

## F4_cliffs_delta — 24 evals (Brief wins 15)

| id | model | dataset | depth | metric | comparison | brief | base | Δ | n |
|---|---|---|---|---|---|---|---|---|---|
| 320 | claude | synthetic | 3 | cliffs_delta | brief_vs_bm25 | 0.3 | 0.0 | 0.3 | 40 |
| 322 | claude | synthetic | 3 | cliffs_delta | brief_vs_tfidf | 0.3 | 0.0 | 0.3 | 40 |
| 324 | claude | synthetic | 3 | cliffs_delta | brief_vs_dense | 0.325 | 0.0 | 0.325 | 40 |
| 326 | claude | synthetic | 3 | cliffs_delta | brief_vs_hybrid_rrf | 0.4 | 0.0 | 0.4 | 40 |
| 328 | claude | synthetic | 3 | cliffs_delta | brief_vs_rerank_ce | 0.625 | 0.0 | 0.625 | 40 |
| 330 | claude | synthetic | 3 | cliffs_delta | brief_vs_raptor | 0.95 | 0.0 | 0.95 | 40 |
| 342 | claude | dcbench | 3 | cliffs_delta | brief_vs_bm25 | -0.0408 | 0.0 | -0.0408 | 14 |
| 344 | claude | dcbench | 3 | cliffs_delta | brief_vs_tfidf | -0.1378 | 0.0 | -0.1378 | 14 |
| 346 | claude | dcbench | 3 | cliffs_delta | brief_vs_dense | -0.0102 | 0.0 | -0.0102 | 14 |
| 348 | claude | dcbench | 3 | cliffs_delta | brief_vs_hybrid_rrf | 0.102 | 0.0 | 0.102 | 14 |
| 350 | claude | dcbench | 3 | cliffs_delta | brief_vs_rerank_ce | -0.0051 | 0.0 | -0.0051 | 14 |
| 352 | claude | dcbench | 3 | cliffs_delta | brief_vs_raptor | 0.0204 | 0.0 | 0.0204 | 14 |
| 364 | claude | swebench | 3 | cliffs_delta | brief_vs_bm25 | -0.0833 | 0.0 | -0.0833 | 12 |
| 366 | claude | swebench | 3 | cliffs_delta | brief_vs_tfidf | -0.25 | 0.0 | -0.25 | 12 |
| 368 | claude | swebench | 3 | cliffs_delta | brief_vs_dense | -0.0833 | 0.0 | -0.0833 | 12 |
| 370 | claude | swebench | 3 | cliffs_delta | brief_vs_hybrid_rrf | 0.0 | 0.0 | 0.0 | 12 |
| 372 | claude | swebench | 3 | cliffs_delta | brief_vs_rerank_ce | -0.25 | 0.0 | -0.25 | 12 |
| 374 | claude | swebench | 3 | cliffs_delta | brief_vs_raptor | 0.0833 | 0.0 | 0.0833 | 12 |
| 386 | gpt | synthetic | 3 | cliffs_delta | brief_vs_bm25 | 0.3 | 0.0 | 0.3 | 20 |
| 388 | gpt | synthetic | 3 | cliffs_delta | brief_vs_tfidf | 0.3 | 0.0 | 0.3 | 20 |
| 390 | gpt | synthetic | 3 | cliffs_delta | brief_vs_dense | 0.4 | 0.0 | 0.4 | 20 |
| 392 | gpt | synthetic | 3 | cliffs_delta | brief_vs_hybrid_rrf | 0.4 | 0.0 | 0.4 | 20 |
| 394 | gpt | synthetic | 3 | cliffs_delta | brief_vs_rerank_ce | 0.7 | 0.0 | 0.7 | 20 |
| 396 | gpt | synthetic | 3 | cliffs_delta | brief_vs_raptor | 0.95 | 0.0 | 0.95 | 20 |

## F4_cohens_h — 24 evals (Brief wins 15)

| id | model | dataset | depth | metric | comparison | brief | base | Δ | n |
|---|---|---|---|---|---|---|---|---|---|
| 321 | claude | synthetic | 3 | cohens_h | brief_vs_bm25 | 1.1593 | 0.0 | 1.1593 | 40 |
| 323 | claude | synthetic | 3 | cohens_h | brief_vs_tfidf | 1.1593 | 0.0 | 1.1593 | 40 |
| 325 | claude | synthetic | 3 | cohens_h | brief_vs_dense | 1.2132 | 0.0 | 1.2132 | 40 |
| 327 | claude | synthetic | 3 | cohens_h | brief_vs_hybrid_rrf | 1.3694 | 0.0 | 1.3694 | 40 |
| 329 | claude | synthetic | 3 | cohens_h | brief_vs_rerank_ce | 1.8235 | 0.0 | 1.8235 | 40 |
| 331 | claude | synthetic | 3 | cohens_h | brief_vs_raptor | 2.6906 | 0.0 | 2.6906 | 40 |
| 343 | claude | dcbench | 3 | cohens_h | brief_vs_bm25 | -0.051 | 0.0 | -0.051 | 14 |
| 345 | claude | dcbench | 3 | cohens_h | brief_vs_tfidf | -0.1992 | 0.0 | -0.1992 | 14 |
| 347 | claude | dcbench | 3 | cohens_h | brief_vs_dense | -0.051 | 0.0 | -0.051 | 14 |
| 349 | claude | dcbench | 3 | cohens_h | brief_vs_hybrid_rrf | 0.0786 | 0.0 | 0.0786 | 14 |
| 351 | claude | dcbench | 3 | cohens_h | brief_vs_rerank_ce | -0.051 | 0.0 | -0.051 | 14 |
| 353 | claude | dcbench | 3 | cohens_h | brief_vs_raptor | 0.0259 | 0.0 | 0.0259 | 14 |
| 365 | claude | swebench | 3 | cohens_h | brief_vs_bm25 | -0.1838 | 0.0 | -0.1838 | 12 |
| 367 | claude | swebench | 3 | cohens_h | brief_vs_tfidf | -0.5236 | 0.0 | -0.5236 | 12 |
| 369 | claude | swebench | 3 | cohens_h | brief_vs_dense | -0.1838 | 0.0 | -0.1838 | 12 |
| 371 | claude | swebench | 3 | cohens_h | brief_vs_hybrid_rrf | 0.0 | 0.0 | 0.0 | 12 |
| 373 | claude | swebench | 3 | cohens_h | brief_vs_rerank_ce | -0.5236 | 0.0 | -0.5236 | 12 |
| 375 | claude | swebench | 3 | cohens_h | brief_vs_raptor | 0.2061 | 0.0 | 0.2061 | 12 |
| 387 | gpt | synthetic | 3 | cohens_h | brief_vs_bm25 | 1.1593 | 0.0 | 1.1593 | 20 |
| 389 | gpt | synthetic | 3 | cohens_h | brief_vs_tfidf | 1.1593 | 0.0 | 1.1593 | 20 |
| 391 | gpt | synthetic | 3 | cohens_h | brief_vs_dense | 1.3694 | 0.0 | 1.3694 | 20 |
| 393 | gpt | synthetic | 3 | cohens_h | brief_vs_hybrid_rrf | 1.3694 | 0.0 | 1.3694 | 20 |
| 395 | gpt | synthetic | 3 | cohens_h | brief_vs_rerank_ce | 1.9823 | 0.0 | 1.9823 | 20 |
| 397 | gpt | synthetic | 3 | cohens_h | brief_vs_raptor | 2.6906 | 0.0 | 2.6906 | 20 |

## F4_bca_ci — 12 evals (Brief wins 7)

| id | model | dataset | depth | metric | comparison | brief | base | Δ | n |
|---|---|---|---|---|---|---|---|---|---|
| 332 | claude | synthetic | 1 | compliance_ci | brief_point+CI | 0.975 | 0.875 | 0.1 | 40 |
| 333 | claude | synthetic | 2 | compliance_ci | brief_point+CI | 1.0 | 1.0 | 0.0 | 40 |
| 334 | claude | synthetic | 3 | compliance_ci | brief_point+CI | 1.0 | 1.0 | 0.0 | 40 |
| 354 | claude | dcbench | 1 | compliance_ci | brief_point+CI | 0.5 | 0.3214 | 0.1786 | 14 |
| 355 | claude | dcbench | 2 | compliance_ci | brief_point+CI | 0.2619 | 0.0833 | 0.1786 | 14 |
| 356 | claude | dcbench | 3 | compliance_ci | brief_point+CI | 0.3095 | 0.1429 | 0.1667 | 14 |
| 376 | claude | swebench | 1 | compliance_ci | brief_point+CI | 0.4762 | 0.2381 | 0.2381 | 21 |
| 377 | claude | swebench | 2 | compliance_ci | brief_point+CI | 0.2381 | 0.0476 | 0.1905 | 21 |
| 378 | claude | swebench | 3 | compliance_ci | brief_point+CI | 0.25 | 0.0 | 0.25 | 12 |
| 398 | gpt | synthetic | 1 | compliance_ci | brief_point+CI | 1.0 | 1.0 | 0.0 | 20 |
| 399 | gpt | synthetic | 2 | compliance_ci | brief_point+CI | 1.0 | 1.0 | 0.0 | 20 |
| 400 | gpt | synthetic | 3 | compliance_ci | brief_point+CI | 1.0 | 1.0 | 0.0 | 20 |

## F5_return_on_tokens — 32 evals (Brief wins 28)

| id | model | dataset | depth | metric | comparison | brief | base | Δ | n |
|---|---|---|---|---|---|---|---|---|---|
| 401 | claude | synthetic | ALL | rot | brief_graph_3hop_vs_none_rot | 0.8451 | 0.0 | 0.8451 | 120 |
| 402 | claude | synthetic | ALL | rot | bm25_vs_none_rot | 0.7566 | 0.0 | 0.7566 | 120 |
| 403 | claude | synthetic | ALL | rot | tfidf_vs_none_rot | 0.7496 | 0.0 | 0.7496 | 120 |
| 404 | claude | synthetic | ALL | rot | dense_vs_none_rot | 0.6989 | 0.0 | 0.6989 | 120 |
| 405 | claude | synthetic | ALL | rot | hybrid_rrf_vs_none_rot | 0.7276 | 0.0 | 0.7276 | 120 |
| 406 | claude | synthetic | ALL | rot | rerank_ce_vs_none_rot | 0.5644 | 0.0 | 0.5644 | 120 |
| 407 | claude | synthetic | ALL | rot | raptor_vs_none_rot | 0.2137 | 0.0 | 0.2137 | 120 |
| 408 | claude | synthetic | ALL | rot | none_vs_none_rot | 0.0 | 0.0 | 0.0 | 120 |
| 409 | claude | dcbench | ALL | rot | brief_graph_3hop_vs_none_rot | 0.2709 | 0.151 | 0.1199 | 42 |
| 410 | claude | dcbench | ALL | rot | bm25_vs_none_rot | 0.2751 | 0.151 | 0.1241 | 42 |
| 411 | claude | dcbench | ALL | rot | tfidf_vs_none_rot | 0.3186 | 0.151 | 0.1676 | 42 |
| 412 | claude | dcbench | ALL | rot | dense_vs_none_rot | 0.26 | 0.151 | 0.109 | 42 |
| 413 | claude | dcbench | ALL | rot | hybrid_rrf_vs_none_rot | 0.2547 | 0.151 | 0.1037 | 42 |
| 414 | claude | dcbench | ALL | rot | rerank_ce_vs_none_rot | 0.292 | 0.151 | 0.141 | 42 |
| 415 | claude | dcbench | ALL | rot | raptor_vs_none_rot | 0.2428 | 0.151 | 0.0918 | 42 |
| 416 | claude | dcbench | ALL | rot | none_vs_none_rot | 0.151 | 0.151 | 0.0 | 42 |
| 417 | claude | swebench | ALL | rot | brief_graph_3hop_vs_none_rot | 0.2012 | 0.1362 | 0.0651 | 54 |
| 418 | claude | swebench | ALL | rot | bm25_vs_none_rot | 0.1904 | 0.1362 | 0.0543 | 54 |
| 419 | claude | swebench | ALL | rot | tfidf_vs_none_rot | 0.2152 | 0.1362 | 0.079 | 54 |
| 420 | claude | swebench | ALL | rot | dense_vs_none_rot | 0.2286 | 0.1362 | 0.0925 | 54 |
| 421 | claude | swebench | ALL | rot | hybrid_rrf_vs_none_rot | 0.2007 | 0.1362 | 0.0645 | 54 |
| 422 | claude | swebench | ALL | rot | rerank_ce_vs_none_rot | 0.1943 | 0.1362 | 0.0581 | 54 |
| 423 | claude | swebench | ALL | rot | raptor_vs_none_rot | 0.1839 | 0.1362 | 0.0477 | 54 |
| 424 | claude | swebench | ALL | rot | none_vs_none_rot | 0.1362 | 0.1362 | 0.0 | 54 |
| 425 | gpt | synthetic | ALL | rot | brief_graph_3hop_vs_none_rot | 0.8896 | 0.0 | 0.8896 | 60 |
| 426 | gpt | synthetic | ALL | rot | bm25_vs_none_rot | 0.7738 | 0.0 | 0.7738 | 60 |
| 427 | gpt | synthetic | ALL | rot | tfidf_vs_none_rot | 0.7594 | 0.0 | 0.7594 | 60 |
| 428 | gpt | synthetic | ALL | rot | dense_vs_none_rot | 0.7015 | 0.0 | 0.7015 | 60 |
| 429 | gpt | synthetic | ALL | rot | hybrid_rrf_vs_none_rot | 0.7446 | 0.0 | 0.7446 | 60 |
| 430 | gpt | synthetic | ALL | rot | rerank_ce_vs_none_rot | 0.6096 | 0.0 | 0.6096 | 60 |
| 431 | gpt | synthetic | ALL | rot | raptor_vs_none_rot | 0.2285 | 0.0 | 0.2285 | 60 |
| 432 | gpt | synthetic | ALL | rot | none_vs_none_rot | 0.0 | 0.0 | 0.0 | 60 |

## F6_ablation — 8 evals (Brief wins 7)

| id | model | dataset | depth | metric | comparison | brief | base | Δ | n |
|---|---|---|---|---|---|---|---|---|---|
| 433 | claude | synthetic | 3 | compliance | stripped_brief_vs_random | 1.0 | 0.0 | 1.0 | 40 |
| 434 | claude | synthetic | 3 | compliance | stripped_brief_vs_floor | 1.0 | 0.0 | 1.0 | 40 |
| 435 | claude | dcbench | 3 | compliance | stripped_brief_vs_random | 0.3095 | 0.1667 | 0.1429 | 14 |
| 436 | claude | dcbench | 3 | compliance | stripped_brief_vs_floor | 0.3095 | 0.0238 | 0.2857 | 14 |
| 437 | claude | swebench | 3 | compliance | stripped_brief_vs_random | 0.25 | 0.3333 | -0.0833 | 12 |
| 438 | claude | swebench | 3 | compliance | stripped_brief_vs_floor | 0.25 | 0.0 | 0.25 | 12 |
| 439 | gpt | synthetic | 3 | compliance | stripped_brief_vs_random | 1.0 | 0.0 | 1.0 | 20 |
| 440 | gpt | synthetic | 3 | compliance | stripped_brief_vs_floor | 1.0 | 0.0 | 1.0 | 20 |
