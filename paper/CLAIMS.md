# Claim → code → table map

Every claim in the paper traces to code and to a regenerable artifact.

| Claim | Code | Artifact |
|---|---|---|
| Similarity recovery has a depth ceiling `s_0^d·ρ^{d(d+1)/2}` | `theory/recovery.py` | derivation + `theory` tests |
| Structured recovery `q^d` decays gently; advantage → ∞ | `theory/recovery.py` | `RecoveryModel` tests |
| A crossover depth `d⋆` exists | `theory/crossover.py` | `membench theory` |
| `s_i = s0·ρ^i` holds on real embeddings | `analysis/decay_fit.py` | decay-fit figure |
| Predicted `d⋆` ≈ measured `d⋆` | `theory/fit.py`, `stats/changepoint.py` | crossover figure + CI |
| Similarity collapses with depth; Brief stays flat | `analysis/tables.py` | depth-crossover table |
| Brief recovers the governing decision via the link | `retrieval/graph/` | unit + e2e tests |
| Joint accuracy + cost win | `theory/information.py`, `stats/dominance.py` | cost-frontier figure |
| Same budget, more shipped-right work (Return on Tokens) | `metrics/cost.py` | report exec summary |
| The win runs *through* retrieval (mechanism) | `stats/mediation.py` | mediation result |
| Brief beats competitor X with probability p | `stats/bayes.py` | competitor matrix |
| Rankings differ significantly across arms | `stats/multiple_comparisons.py` | critical-difference diagram |
