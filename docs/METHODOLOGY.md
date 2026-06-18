# Methodology

## Thesis
Coding agents fail not from token scarcity but because similarity retrieval cannot
recover a decision that is causally linked to the task yet phrased unlike it and sitting
several hops away. Structured memory follows the link instead of guessing by
resemblance, so at a fixed per-call budget it honours past decisions more often and the
total cost-to-correct drops.

## Design
- **Arms** implement one contract (`write` / `retrieve(query, budget)`); the harness
  swaps the arm and holds everything else fixed (the fairness lock).
- **Depth = strip-count.** d=1 states the constraint in the query (control); d=2 strips
  it into one corpus node; d=3 splits it into a constraint + a justification node linked
  by `constrains`. The synthetic benchmark dials depth directly and makes node→query
  similarity decay geometrically along the chain (an explicit `s_i = s0·ρ^i`).
- **Budget** is per dataset, identical across arms. The depth penalty bites under tight
  budgets (the realistic regime); we report the budget sweep rather than one point.
- **Offline & deterministic.** A stub agent honours exactly the identifiers present in
  the retrieved context, so compliance is driven by retrieval quality and cost by real
  prompt size — the accuracy and token-economics results fall out with no API calls.

## From measurement to theory
The decay law `s_i = s0·ρ^i` is fit from embedding similarities along chains; the
structured per-link reliability `q` is fit from observed recovery; together they predict
`d⋆`, which is compared against the empirically measured crossover (with a bootstrap CI).
AIC/BIC select the geometric law over power-law / stretched-exponential alternatives.

## Reporting
Every headline number ships with a confidence interval, a paired significance test, an
effect size, and — for head-to-heads — a Bayesian `P(Brief > competitor)`. Many arms are
compared with a Friedman omnibus + Nemenyi critical-difference diagram.
