# measured results

real-backend benchmark runs kept as durable artifacts, so every headline number can be
traced to the rows it came from. tier-1 measured results (identical conditions across
arms — the fairness lock), distinct from the tier-2 vendor-reported numbers in
`competitors/`.

## what's here

- `rows.jsonl` — the canonical scored output: 1,824 attempts (8 arms × 3 datasets × 3
  depths × n=40) on the real `claude-sonnet-4-6` model. Every table is reproducible from
  this file.
- `summary.md` — the headline (the three marquee results + the honest boundary).
- `dossier.md` — the full **60-eval breakdown**: performance by metric × dataset,
  depth-stratified, Product-Navigator (Brief vs agent-alone), the four-arm ablation,
  Brief-vs-each-competitor pairwise, the statistical battery (Friedman/Nemenyi, BCa CIs,
  Bayesian superiority, Cliff's δ, Cohen's h), token economics / Return on Tokens, and
  robustness cross-cuts.

## how it was produced

every arm sees the same model, the same per-dataset retrieval budget, and the same
tools; memory architecture is the only variable. the model is called through the
standard `AnthropicLLMClient` with `offline=false`; datasets are deterministic (seed=0).
the structured-graph arm packs dependency-linked nodes (the decisions code/justifications
link to) ahead of incidental neighbours, so the governing context is not diluted — a
retrieval-quality change to the arm, with the baselines untouched. the api key is read
from the environment at run time and never stored in the repo.

## provenance rules

- numbers here are reproducible from `rows.jsonl` by re-running the analysis; nothing is
  hand-edited.
- measured and vendor-reported numbers never share a table.
- results that don't favour the structured arm (the natural-data raw-retrieval
  comparisons) are reported as measured, not dropped.
