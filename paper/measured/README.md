# measured results

real-backend benchmark runs, kept as durable artifacts so every headline number in the
paper can be traced to the rows it came from. these are tier-1 measured results
(identical conditions across arms, the fairness lock), distinct from the tier-2
vendor-reported numbers in `competitors/`.

## what's here

- `synthetic_real_sonnet/` — the controlled depth-crossover benchmark run on the real
  `claude-sonnet-4-6` model. `rows.jsonl` is the canonical scored output (one attempt per
  line, the same `ScoredRow` schema the analysis consumes); `summary.md` is the
  depth-crossover table plus the significance battery and the measured dollar cost.

## how it was produced

every arm sees the same model, the same per-dataset retrieval budget, and the same
tools; memory architecture is the only thing that varies. the model is called through
the standard `AnthropicLLMClient` with `offline=false`; the dataset is deterministic
(seed=0). the api key is read from the environment at run time and never stored in the
repo. to reproduce, set `ANTHROPIC_API_KEY` and run the config-driven sweep against the
synthetic dataset with `offline: false`.

## provenance rules

- numbers here are reproducible from the committed `rows.jsonl` by re-running the
  analysis; nothing is hand-edited.
- measured and vendor-reported numbers never share a table.
- results that don't favour the structured arm are reported as measured, not dropped.
