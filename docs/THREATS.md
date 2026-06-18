# Threats to validity

We list these so reviewers don't have to find them.

- **Depth labels are single-annotator.** `benchmarks/depth_labels.jsonl` is an automated
  one-pass heuristic (`dual_annotated: false`). The depth-crossover table should clear a
  real two-annotator pass (the apparatus — Cohen's κ + certification — ships in
  `datasets/depth_labeling.py`) before its numbers are published.
- **The similarity arms stand in for their class.** BM25/TF-IDF are exact; the dense arm
  uses a deterministic hashing embedding offline (a real neural embedding backend is
  available behind the `neural` extra). Mem0/Zep/GraphRAG/etc. require their own isolated
  environments to run live; offline we report the runnable arms and keep vendor numbers
  in a separate, cited tier.
- **Merge-ready correctness is a proxy** (non-empty + non-refusal + all decisions
  honoured) over a stub agent, not a real diff+test harness; the contract is arranged so
  a real harness can replace it without touching scoring.
- **The synthetic benchmark is controlled, not natural.** It isolates the mechanism the
  theory predicts and is grounded in how real codebases hide decisions, but it is
  generated; dcbench/SWE-bench/LongMemEval provide the ecological evidence, reported
  honestly (where the effect is modest, we say so).
- **The empirical `d⋆` uses an interpolation-based changepoint** with a cluster bootstrap,
  not a likelihood-fit segmented logistic; both are valid, and the estimate ships with a
  CI and an existence fraction.
- **Robustness across base models** is partial: GPT/open-weight backends are wired but
  the live robustness row needs their API keys.
