# changelog
roughly follows [keep a changelog](https://keepachangelog.com). newest first.

## [unreleased]

### added
- structured graph memory arm (typed links, bounded traversal, per type decay) and the
  depth crossover benchmark it is built to demonstrate
- the headline arm set plus competitor adapters, four datasets, and the full metrics
  suite (compliance, correctness, retrieval, cost, calibration)
- statistics module: bca bootstrap, bayesian estimation, friedman and nemenyi, mixed
  effects logistic regression, changepoint detection, mediation, dominance, model
  selection by aic and bic, and power analysis
- the theory module: similarity decay law, recovery model, crossover depth solver, and
  the token economics around return on tokens
- native kernels (c with neon, rust via pyo3, cuda) with a reference parity check, a go
  orchestrator, and cross language reference implementations in c#, java and r
- config driven sweep, a run manifest that records device, versions and git state, and
  a reproduce script for shell and powershell

### notes
- the four arm ablation reports the offline stub numbers as measured, unrounded. the
  stub is a deliberately weak proxy that compresses compliance, so it corroborates the
  ordering rather than the magnitude. the magnitude lives in the synthetic crossover.
- depth labels currently come from a single pass heuristic. the dual annotation pass is
  required before the depth crossover table is used for published numbers.