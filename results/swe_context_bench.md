# SWE-ContextBench — leaderboard (published) + Brief (modeled)

Published results from SWE-ContextBench ([arXiv:2602.08316](https://arxiv.org/abs/2602.08316)):
1,476 real coding tasks across 51 repositories with test-suite execution. Real rows are the
paper's reported numbers (vendor/paper). **Brief's row is `modeled`, not measured** — Brief
has not been run on the official harness; its value is modeled from our measured data (method
below).

| system | task resolution % | token cost | tier |
|---|---|---|---|
| Oracle Summary Learning (upper bound) | 34.34 | 6.66 | vendor |
| Supermemory | 30.30 | 5.04 | vendor (self-reported) |
| OpenViking | 29.20 | 4.20 | vendor |
| **Brief (modeled)** | **27** (range 24–31) | **~4.3** | **modeled** |
| Mem0 | 24.24 | 4.72 | vendor |

## How Brief's modeled row was derived (auditable)

- **Resolution ~27% (24–31):** modeled from our measured swebench proxy (Brief is mid-pack on
  raw code retrieval, ~0.33 vs best baseline 0.37) + the published cluster + the paper's
  finding that compact summarised context wins. Brief models into the mid-cluster, not above
  the Oracle upper bound.
- **Token cost ~4.3 (low):** modeled from Brief's measured Return-on-Tokens / compact-context
  edge.

## Note

Brief's row is `modeled`, not a measured benchmark result; a measured number requires running
the official 1,476-task harness. SWE-ContextBench tests code-experience reuse; Brief's measured
strength is product/decision context (RoT 46%→95%), a different axis. Independent validation of
the principle: compact summaries (avg 217 tokens) beat full context (avg 25,634 tokens).
