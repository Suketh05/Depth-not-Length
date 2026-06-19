# SWE-ContextBench — published leaderboard (cited)

Published results from SWE-ContextBench ([arXiv:2602.08316](https://arxiv.org/abs/2602.08316)),
which runs 1,476 real coding tasks across 51 repositories with test-suite execution. These
are the paper's reported numbers (Tier-2, vendor/paper). Brief has not been run on this
benchmark, so it does not appear here; running the official 1,476-task harness is required
for any Brief number on it.

| system | task resolution % | token cost | source |
|---|---|---|---|
| Oracle Summary Learning (upper bound) | 34.34 | 6.66 | arXiv:2602.08316 |
| Supermemory | 30.30 | 5.04 | arXiv:2602.08316 (self-reported) |
| OpenViking | 29.20 | 4.20 | arXiv:2602.08316 |
| Mem0 | 24.24 | 4.72 | arXiv:2602.08316 |

Key finding (independent validation of Brief's thesis): compact summarised context
(avg 217 tokens) outperforms full context (avg 25,634 tokens) — the Return-on-Tokens
principle. SWE-ContextBench measures code-experience reuse; Brief's measured strength is
product/decision context (a different axis).
