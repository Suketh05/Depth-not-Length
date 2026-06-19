# SWE-ContextBench — real leaderboard + Brief PROJECTION

**Brief's row is a PROJECTION, not a measurement.** SWE-ContextBench runs 1,476 real coding tasks across 51 repos with test-suite execution; we did not run it. Source for all real rows: [arXiv:2602.08316](https://arxiv.org/abs/2602.08316).

## Real published leaderboard (Tier-2, cited)

| system | task resolution % | token cost | status |
|---|---|---|---|
| Oracle Summary Learning (upper bound) | 34.34 | 6.66 | vendor/paper |
| Supermemory | 30.30 | 5.04 | vendor (self-reported) |
| OpenViking | 29.20 | 4.20 | vendor |
| Mem0 | 24.24 | 4.72 | vendor |
| **Brief (PROJECTED)** | **27** (range 24–31) | **~4.3** | **projected — NOT measured** |

## Projection methodology (so it's auditable)

- **Resolution rate ~27% (range 24–31):** anchored to our measured swebench proxy, where Brief's graph is *middling* on raw code retrieval (compliance ~0.33 vs best baseline ~0.37). That places Brief in the **mid-cluster** with Mem0/OpenViking — **not** at or above the Oracle upper bound. We deliberately did **not** project a win.
- **Token cost ~4.3 (low end):** Brief's structured graph surfaces compact, dependency-relevant context (post-pruning precision), aligned with the paper's finding that compact summaries (217 tokens) beat full context (25.6k tokens). This is Brief's one genuinely data-backed edge here (token efficiency / Return on Tokens).

## Honest verdict

Grounded in real data, Brief projects **competitive but mid-pack on raw coding resolution, with a token-efficiency advantage** — it does **not** project to beat everyone, and inflating it would be fabrication. SWE-ContextBench tests code-experience reuse; Brief's measured strength is **product/decision context** (RoT study 46%→95%), a different axis. A real leaderboard claim requires running the official 1,476-task harness.
