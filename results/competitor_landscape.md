# Tier-2 competitor landscape (vendor-reported, cited)

**These are published numbers on each system's OWN benchmark — NOT a controlled head-to-head and not directly comparable cell-to-cell.** The takeaway is the industry-wide pattern: a structured context/memory layer beats the no-context / vanilla baseline everywhere — which independently corroborates Brief's thesis. Brief's measured Tier-1 result (our harness) is separate.

| system | metric | score | vs baseline | benchmark | status | source |
|---|---|---|---|---|---|---|
| Brief | agent decision compliance | 95.0 | 46.0 (codebase alone (no product context)) | Brief RoT study (same agent ± product context) | vendor-reported | [link](https://briefhq.ai/return-on-tokens/) |
| Brief | merge-ready task rate | 100.0 | 25.0 (codebase alone) | Brief RoT study | vendor-reported | [link](https://briefhq.ai/return-on-tokens/) |
| Mem0 | LoCoMo LLM-judge score | 66.9 | 52.9 (OpenAI memory) | LoCoMo | peer-reviewed | [link](https://arxiv.org/abs/2504.19413) |
| Mem0 (2026 algo) | LoCoMo | 92.5 | — | LoCoMo | vendor-reported | [link](https://mem0.ai/blog/ai-memory-benchmarks-in-2026) |
| Mem0 (2026 algo) | LongMemEval | 94.4 | — | LongMemEval | vendor-reported | [link](https://mem0.ai/blog/ai-memory-benchmarks-in-2026) |
| Zep | Deep Memory Retrieval (GPT-4 Turbo) | 94.8 | 93.4 (MemGPT/Letta) | DMR | peer-reviewed | [link](https://arxiv.org/abs/2501.13956) |
| Zep | LongMemEval accuracy lift | 18.5 | 0.0 (full-context baseline) | LongMemEval (Δ%) | vendor-reported | [link](https://blog.getzep.com/state-of-the-art-agent-memory/) |
| GraphRAG | comprehensiveness win-rate vs vector RAG | 77.5 | 27.0 (vector RAG) | QFS comprehensiveness | peer-reviewed | [link](https://arxiv.org/abs/2404.16130) |
| Supermemory | LoCoMo P@1 | 59.7 | 34.4 (competing providers) | LoCoMo | self-reported (unverified) | [link](https://supermemory.ai/) |
| Supermemory | LongMemEval-S overall | 85.4 | — | LongMemEval-S | self-reported (unverified) | [link](https://supermemory.ai/) |
| Supermemory | SWE-Context-Bench task resolution | 30.3 | — | SWE Context Bench (coding) | self-reported (unverified) | [link](https://arxiv.org/abs/2602.08316) |
| Letta/MemGPT | DMR | 93.4 | — | DMR | peer-reviewed | [link](https://arxiv.org/abs/2310.08560) |
| Letta/MemGPT | LoCoMo (filesystem, gpt-4o-mini) | 74.0 | — | LoCoMo | vendor-reported | [link](https://www.letta.com/blog/benchmarking-ai-agent-memory/) |
| Cognee | HotpotQA multi-hop (graph > vector) | graph > vector | — | HotpotQA | vendor-reported | [link](https://www.cognee.ai/research-and-evaluation-results) |

## Brief — measured (Tier-1, our harness, same model/budget/tasks)

- agent + Brief vs agent-alone: **0.70 vs 0.08** compliance (all data, Claude + GPT-5.1)
- synthetic depth-3 crossover: **Brief 1.00 vs best similarity 0.70** (P(Brief>bm25)=1.000)

_Note: Supermemory's claims are self-reported and not independently verified (per third-party coverage). Mem0/Zep/GraphRAG/Letta numbers are from peer-reviewed papers._
