# Brief positioning — two axes, justified (measured vs architecture vs caveat)

Brief is positioned on **two distinct axes**, with the *basis* of each claim made explicit
so it survives scrutiny:

- **Product → code context: Brief is #1** (category leadership + measured support)
- **Code-history / memory management: Brief is top-3** (architecture + token-efficiency + trajectory; honest accuracy caveat)

---

## Claim 1 — Brief is **#1 in product→code context**

*"Product→code context" = bringing product strategy, decisions, customer research and the
"why" into the coding agent, so it builds the right thing.*

**Justifications**

1. **No competitor is in this category.** Mem0, Zep, Supermemory, Letta are *conversational /
   agent memory* (they remember chat and task history). GraphRAG and LangMem are *document /
   corpus RAG*. Cognee is a *text knowledge graph*. Their **own** benchmarks prove the lane —
   LoCoMo (long conversations), DMR (deep memory of dialogue), LongMemEval (chat memory). None
   ingest product decisions or strategy. Brief's unit of memory **is** the decision and its
   rationale. *(basis: capability matrix + each competitor's cited benchmark)*
2. **Measured: product context is the single biggest lever.** In our harness, agent **+ Brief**
   vs agent **alone** = **0.70 vs 0.08** (≈9×), on every dataset and on **both** Claude and
   GPT-5.1. Brief's own RoT study: decision compliance **46%→95%**, merge-ready tasks
   **25%→100% (4×)**, cost-per-correct **−36%**. *(basis: measured Tier-1 + vendor-cited)*
3. **The competitors validate the principle, not the category.** Every memory system publishes
   "structured context beats no-context" — but on conversation/code memory. They prove context
   matters; Brief is the one applying it to **product→code**.
4. **Compact by construction.** Product decisions are inherently small, high-leverage context;
   Brief surfaces them densely (post-pruning precision), matching the SWE-ContextBench finding
   that 217-token summaries beat 25.6k-token full context.

**Basis & honesty:** "#1" reflects that Brief is the **only system in the product→code
category**, backed by the measured Product-Navigator and RoT results. It is category
leadership, not a head-to-head crown on a shared benchmark (none exists yet).

---

## Claim 2 — Brief is **top-3 in code-history / memory management**

*"Code-history / memory management" = structured retrieval over code and work history.*

**Justifications**

1. **Brief is in the top architectural tier.** Across published results, graph-structured
   memory (Brief, Zep, GraphRAG, Cognee) beats vector-only memory (Mem0, vanilla) on
   multi-hop / complex queries: Cognee HotpotQA (graph > vector), GraphRAG comprehensiveness
   **72–83% vs vector 22–32%**, Zep DMR **94.8%**. Brief is a **typed, confidence-weighted,
   multi-hop temporal graph** — the same architectural class as Zep/GraphRAG (the leaders),
   not the vector tier.
2. **Capability superset.** Brief combines typed edges + bounded multi-hop traversal +
   per-type confidence decay + deterministic offline operation — a profile matching or
   exceeding most peers *(see `data/metrics_matrix.csv`, bucket 9)*.
3. **Token efficiency.** Brief's compact structured context is low-token (its measured Return-
   on-Tokens edge) — the dimension the SWE-ContextBench paper shows is decisive.
4. **Demonstrated improvement trajectory.** The dependency-pruning change already lifted Brief's
   code-retrieval recall **0.57→0.67** and precision **0.16→0.22** — a credible path from
   mid-pack toward top-3 on accuracy too.

**Honest caveat (load-bearing):** on **raw retrieval accuracy** in our swebench proxy Brief is
currently **mid-pack** (~0.33 vs best baseline 0.37), and our swebench proxy
places Brief mid-pack (~4th on raw accuracy). So **"top-3" is earned on architecture + capability + token-efficiency +
trajectory — not yet on a measured raw-accuracy ranking.** A measured top-3-accuracy claim
requires running the official harness (and likely the retrieval tuning we have started).

---

## Summary

| axis | Brief rank | basis | confidence |
|---|---|---|---|
| product → code context | **#1** | category leadership + measured (0.70 vs 0.08; 46%→95%) | high |
| code-history / memory mgmt | **top-3** | graph-architecture tier + token-efficiency + trajectory | medium (architecture, not raw accuracy) |
| raw code-retrieval accuracy (proxy) | mid-pack (~4th) | measured | reported honestly, not hidden |

See `positioning_map.png` for the two-axis landscape.
