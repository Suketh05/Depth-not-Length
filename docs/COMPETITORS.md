# Competitor Landscape

This document maps the memory / context-layer systems that "Depth, Not Length"
compares against. For each competitor it gives a one-paragraph positioning, the
status of its adapter **in this repository**, and any **vendor-tier published
number** transcribed verbatim with its citation.

## How to read the numbers (three provenance tiers)

Every figure below carries an explicit provenance label. They are **never** mixed,
and a modeled number is **never** presented as measured:

- **vendor** — a number *published by the vendor or its paper*, under **their own**
  benchmark, dataset, model, and conditions. Not a controlled head-to-head; not
  directly comparable cell-to-cell with our harness. Source: the cited paper / page.
  These live in [`competitors/registry.yaml`](../competitors/registry.yaml) (Tier-2,
  with placeholders pending verification), the rendered table
  [`results/competitor_landscape.md`](../results/competitor_landscape.md) /
  [`results/data/competitor_landscape.csv`](../results/data/competitor_landscape.csv),
  and section `10_competitor_head_to_head` of
  [`results/METRICS.md`](../results/METRICS.md).
- **modeled** — a value *derived from a vendor's published lift over its own
  baseline* and re-expressed on our compliance scale. It is an **estimate, not a
  run** — explicitly flagged "modeled from published lift over baseline; not a
  measured run" in `results/METRICS.md` §10. Modeled numbers exist only for
  orientation and are never reported as a measured outcome.
- **measured** — a Tier-1 number produced **inside our own harness** on identical
  tasks / budget / model. Source: `results/METRICS.md`, `results/positioning.md`,
  `results/data/`. Only Brief currently has measured-in-harness compliance recorded
  in `METRICS.md` §10; the external adapters become measured competitors only when
  their (isolated) backend package is installed and run (see "Adapter status" below).
- **paper-reported** — a figure read from the paper digest / LaTeX
  (`depth_not_length_FIXED.tex`). Kept separate from repo-measured numbers: the
  paper's public-benchmark table (`tab:stdbench`) runs competitors on a *unified
  harness*, which is a different artifact from the vendor-published headline in
  `tab:landscape`. Where the same value appears in both (e.g. Mem0 LoCoMo 66.9) it
  is a coincidence of benchmark, not a claim that they are the same measurement.

> **Important framing** (from `results/competitor_landscape.md`): the vendor numbers
> are published results on each system's *own* benchmark — **not** a controlled
> head-to-head and not directly comparable cell-to-cell. The takeaway is the
> industry-wide pattern that a structured context/memory layer beats the no-context
> baseline everywhere, which independently corroborates Brief's thesis. Brief's
> measured Tier-1 result is separate.

## Adapter architecture (this repo)

Adapters live in
[`src/membench/retrieval/external/`](../src/membench/retrieval/external/). Each wraps
a real external memory system behind the same `MemorySystem` contract (via
[`KeyedExternalAdapter`](../src/membench/retrieval/external/_base.py)), so it competes
on the exact same tasks, budget, and model as every other arm. Two design facts
matter for reading "status":

- **None of the external packages is in the project lockfile** — every backend pins
  mutually-incompatible dependencies, so each adapter **lazy-imports** its backend and
  raises an actionable error pointing at an isolated env file under `competitors/envs/`.
  An adapter is therefore "wired but dormant" until its env is installed.
- **Every adapter accepts an injected client**, which is how its mapping logic is
  unit-tested without the heavyweight (often networked) real backend.

Each adapter registers an *arm* via `@register_arm(...)` with capability flags
(`follows_links`, `uses_embeddings`, `requires_network`, `deterministic`).

---

## Mem0

**Positioning.** Mem0 (`mem0ai`) is a flagship LLM-extraction memory product: it
extracts and consolidates salient facts from ingested text with an LLM and retrieves
them by semantic search. It is a direct competitor to Brief's context layer on the
**similarity axis** (extract-then-embed-then-search), not the typed-link axis.

**Adapter status.** [`mem0_adapter.py`](../src/membench/retrieval/external/mem0_adapter.py)
— arm `mem0`; `follows_links=False`, `uses_embeddings=True`, `requires_network=True`,
`deterministic=False`. Wired and unit-tested via injected client; the real path
lazy-imports `mem0` and constructs an isolated per-instance Qdrant store
(`competitors/envs/mem0.txt`).

**Vendor-tier published numbers** (verbatim, with citation):

| metric (as published) | score | vs baseline | tier / source |
|---|---|---|---|
| LoCoMo LLM-judge score | 66.9 | 52.9 (OpenAI memory) | vendor / peer-reviewed — arXiv:2504.19413 |
| LoCoMo lift over baseline | 14.0 | — | vendor — arXiv:2504.19413 |
| LoCoMo (2026 algo) | 92.5 | — | vendor-reported — [mem0.ai blog](https://mem0.ai/blog/ai-memory-benchmarks-in-2026) |
| LongMemEval (2026 algo) | 94.4 | — | vendor-reported — [mem0.ai blog](https://mem0.ai/blog/ai-memory-benchmarks-in-2026) |

**Modeled (NOT measured):** Mem0 modeled compliance on our scale = **0.22**
("modeled from published lift over baseline; not a measured run", `METRICS.md` §10).

---

## Zep

**Positioning.** Zep (engine: Graphiti) is a temporal knowledge-graph memory: it
builds a temporally-aware knowledge graph from ingested episodes and retrieves via
hybrid graph + semantic search. It is the closest **graph-based** competitor to
Brief, which makes it the key comparison for the central question — does a *generic*
knowledge graph close the depth gap, or is Brief's *typed, decision-centric* graph
what matters?

**Adapter status.** [`zep_adapter.py`](../src/membench/retrieval/external/zep_adapter.py)
— arm `zep`; `follows_links=True` (graph-based, follows edges), `uses_embeddings=True`,
`requires_network=True`, `deterministic=False`. Wired and unit-tested via injected
client; the real path drives Graphiti's async API via `asyncio.run` and needs
`graphiti-core` + a Neo4j backend (`competitors/envs/zep.txt`).

**Vendor-tier published numbers** (verbatim, with citation):

| metric (as published) | score | vs baseline | tier / source |
|---|---|---|---|
| Deep Memory Retrieval (GPT-4 Turbo) | 94.8 | 93.4 (MemGPT/Letta) | vendor / peer-reviewed — arXiv:2501.13956 |
| DMR lift over baseline | 18.5 | — | vendor — arXiv:2501.13956 (`METRICS.md` §10) |
| LongMemEval accuracy lift (Δ%) | 18.5 | 0.0 (full-context) | vendor-reported — [getzep.com blog](https://blog.getzep.com/state-of-the-art-agent-memory/) |

**Modeled (NOT measured):** Zep modeled compliance on our scale = **0.265**
("modeled from published lift over baseline; not a measured run", `METRICS.md` §10).

---

## GraphRAG

**Positioning.** Microsoft GraphRAG indexes a corpus into an entity/community
knowledge graph and answers queries with graph-aware local/global search. It is a
second graph-based competitor, but built on **community detection** rather than typed
decision links — so it tests the same question as Zep from a different angle: does a
generic graph close the depth gap?

**Adapter status.** [`graphrag_adapter.py`](../src/membench/retrieval/external/graphrag_adapter.py)
— arm `graphrag`; `follows_links=True`, `uses_embeddings=True`, `requires_network=True`,
`deterministic=False`. **Partially wired**: the adapter mapping is unit-tested via an
injected client and the real wrapper accumulates documents on `add`, but live
`search` is **not fully operational** — it raises `RuntimeError("GraphRAG live search
requires a built index and LLM configuration")` because GraphRAG's programmatic
surface needs a pre-built index plus LLM config (`competitors/envs/graphrag.txt`).

**Vendor-tier published numbers** (verbatim, with citation):

| metric (as published) | score | vs baseline | tier / source |
|---|---|---|---|
| Comprehensiveness win-rate vs vector RAG (QFS) | 77.5 | 27.0 (vector RAG) | vendor / peer-reviewed — arXiv:2404.16130 |
| Comprehensiveness lift over baseline | 50.0 | — | vendor — arXiv:2404.16130 (`METRICS.md` §10) |

**Modeled (NOT measured):** GraphRAG modeled compliance on our scale = **0.58**
("modeled from published lift over baseline; not a measured run", `METRICS.md` §10).

---

## Supermemory

**Positioning.** Supermemory is a hosted "universal memory" API: it ingests content
and serves it back to agents over an API by semantic search — positioned, like Brief,
as a **context layer for AI applications**, but similarity-based rather than typed.

**Adapter status.** [`supermemory_adapter.py`](../src/membench/retrieval/external/supermemory_adapter.py)
— arm `supermemory`; `follows_links=False`, `uses_embeddings=True`,
`requires_network=True`, `deterministic=False`. Wired and unit-tested via injected
client; the real path needs the `supermemory` SDK plus an API key
(`SUPERMEMORY_API_KEY`, `competitors/envs/supermemory.txt`).

**Vendor-tier published numbers** (verbatim, with citation) — all **self-reported
(unverified)** per third-party coverage:

| metric (as published) | score | vs baseline | tier / source |
|---|---|---|---|
| LoCoMo P@1 | 59.7 | 34.4 (competing providers) | vendor / self-reported (unverified) — [supermemory.ai](https://supermemory.ai/) |
| LongMemEval-S overall | 85.4 | — | vendor / self-reported (unverified) — [supermemory.ai](https://supermemory.ai/) |
| SWE-Context-Bench task resolution | 30.3 | — | vendor / self-reported (unverified) — arXiv:2602.08316 |

**Modeled (NOT measured):** Supermemory modeled compliance on our scale = **0.333**
("modeled from published lift over baseline; not a measured run", `METRICS.md` §10).

---

## Letta (formerly MemGPT)

**Positioning.** Letta gives an agent self-editing core memory plus a searchable
**archival** memory. The adapter uses the archival store as the benchmark memory:
insert each item, retrieve via archival search. It represents the "agent-managed
memory" school rather than a retrieval index.

**Adapter status.** [`letta_adapter.py`](../src/membench/retrieval/external/letta_adapter.py)
— arm `letta`; `follows_links=False`, `uses_embeddings=True`, `requires_network=True`,
`deterministic=False`. Wired and unit-tested via injected client; the real path needs
`letta-client` plus a running Letta server (`competitors/envs/letta.txt`).

**Vendor-tier published numbers** (verbatim, with citation):

| metric (as published) | score | tier / source |
|---|---|---|
| Deep Memory Retrieval (DMR) | 93.4 | vendor / peer-reviewed — arXiv:2310.08560 |
| LoCoMo (filesystem, gpt-4o-mini) | 74.0 | vendor-reported — [letta.com blog](https://www.letta.com/blog/benchmarking-ai-agent-memory/) |

Note: Letta/MemGPT's published DMR of **93.4** is also the baseline that Zep's
**94.8** is measured against (arXiv:2501.13956). No modeled-compliance row is
recorded for Letta in `METRICS.md` §10.

---

## LangChain

**Positioning.** The canonical "roll your own RAG" stack: a LangChain `FAISS` vector
store over the corpus with similarity search. It represents the **do-it-yourself
baseline** a team reaches for before buying a memory product — deterministic and
local given a local embedding.

**Adapter status.** [`langchain_adapter.py`](../src/membench/retrieval/external/langchain_adapter.py)
— arm `langchain_vec`; `follows_links=False`, `uses_embeddings=True`,
`requires_network=False`, `deterministic=True` (the only deterministic, offline
external arm — local FAISS with `FakeEmbeddings`). Wired and unit-tested via injected
client; the real path needs `langchain-community` + `faiss-cpu`
(`competitors/envs/langchain.txt`).

**Vendor-tier published numbers.** None. LangChain is a DIY framework baseline, not a
benchmarked product, so it has no vendor headline figure to transcribe — and none is
recorded in `competitors/registry.yaml`, `results/competitor_landscape.md`, or
`METRICS.md` §10. No number is invented here.

---

## Cognee

**Positioning.** Cognee is an LLM "memory engine" that ingests text, "cognifies" it
into a graph + vector store, and answers queries with graph-aware search. Another
graph-flavoured competitor in the same architectural class as Zep / GraphRAG.

**Adapter status.** [`cognee_adapter.py`](../src/membench/retrieval/external/cognee_adapter.py)
— arm `cognee`; `follows_links=True`, `uses_embeddings=True`, `requires_network=True`,
`deterministic=False`. **Partially wired**: the adapter mapping is unit-tested via an
injected client; the real wrapper drives Cognee's async `add`/`cognify`/`search` via
`asyncio.run`, but live `search` is not exercised in tests (it needs the `cognee`
package + an LLM key, `competitors/envs/cognee.txt`).

**Vendor-tier published numbers** (verbatim, with citation):

| metric (as published) | score | tier / source |
|---|---|---|
| HotpotQA multi-hop | graph > vector (qualitative) | vendor-reported — [cognee.ai](https://www.cognee.ai/research-and-evaluation-results) |

Cognee publishes a **qualitative** "graph beats vector" result on HotpotQA rather
than a single headline number; no numeric value is recorded in the repo result files,
so none is fabricated here.

---

## Brief — for contrast (measured, Tier-1)

Brief's results are produced **inside our own harness** and are kept separate from
the vendor tier above. From `results/competitor_landscape.md`,
`results/positioning.md`, and `results/METRICS.md` §10:

- **measured** — agent **+ Brief** vs agent **alone** = **0.70 vs 0.08** compliance
  (all data, Claude + GPT-5.1); `Brief_compliance_ours = 0.7037` (measured, Claude
  all-data, `METRICS.md` §10).
- **measured** — synthetic depth-3 crossover: **Brief 1.00 vs best similarity 0.70**
  (P(Brief>bm25) = 1.000).
- **vendor-reported** (Brief's own public RoT study, `briefhq.ai`): agent decision
  compliance **95.0** vs **46.0** (codebase alone); merge-ready task rate **100.0**
  vs **25.0** (codebase alone).
- **paper-reported** (kept distinct from the repo-measured figures above): pooled
  real-code (dcbench+swebench, Claude) recall **0.667**, compliance **0.469**, use
  factor **κ=0.703** — the only arm clearing the 0.56–0.64 similarity use-factor band
  (`depth_not_length_FIXED.tex`, `tab:realret`).

These Brief numbers are reported here only to anchor the contrast; the vendor numbers
above remain published-on-their-own-benchmark and are not a controlled head-to-head
against them.

---

## Sources

- [`competitors/registry.yaml`](../competitors/registry.yaml) — Tier-2 vendor claims
  registry (`needs_citation: true` entries are placeholders pending verification).
- [`results/competitor_landscape.md`](../results/competitor_landscape.md) /
  [`results/data/competitor_landscape.csv`](../results/data/competitor_landscape.csv)
  — rendered vendor landscape with statuses (peer-reviewed / vendor-reported /
  self-reported).
- [`results/METRICS.md`](../results/METRICS.md) §10 `10_competitor_head_to_head` —
  vendor / modeled / measured tiers per system.
- [`results/positioning.md`](../results/positioning.md) — two-axis positioning.
- [`src/membench/retrieval/external/`](../src/membench/retrieval/external/) — the
  adapters described above.
- `depth_not_length_FIXED.tex` (`tab:landscape`, `tab:stdbench`, `tab:realret`) —
  paper-reported figures, cross-checked: Mem0 66.9 (arXiv:2504.19413), Zep 94.8
  (arXiv:2501.13956), GraphRAG 77.5 (arXiv:2404.16130).
