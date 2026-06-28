# Frequently Asked Questions

This FAQ answers the questions a reader or reviewer is most likely to have about
BriefBench (`membench`) and the paper *Depth, Not Length*. Every number below is copied
from a committed source and labelled by **provenance**:

- **repo-measured** — produced by this harness under identical conditions across arms
  (the fairness lock). Cited from [`results/`](../results/) and
  [`paper/measured/`](../paper/measured/).
- **paper-reported** — figures stated in the paper. Cited as paper-reported and never
  presented as if they were re-measured here.
- **vendor-reported** — published numbers from other systems on their own benchmarks,
  kept in a separate tier (see [`results/competitor_landscape.md`](../results/competitor_landscape.md)).

Repo-measured and paper-reported numbers are kept clearly separate, because they were
produced under different conditions.

---

## 1. Why "depth, not length"?

Because the failure mode this project measures is **causal depth**, not context-window
length. As the [README](../README.md) and [methodology](METHODOLOGY.md) put it: coding
agents fail "not from token scarcity but because similarity retrieval cannot recover a
decision that is causally linked to the task yet phrased unlike it and sitting several
hops away."

Depth is defined operationally as a *strip-count* ([`docs/METHODOLOGY.md`](METHODOLOGY.md)):
`d=1` states the governing constraint in the query, `d=2` strips it into one corpus node,
and `d=3` splits it into a constraint plus a justification node joined by a `constrains`
link. Similarity retrieval's chance of recovering the governing decision decays as depth
grows, while a store that follows typed links pays only a steady per-hop cost — so the
interesting variable is how many hops away the decision is, not how many tokens you can
afford to keep.

## 2. Is this just marketing for Brief?

No. The crossover is a property of **any link-following mechanism**, not of Brief
specifically. The [README](../README.md) states this directly: "This holds for **any**
link-following system — it is a property of the mechanism, not of Brief specifically."

The theory is stated abstractly (see [`paper/CLAIMS.md`](../paper/CLAIMS.md)): similarity
recovery has a depth ceiling that decays super-geometrically, structured recovery decays
gently, and a crossover depth `d⋆` therefore exists. Brief's typed decision graph is the
concrete instance that is benchmarked, but the same argument applies to any store with
governance edges to traverse. The harness is also built to keep the comparison honest
rather than flattering (see Q3, Q5, and Q8).

## 3. How is fairness guaranteed?

By a **fairness lock**: identical model, identical retrieval budget, and identical
code-search tool across every arm, so memory architecture is the only variable
([README, "Integrity"](../README.md); [`paper/measured/README.md`](../paper/measured/README.md)).

Concretely:

- The budget is keyed by **dataset, never by arm** — every arm sees the same budget for a
  given dataset ([`configs/default.yaml`](../configs/default.yaml), which sets
  `budget_tokens_by_dataset` and `seed: 0`).
- Every arm implements one contract — `write` / `retrieve(query, budget)` — and the
  harness swaps the arm while holding everything else fixed
  ([`docs/METHODOLOGY.md`](METHODOLOGY.md)).
- Headline metrics are additionally cross-checked by independent re-implementations in
  C#, Java, R, SQL, and C/Rust/CUDA; if a reference disagrees with the Python on the same
  inputs, CI fails ([`docs/VALIDATION.md`](VALIDATION.md)).

## 4. What runs offline, and what needs API keys?

**Everything in the core pipeline runs offline and deterministically.** The README states
the default path uses "a stub model + a hashing embedding," and the core install is
"deliberately light and offline-capable" ([`pyproject.toml`](../pyproject.toml)). The
reproduction script even runs the test gate as `pytest -m "not live and not neural"`
([`scripts/reproduce.sh`](../scripts/reproduce.sh)), and [`configs/default.yaml`](../configs/default.yaml)
sets `offline: true`.

Real backends live behind **optional extras** ([`pyproject.toml`](../pyproject.toml)):

| Extra | Pulls in | Needs a key? |
|---|---|---|
| `anthropic` | `anthropic` SDK | `ANTHROPIC_API_KEY` (the main model) |
| `openai` | `openai` SDK | `OPENAI_API_KEY` |
| `neural` | `sentence-transformers` (+ torch) for dense embeddings / cross-encoder rerankers | no |
| `viz` | `matplotlib`, `seaborn` for figures | no |
| `mcp` | `mcp` for the Brief MCP arm | `BRIEF_OAUTH_TOKEN` |
| `dev` | pytest, ruff, mypy, etc. | no |

[`env.example`](../env.example) lists `ANTHROPIC_API_KEY` and `BRIEF_OAUTH_TOKEN` as the
keys you set for real runs; the brief arm is "automatically skipped with a notice" if its
token is absent. `OPENAI_API_KEY` and `TOGETHER_API_KEY` are marked **not yet
implemented** there.

Third-party memory systems (Mem0, Zep/Graphiti, GraphRAG, Letta, Cognee, LangChain) are
deliberately *not* in the extras: several pin mutually-incompatible transitive deps, so
each runs in its own isolated environment under
[`competitors/envs/`](../competitors/envs/), and the adapters lazy-import and raise an
actionable error if the package is missing ([`pyproject.toml`](../pyproject.toml)).

## 5. How are competitor numbers handled?

With a **two-tier provenance rule**: numbers measured under identical conditions (Tier-1)
are never mixed with vendor-reported claims (Tier-2)
([README, "Integrity"](../README.md); [`paper/measured/README.md`](../paper/measured/README.md):
"measured and vendor-reported numbers never share a table").

- **Tier-2 vendor-reported** numbers are kept in
  [`results/competitor_landscape.md`](../results/competitor_landscape.md), each with its
  source, benchmark, and status (peer-reviewed / vendor-reported / self-reported). That
  file is explicit that these are "published numbers on each system's OWN benchmark — NOT
  a controlled head-to-head and not directly comparable cell-to-cell."
- **Modeled** numbers are labelled as such. For SWE-ContextBench,
  [`results/swe_context_bench.md`](../results/swe_context_bench.md) states that Brief "has
  not been run on the official harness" and reports a **modeled** resolution of **27%
  (range 24–31)** — placed mid-cluster, below the published Oracle Summary Learning upper
  bound of 34.34% and Supermemory's self-reported 30.30% — not a measured benchmark
  result.

For completeness, note a provenance difference a reviewer will spot: the paper digest's
unified-harness table reports a **paper-reported** SWE-ContextBench resolution of **47.3%**
for Brief, whereas the committed [`results/swe_context_bench.md`](../results/swe_context_bench.md)
carries only the more conservative **modeled 27%** with the "not run on the official
harness" caveat. The repo's modeled figure is the one backed by a committed file; the
47.3% is paper-reported.

## 6. What does "use factor" / κ (kappa) mean?

It is the second term in the **compliance factorization** from the theory
([`paper/CLAIMS.md`](../paper/CLAIMS.md)):

```
P_comply(d) = P_ret(d) · κ(d)
```

`P_ret(d)` is the probability the governing decision is *retrieved* at depth `d`; **κ
(the "use factor")** is the conditional probability the agent actually *complies* given
that the decision was retrieved — i.e. how much retrieval converts into honored
decisions. A "use ceiling" is the observation that κ for similarity retrievers tops out
around 0.6 on real code.

Paper-reported values (pooled real-code, from the paper's `tab:realret`): the typed store
clears the ceiling with **κ = 0.703**, the only arm above the **0.56–0.64** band that
ceilings every similarity arm. These are **paper-reported**; treat them as such.

## 7. Does Brief always win?

**No — and the losses are reported, not buried.** [`results/EVALS.md`](../results/EVALS.md)
covers **440 evals** across 9 families and states up front that "cuts where Brief does not
win are included (honest)": **Brief wins 312/440 head-to-head cuts** (repo-measured), which
means it loses or ties the remaining 128.

Specific honest losing/tying cuts, all **repo-measured** from
[`results/EVALS.md`](../results/EVALS.md):

- **dcbench, depth 2, compliance, Brief vs best similarity** — Brief `0.2619` vs `0.3929`
  (Δ `-0.131`): Brief loses.
- **swebench, depth 3, compliance, Brief vs best similarity** — Brief `0.25` vs `0.5`
  (Δ `-0.25`): Brief loses (note this `d=3` swebench cell has only `n=12` and is
  under-powered).
- **dcbench, all depths, compliance, Brief vs tfidf** — Brief `0.3571` vs `0.4206`
  (Δ `-0.0635`): Brief loses.

This matches the "honest boundary" in [`paper/measured/summary.md`](../paper/measured/summary.md):
as a *raw retriever* on the two natural code datasets, Brief is competitive but does not
beat the best similarity baseline — **dcbench 0.36 vs 0.42** and **swebench 0.33 vs 0.37**
(repo-measured). The paper's own scorecard reflects the same posture (paper-reported:
**22 wins / 14 near-ties / 5 losses** over 41 axes).

## 8. Where Brief *does* win, what are the headline numbers?

Two clean, **repo-measured** results from [`paper/measured/summary.md`](../paper/measured/summary.md):

- **Product Navigator** (agent + Brief vs agent alone, all data): **0.70 vs 0.08**
  compliance.
- **Depth crossover** at `d=3` on the controlled synthetic suite: **Brief 1.00 vs best
  similarity 0.70**; the four-arm ablation shows **Brief 0.31 vs random control 0.17** on
  dcbench `d=3` (structure beats budget).

Paper-reported headline numbers (label them as such): on pooled real code the typed store
ranks first on recall (**0.667**) and compliance (**0.469**); on unified public
benchmarks it reports LoCoMo **87.6**, DMR fact-F1 **94.2**, and (see Q5 on provenance)
SWE-ContextBench **47.3%**. These come from the paper digest, not from re-running the
harness here.

## 9. How do I reproduce a figure?

The figures are regenerated from the scored rows. The end-to-end path
([README](../README.md), [`scripts/reproduce.sh`](../scripts/reproduce.sh)) is:

```sh
uv sync --extra dev --extra viz          # viz extra pulls matplotlib/seaborn
uv run membench run     --dataset synthetic --budget 150 --out results/rows.jsonl
uv run membench figures --src results/rows.jsonl --out-dir figures
```

`scripts/reproduce.sh` runs the whole chain (sync → build → offline test gate → run →
tables/report/figures) in one command. The catalogue of what gets produced — 43
visualizations with their captions — is [`results/FIGURES.md`](../results/FIGURES.md);
the claim → code → artifact map is [`paper/CLAIMS.md`](../paper/CLAIMS.md).

## 10. What models were evaluated?

**Repo-measured:** the headline runs use **`claude-sonnet-4-6`** via the real Anthropic
Messages API — **1,824 attempts** (8 arms × 3 datasets × 3 depths, n=40), total measured
spend ≈ **$33.12** ([`paper/measured/summary.md`](../paper/measured/summary.md),
[`paper/measured/README.md`](../paper/measured/README.md)). A second model, **GPT-5.1**, is
used for the cross-model synthetic replication; both appear in
[`results/EVALS.md`](../results/EVALS.md) ("on Claude Sonnet + GPT-5.1") and in the
committed rows under [`results/data/`](../results/data/) (`all_rows_claude.jsonl`,
`all_rows_gpt.jsonl`). [`configs/default.yaml`](../configs/default.yaml) defaults to
`models: [claude]`.

**Paper-reported:** the token-economics sweep additionally spans 12 backend models
(including GPT-5.5, GPT-5.3 Codex, Claude Opus 4.8, Mistral Large 3, Gemini 3.5 Flash,
Composer 2.5, and Kimi K2.5). Treat those as paper-reported. The paper also notes the
friendly model names are not reproducible identifiers; exact snapshot strings are promised
for the camera-ready.

## 11. What are the datasets, and how big are they?

Three datasets (paper-reported sizes from the paper digest's `tab:datastats`; the same
splits drive the repo runs):

| dataset | tasks | per depth (d1/d2/d3) | origin |
|---|---|---|---|
| synthetic | 120 | 40 / 40 / 40 | authored, controlled (drift ρ≈0.67) |
| dcbench | 42 | 14 / 14 / 14 | purpose-built repo, 41 weighted decision points |
| swebench | 54 | 21 / 21 / 12 | real SWE-bench GitHub issues |

The synthetic suite isolates the depth mechanism; dcbench and swebench supply ecological
evidence. The swebench `d=3` cell (`n=12`) is under-powered and should be read as
inconclusive ([`docs/THREATS.md`](THREATS.md)).

## 12. What are the known threats to validity?

They are listed so reviewers don't have to find them ([`docs/THREATS.md`](THREATS.md)):

- **Depth labels are single-annotator** (an automated one-pass heuristic); the
  depth-crossover table should clear a two-annotator pass before publication.
- **Similarity arms stand in for their class** — BM25/TF-IDF are exact, but the dense arm
  uses a deterministic hashing embedding offline (a real neural backend is behind the
  `neural` extra).
- **Merge-ready correctness is a proxy** over a stub agent, not a real diff+test harness.
- **The synthetic benchmark is controlled, not natural** — the natural datasets carry the
  ecological evidence, "reported honestly (where the effect is modest, we say so)."
- **Cross-model robustness is partial** — GPT/open-weight backends are wired but the live
  robustness rows need their API keys.

## 13. What is the central scope caveat?

The harness supplies a common, pre-extracted decision corpus, so it measures the
**retrieval and use of typed links — not extraction/capture.** The paper digest is
explicit: decision recall falls from 1.00 to 0.42 when the typed links are stripped, and
the capture factor (≈0.30 → 1.00) is out of scope. In other words, these results isolate
the *mechanism* (link-following vs. similarity); they are not an end-to-end product claim.

## 14. Where do I look next?

- [`README.md`](../README.md) — the result in one figure, plus repo layout.
- [`docs/METHODOLOGY.md`](METHODOLOGY.md) — design, depth definition, theory fit.
- [`docs/THREATS.md`](THREATS.md) — threats to validity.
- [`docs/VALIDATION.md`](VALIDATION.md) — cross-language validation.
- [`results/EVALS.md`](../results/EVALS.md) / [`results/METRICS.md`](../results/METRICS.md) /
  [`results/FIGURES.md`](../results/FIGURES.md) — the measured matrices and figure index.
- [`paper/CLAIMS.md`](../paper/CLAIMS.md) — claim → code → artifact map.
- [`paper/measured/`](../paper/measured/) — durable real-backend artifacts and the headline
  summary.
