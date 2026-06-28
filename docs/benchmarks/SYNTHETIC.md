# The Synthetic Depth-Crossover Suite

A deep-dive on the controllable synthetic benchmark that isolates the core thesis of
*Depth, Not Length*: that similarity retrieval degrades super-geometrically in causal-hop
distance while a link-following store pays only a steady per-hop cost. Unlike the real-code
datasets (dcbench, swebench), every variable here is dialed by us, so the depth mechanism is
the only thing that moves.

Source of truth for this document:

- Generator: [`src/membench/datasets/synthetic.py`](../../src/membench/datasets/synthetic.py)
- Depth-label reliability: [`src/membench/datasets/depth_labeling.py`](../../src/membench/datasets/depth_labeling.py)
- Run configuration: [`configs/default.yaml`](../../configs/default.yaml) and
  [`src/membench/harness.py`](../../src/membench/harness.py)
- Repo-measured numbers: [`results/METRICS.md`](../../results/METRICS.md) and
  [`README.md`](../../README.md)

All benchmark numbers below are copied verbatim from those files (repo-measured) or, where
explicitly labelled "paper-reported", from the paper. Repo-measured and paper-reported figures
are kept in separate tables and never presented as identical — they come from different runs and
do not match cell-for-cell.

---

## 1. What a depth-`d` item is

Each task models how a real codebase hides a governing decision. The agent is handed a feature
request that names an *area* — for example *"add a CSV export button to the data export module so
users can download their data"* — and the rule that actually governs the edit
(*"every operation in this path must call `withAuditLog()` before proceeding"*) sits `d` reasoning
hops away, phrased in unrelated vocabulary and discoverable only because it is *linked* to the area
through derived policy.

The ten engineering areas are defined as `Area` records in
[`synthetic.py`](../../src/membench/datasets/synthetic.py) (`AREAS`): `export`, `auth`, `billing`,
`search`, `uploads`, `notifications`, `caching`, `ratelimit`, `reporting`, `flags`. Each carries a
feature, an action, a code-identifier `guard` (the compliance target — e.g. `withAuditLog`,
`requireMfa`, `withIdempotencyKey`, `sanitizeQuery`, `scanForMalware`), and a tuple of
intermediate-policy vocabulary used to populate the chain.

### The chain construction (`_chain`)

For a target area at depth `d`, `_chain(area, depth)` builds a path of `d + 1` nodes:

```
{slug}-entry  ->  {slug}-policy1  ->  ...  ->  {slug}-policy(d-1)  ->  {slug}-rule
```

- **Node 0 (`entry`, `node_type="entry"`)** carries the full feature vocabulary — it looks like the
  query, so similarity retrieval ranks it first.
- **Intermediate nodes (`node_type="policy"`)** are "policy layer, refining an upstream constraint
  it derives from" — query-relevant enough to crowd the context window, but not the governing rule.
- **Node `d` (`node_type="governing"`)** is the terminal `{slug}-rule` node that states
  `RULE: every operation in this path must call {guard}() before proceeding.` and is registered as
  the task's `governing_decisions`.

Edges are typed and directed: every node links to its successor with relation `derived_from`,
**except the final edge into the governing rule, which is `constrains`**. This is the chain a graph
arm traverses; similarity arms cannot see it. A depth-1 task is therefore a single
`entry --constrains--> rule` edge; a depth-3 task interposes two `policy` nodes.

### Why similarity decays along the chain (`_node_text`)

The keyword overlap with the query is engineered to fall off geometrically toward the governing
node. `_node_text` selects a fraction `(depth - hop + 1) / (depth + 1)` of the area's query keywords
at each hop, so the entry node gets nearly all of them and the rule node gets the floor (one keyword,
or none). The module docstring states this directly: it instantiates the theory's similarity-decay
model `s_i = s_0 * rho^i`. The deeper the rule, the lower its similarity score, the further it is
pushed down the ranking under a fixed token budget — exactly the failure the benchmark is built to
expose.

### The haystack

For each task, `load_synthetic_tasks` plants the target chain, then adds `distractor_areas`
*other* areas' chains (query-irrelevant at the same depth) plus six fixed `_FILLER` nodes
("telemetry pipeline ingestion buffer", "design system token palette spacing", …). The whole corpus
is shuffled with a seeded RNG. Within a depth, the target area rotates with the task index
(`AREAS[(i + depth) % len(AREAS)]`) so no single area drives the result.

---

## 2. The controllable knobs

The generator's public entry point is `load_synthetic_tasks(...)` in
[`synthetic.py`](../../src/membench/datasets/synthetic.py):

| knob | parameter | code default | meaning |
|---|---|---|---|
| Depth grid | `depths` | `(1, 2, 3)` | which causal-hop distances to generate; the x-axis of the crossover |
| Tasks per depth | `per_depth` | `15` (generator) | how many tasks at each depth (target area rotates) |
| Seed | `seed` | `0` | deterministic corpus shuffle and distractor selection |
| Distractor breadth | `distractor_areas` | `6` | how many *other* area chains pad the haystack |
| Similarity decay | (internal `_node_text`) | `(depth-hop+1)/(depth+1)` | per-hop keyword-overlap fraction, the `s_i = s_0 rho^i` model |

Two harness-level knobs complete the run, from
[`configs/default.yaml`](../../configs/default.yaml) (the "fairness lock" — budget is keyed by
dataset, never by arm, so every arm sees the same budget):

| knob | config key | value | meaning |
|---|---|---|---|
| Token budget | `budget_tokens_by_dataset.synthetic` | `150` | retrieval budget every arm shares on synthetic |
| Tasks per depth | `per_depth` | `10` | per-depth count the default run uses (overrides the generator default) |
| Seed | `seed` | `0` | run seed |
| Arms | `arms` | `none, bm25, tfidf, dense, hybrid_rrf, rerank_ce, raptor, brief_graph_3hop` | the eight evaluated arms |
| Retry | `retry_policy` | `single_shot` | one-shot retrieval, no re-query |
| Offline | `offline` | `true` | deterministic stub model + hashing embedding |

> Note on counts. The code defaults produce a small smoke run (`per_depth=10` in the harness,
> `15` in the generator signature). The paper's evaluated suite is larger: the paper-reported
> dataset-statistics table lists the synthetic suite as **120 tasks, 40 per depth (d1/d2/d3 =
> 40/40/40), authored, drift ρ≈0.67** (paper-reported). The repo-measured `d=all` aggregate in
> [`results/METRICS.md`](../../results/METRICS.md) is consistent with 40 tasks per depth
> (Brief compliance `0.9917` ≈ 119/120).

---

## 3. Why this isolates the depth mechanism

Three properties make the suite an *independent* test of the mechanism rather than a Brief-tuned
demo (stated in the generator docstring and the repo [`README.md`](../../README.md)):

1. **Depth is the only dial that moves.** Every arm sees an identical task set, identical corpus,
   and an identical token budget (the fairness lock in `configs/default.yaml`); the only thing that
   changes across the x-axis is how many hops separate the query-similar entry from the governing
   rule.
2. **The similarity decay is engineered, not incidental.** `_node_text` bakes in
   `s_i = s_0 * rho^i`, so the corpus realises the theory's per-hop survival model directly — the
   benchmark and the theory share one parameterisation.
3. **It is mechanism-specific, not vendor-specific.** The win is a property of *link-following*: any
   system that traverses the typed `derived_from`/`constrains` edges reaches the rule regardless of
   depth. The README states it plainly: "This holds for **any** link-following system — it is a
   property of the mechanism, not of Brief specifically."

Depth-label reliability is handled by
[`depth_labeling.py`](../../src/membench/datasets/depth_labeling.py): a dual-annotation protocol
with linear-weighted Cohen's κ. Two annotators trace each chain; tasks disagreeing by more than one
hop are dropped; survivors are certified to the *lower* of the two labels. For the synthetic suite
depth is generated, not labelled, so this is exact by construction; the module exists so the same
reliability gate can be applied to the real-code datasets where depth is annotated.

---

## 4. Repo-measured synthetic depth-crossover numbers (verbatim)

The following compliance-by-depth values are copied verbatim from the `4_depth_robustness` bucket
of [`results/METRICS.md`](../../results/METRICS.md). Provenance per cell is `measured`, model
`claude`, dataset `synthetic`.

**Compliance by arm and depth (repo-measured, `results/METRICS.md`):**

| arm | d=1 | d=2 | d=3 |
|---|---|---|---|
| Brief | 0.975 | 1.0 | 1.0 |
| dense (best baseline) | 0.925 | 0.875 | 0.675 |
| BM25 | 0.975 | 1.0 | 0.7 |
| none (no context) | 0.0 | 0.0 | 0.0 |

(The `random_context` cells for these rows are blank in `results/METRICS.md` and are therefore
omitted here.)

**Synthetic aggregate (`d=all`), repo-measured, `results/METRICS.md`:**

| metric | Brief | dense (best baseline) | BM25 | none |
|---|---|---|---|---|
| compliance@synthetic | 0.9917 | 0.825 | 0.8917 | 0.0 |
| recall@synthetic | 1.0 | 0.8417 | 0.9 | 0.0 |

Reading: the `none` arm sits at `0.0` compliance at every depth — the irreducible context-free
floor (the correct action is not a function of the task surface alone). Brief holds `1.0` at d=2 and
d=3, while the strongest similarity baseline (dense) falls to `0.675` at d=3 and BM25 to `0.7`. The
collapse is what the structure escapes.

### README headline table (repo-measured, `README.md`)

The repo [`README.md`](../../README.md) reports the result as *governing-decision recovery vs.
causal depth* (recall), measured offline and deterministic, copied verbatim:

| arm | d=1 | d=2 | d=3 |
|---|---|---|---|
| BM25 / TF-IDF / dense / hybrid | 0.95–1.00 | 0.85–1.00 | 0.35–0.70 |
| RAPTOR (hierarchical) | 0.55 | 0.25 | 0.05 |
| **Brief graph (link-following)** | **1.00** | **1.00** | **1.00** |

These are recovery (recall) ranges across the similarity arms, not the per-arm compliance cells in
the table above; the README headline and the `METRICS.md` compliance grid are different metrics and
should not be read as the same numbers.

---

## 5. Paper-reported numbers (kept separate)

The paper reports its own Claude synthetic compliance-by-depth table (paper-reported; transcribed
from the verified paper digest, `tab:slope`). **These differ from the repo-measured cells in §4 —
they come from a different run and are reproduced here only so the two sources are not conflated.**

| arm | d=1 | d=2 | d=3 | slope = comply(d3)−comply(d1) |
|---|---|---|---|---|
| Brief | 0.950 | 0.975 | 0.875 | −0.075 |
| bm25 | 0.925 | 0.925 | 0.775 | −0.150 |
| tfidf | 0.925 | 0.925 | 0.800 | −0.125 |
| dense | 0.925 | 0.900 | 0.750 | −0.175 |
| hybrid_rrf | 0.925 | 0.950 | 0.775 | −0.150 |
| rerank_ce | 0.925 | 0.900 | 0.725 | −0.200 |
| Δ(Brief − best-sim) | +0.025 | +0.025 | +0.075 | |

Paper-reported reading: Brief shows the flattest depth slope (−0.075, the least decay); the typed
store's advantage over the best similarity arm widens with depth (+0.025 → +0.025 → +0.075). The
paper's pre-registered primary endpoint is synthetic compliance at d=3.

---

## 6. How `d⋆` (crossover near `d=2`) emerges

The crossover is where the structured arm's recovery overtakes similarity. The paper's calibrated
operating point (paper-reported) is `s_0 = 0.70`, `ρ = 0.67`, `q = 0.97`, and the structured
advantage is

```
g(d) = P_ret_struct(d) − P_ret_sim(d) = q^d − s_0^d · ρ^{d(d+1)/2}
```

- Similarity recovery is **super-geometric**: `P_ret_sim(d) = s_0^d · ρ^{d(d+1)/2}`. The
  `d(d+1)/2` exponent on `ρ` means each extra hop costs more than the last — the curve is concave
  and its slope heads to −∞.
- Structured (bounded-traversal) recovery is **steady per hop**: `P_ret_struct(d) = q^d`, with no
  corpus-size penalty.

At the calibrated point the gap is `g(1) ≈ 0.50` and `g(2) ≈ 0.79` (paper-reported). The crossover
depth `d⋆` is the smallest `d` at which structure wins by the chosen margin `τ`: `d⋆ = 1` for
`τ ≤ 0.50`, and `d⋆ = 2` for `0.50 < τ ≤ 0.79`. The closed-form crossover (smallest `d` with
`g > 0`) is `d⋆ = 1`; the **reported `d⋆ = 2` is calibrated post-hoc** and is the value the README
points at ("the crossover appears near `d=2`").

By `d = 3` the two mechanisms have separated decisively: traversal `q^3 ≈ 0.91` versus similarity
`s_0^3 · ρ^6 ≈ 0.031` — roughly a 29× gap (paper-reported, in-sample fit). The repo-measured grid in
§4 shows the same shape with measured cells: similarity compliance drops to `0.675`/`0.7` at d=3
while Brief holds `1.0`, and the README recall table shows the similarity band collapsing into the
`0.35–0.70` range at d=3 while the link-following arm stays at `1.00`.

In short: depth, not corpus length, is what breaks similarity retrieval, and a typed
link-following store is the thing that walks straight past it.
