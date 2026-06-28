# Provenance: the tiers that keep the benchmark honest

Every number this benchmark reports carries a **provenance tier** that says how it
was produced. The tier is not cosmetic: it decides which numbers may sit in the
same column, which may be compared head-to-head, and which must wear a method note
that forbids reading them as a measurement. The code enforces the load-bearing
half of this discipline; the data files carry the rest as a labelled convention.

This document explains the three tiers, the **fairness lock** that makes the
measured tier an apples-to-apples comparison, and the guard that prevents tiers
from being mixed in a single view.

## The three tiers

The metrics matrix in [`results/METRICS.md`](../results/METRICS.md) is built from
**387 cells across 10 buckets**, and it states the provenance split verbatim:

> 387 cells across 10 buckets. **Provenance per cell:** measured=349, modeled=13, vendor=25.

So of the 387 cells: **measured=349**, **modeled=13**, **vendor=25** (349 + 13 + 25 = 387).
Each cell in the matrix carries a `tier` column with exactly one of these labels.

### Tier 1 — `measured` (349 cells)

Numbers produced by **our own harness**, with every arm run under **identical
conditions**: the same tasks, the same model, the same per-call token budget, and
the same retrieval contract. These are the load-bearing, apples-to-apples results —
the only tier from which we draw a ranking claim.

The module docstring in
[`src/membench/competitors.py`](../src/membench/competitors.py) states it plainly:

> **Tier 1 — measured.** Produced by this harness, with every arm on the same
> tasks, budget, and model. These are the load-bearing, apples-to-apples results.

Measured cells are tagged in `results/METRICS.md` with a note recording the exact
slice they come from — e.g. `claude/synthetic/d=all`, `claude/dcbench/d=3`,
`claude all-data`. A few examples, copied verbatim from `results/METRICS.md`
(all `measured`):

| metric | system | value | tier |
|---|---|---|---|
| recall@all | Brief | 0.8441 | measured |
| compliance@all | Brief | 0.7037 | measured |
| compliance@all | dense (best baseline) | 0.6134 | measured |
| compliance@all | none (no context) | 0.0826 | measured |
| return_on_tokens@all | Brief | 0.5725 | measured |

These repo-measured figures are what the harness emits directly. They are **not**
the same as the paper-reported headline figures (see "Repo-measured vs.
paper-reported" below), and the two are never presented as identical.

### Tier 2 — `vendor` (25 cells)

Numbers a **competitor (or its paper) published under its own conditions** —
different datasets, different metrics, different harnesses. They are useful
orientation but are **not comparable** to our measured tier, so they live in
clearly labelled, separate columns, each carrying a **citation** and the original
conditions. From `src/membench/competitors.py`:

> **Tier 2 — vendor-reported.** Numbers a competitor (or its paper) published under
> *its own* conditions. Useful context, but not comparable to Tier 1, so they live
> in clearly labelled separate columns with a citation and the original conditions.

The vendor tier is loaded from
[`competitors/registry.yaml`](../competitors/registry.yaml), which is explicit that
these numbers are never fused with measured ones:

> This file holds numbers REPORTED BY VENDORS or their papers, under THEIR OWN
> conditions. They are NEVER mixed with numbers we measured under identical
> conditions (Tier 1); the analysis layer keeps the two tiers in separate, labelled
> columns. Every entry carries a source_url and the conditions under which the
> number was produced.

Each registry entry carries a `source_url`, the `conditions` under which the number
was produced, an `accessed` date, and a `needs_citation` flag. Entries with
`needs_citation: true` are placeholders pending a verified primary source and
**must not be published until verified** — `verified_claims()` in
`src/membench/competitors.py` filters them out.

The published competitor headlines that *are* carried (verbatim from the
`10_competitor_head_to_head` bucket of `results/METRICS.md`) are cited by arXiv id:

| metric | system | value | tier | citation |
|---|---|---|---|---|
| Mem0_headline_score | Mem0 | 66.9 | vendor | arXiv:2504.19413 |
| Zep_headline_score | Zep | 94.8 | vendor | arXiv:2501.13956 |
| GraphRAG_headline_score | GraphRAG | 77.5 | vendor | arXiv:2404.16130 |
| Supermemory_headline_score | Supermemory | 59.7 | vendor | supermemory.ai (unverified) |

(The Supermemory figure is marked `unverified` in the matrix — it is a self-reported
number without a verified primary source, kept distinct from the arXiv-cited ones.)

### Tier 3 — `modeled` (13 cells)

Numbers that are **derived**, not measured. Each modeled cell records the method in
its note and is **never shown as a measurement**. From `results/METRICS.md`:

> `modeled` = derived (method in the note; not a measurement). Modeled cells are
> clearly tagged and never shown as measured.

The 13 modeled cells fall into two groups, both verbatim from `results/METRICS.md`:

- **Competitor compliance re-expressed on our scale** (`10_competitor_head_to_head`),
  derived from each vendor's published lift over its own baseline — e.g.
  `Mem0_modeled_compliance_ours = 0.22`, `Zep_modeled_compliance_ours = 0.265`,
  `GraphRAG_modeled_compliance_ours = 0.58`, `Supermemory_modeled_compliance_ours = 0.333`,
  each noted `modeled from published lift over baseline; not a measured run`.
- **Latency and index-build estimates** (`8_latency_scale`), e.g.
  `retrieval_latency_ms_Brief = 120`, `retrieval_latency_ms_GraphRAG = 900`,
  `index_build_cost_rel_GraphRAG = 12.0`, each noted `modeled from architecture; not measured`.

This is why the system is described as **two-tier (really three-tier)**: the code's
hard guard (below) enforces the *measured vs. vendor* boundary, while `modeled` is a
third, data-level provenance label — clearly tagged in the matrix and excluded from
any measured comparison — so derived numbers can never be mistaken for harness output.

## The fairness lock

The measured tier is only meaningful if every arm faced the same conditions. That
invariant is the **fairness lock**, and it is encoded in the run configuration,
[`configs/default.yaml`](../configs/default.yaml), whose header states it directly:

> Default benchmark run configuration. The budget is keyed by dataset, never by arm:
> every arm sees the same budget for a given dataset (the fairness lock).

The lock has three parts:

1. **Identical model, budget, and tools across arms.** The configured `arms`
   (`none, bm25, tfidf, dense, hybrid_rrf, rerank_ce, raptor, brief_graph_3hop`)
   all run on the same `models` (`claude`) over the same `datasets`
   (`synthetic, dcbench, swebench`). Each arm implements one contract
   (`write` / `retrieve(query, budget)`); the harness swaps the arm and holds
   everything else fixed. As [`docs/METHODOLOGY.md`](./METHODOLOGY.md) puts it:
   "the harness swaps the arm and holds everything else fixed (the fairness lock)."

2. **Budget keyed by dataset, not by arm.** `budget_tokens_by_dataset` in
   `configs/default.yaml` assigns one budget per dataset
   (`synthetic: 150`, `dcbench: 250`, `swebench: 500`, `longmemeval: 8000`), and
   *every* arm sees that same budget for the dataset. No arm can be handed a larger
   context window than its competitors. The measured token-per-query cells bear this
   out: at the pooled `@all` slice, `tokens_per_query` is `Brief 1389.0278` vs.
   `dense 1399.2454` vs. `BM25 1400.6898` (all `measured` in `results/METRICS.md`) —
   the retrieval arms are matched to ~1%.

3. **Offline and deterministic.** `offline: true`, `retry_policy: single_shot`, and
   a fixed `seed: 0` make the harness reproducible: a stub agent honours exactly the
   identifiers present in the retrieved context, so compliance is driven by retrieval
   quality and cost by real prompt size, with no API calls. This is what lets the
   349 measured cells be a clean function of the retrieval arm under test.

Because the budget is keyed by dataset rather than by arm, the comparison cannot be
gamed by quietly giving the structured arm a bigger budget — the headline measured
results (e.g. `recall@all` Brief `0.8441` vs. dense `0.7677`) are read at equal cost.

## How the code prevents mixing tiers

The separation is not just a documentation convention; the analysis layer **refuses
to render a comparison that mixes tiers.** The guard lives in
[`src/membench/competitors.py`](../src/membench/competitors.py):

- `ProvenanceTier` is an enum with `MEASURED` and `VENDOR_REPORTED`.
- `MixedTierError` is raised "when measured and vendor-reported numbers would be
  mixed in one view."
- `assert_single_tier(tiers)` returns the single tier present, or raises
  `MixedTierError` if more than one distinct tier is supplied:

  > The analysis layer calls this before rendering any comparison so a measured
  > number and a vendor-reported number can never share a column.

The competitor table builder in
[`src/membench/analysis/tables.py`](../src/membench/analysis/tables.py) uses it:
`competitor_matrix(...)` produces a `TwoTierMatrix` with the measured arms in one
block and the cited vendor-reported claims in a separate, labelled block, calling
`assert_single_tier` on each block so "a measured number and a vendor-reported
number can never share a column." Its docstring is explicit: "the two tiers are
never merged."

So three mechanisms compound:

- **At table-render time**, `assert_single_tier` makes mixing measured and vendor
  numbers a hard error (`MixedTierError`), not a stylistic preference.
- **At load time**, `verified_claims()` drops any vendor entry still flagged
  `needs_citation`, so unverified placeholders cannot leak into a published table.
- **At the data level**, every cell in `results/METRICS.md` carries an explicit
  `tier` label and the `modeled` notes spell out the derivation, so derived numbers
  are visibly not measurements.

## Repo-measured vs. paper-reported

This repository's result files (`results/METRICS.md`, `results/data/`) are the
**repo-measured** source of truth and are tagged `measured` / `modeled` / `vendor`
as above. The companion paper ("Depth, Not Length") reports its own
**paper-reported** headline figures, which are rounded or pooled differently and
must not be presented as identical to the repo-measured cells.

For example, the pooled real-code use factor is reported in the paper as
**κ = 0.703** (paper-reported), while the repo's `results/METRICS.md` reports the
pooled all-data compliance for Brief as **0.7037** (repo-measured) — related
quantities, distinct numbers, distinct provenance. When citing a figure, keep its
origin attached: repo-measured cells point to `results/`; paper-reported figures
point to the paper. They are never collapsed into one number.

## Files

- [`results/METRICS.md`](../results/METRICS.md) — the 387-cell, 10-bucket matrix; source of the tier split (measured=349, modeled=13, vendor=25).
- [`competitors/registry.yaml`](../competitors/registry.yaml) — the Tier-2 vendor-reported claims, each with `source_url`, `conditions`, and `needs_citation`.
- [`src/membench/competitors.py`](../src/membench/competitors.py) — `ProvenanceTier`, `MixedTierError`, `assert_single_tier`, `verified_claims`.
- [`src/membench/analysis/tables.py`](../src/membench/analysis/tables.py) — `competitor_matrix` / `TwoTierMatrix`; applies the guard per block.
- [`configs/default.yaml`](../configs/default.yaml) — the fairness lock: arms, model, and `budget_tokens_by_dataset` (keyed by dataset, never by arm).
- [`docs/METHODOLOGY.md`](./METHODOLOGY.md) — the design rationale behind the fairness lock.
