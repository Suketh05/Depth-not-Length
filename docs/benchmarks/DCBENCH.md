# DC-bench: Decisions Invisible in Code

A deep-dive on the `dcbench` dataset — what it is, how its tasks and decisions are
shaped, how depth is dialed, how compliance and merge-readiness are scored on it,
and the repo-measured headline numbers (kept clearly separate from paper-reported
figures). The animating idea: on a real product repo, the constraint that should
govern an edit is often **not visible in the code you are about to touch**. It lives
in a team decision recorded somewhere else, and the agent has to recover it.

## Where it comes from

`dcbench` is a vendored snapshot of `github.com/brief-hq/dcbench`, a purpose-built
Next.js benchmark repo. The loader pins the exact source commit:

- `_SOURCE_COMMIT = "9f507ce0a6ef61124aca90cc4e17fb80592a5e4e"`

(see [`src/membench/datasets/dcbench.py`](../../src/membench/datasets/dcbench.py)).

Two JSON files carry the data, both transcribed verbatim from the source repo with
provenance recorded in a `_provenance` block:

- [`decisions.json`](../../src/membench/datasets/data/dcbench/decisions.json) —
  the team decisions (from `benchmark/seed-data.ts`, `BENCHMARK_DECISIONS`).
- [`tasks.json`](../../src/membench/datasets/data/dcbench/tasks.json) —
  the coding tasks (from `benchmark/tasks.ts`, `BENCHMARK_TASKS`).

The transcription drops the original TypeScript scoring fields (git-diff regex
pass/fail patterns, `expectedFiles`) because they are not part of this harness's
`Task`/`MemoryItem` contract, and it deliberately excludes `TASK-014`: its only
gotcha (`D-PERSONA-001`) tests a persona, not a seeded decision, so it falls outside
dcbench's "decisions invisible in code" framing.

## What a "decision" looks like

Each entry in `decisions.json` is a structured team decision. Example (`D-001`,
verbatim from the data):

```json
{
  "id": "D-001",
  "topic": "Date range component standardization",
  "decision": "DateRangePicker is the standard date range component. CalendarRange is deprecated after the mobile redesign and must not be used in new code.",
  "rationale": "CalendarRange had accessibility issues on mobile and was replaced during the mobile redesign. All new date-range UIs must use DateRangePicker from the design system.",
  "category": "design",
  "severity": "important"
}
```

Fields:

- **`id`** — `D-001`..`D-015`, assigned by array order, matching the
  `gotcha.decisionId` references in the source `tasks.ts` and the repo README.
- **`decision`** — the normative constraint itself (what to do / not do).
- **`rationale`** — *why* the constraint exists. This is the field the deep-depth
  variant splits off into a separate, link-connected node (see Depth labeling).
- **`topic`** — short label used as retrieval metadata.
- **`category`** — one of `tech`, `design`, `product`, `process`.
- **`severity`** — one of `blocking`, `important`, `info`.

## What a "task" looks like

Each entry in `tasks.json` is a feature request plus the decisions it silently
depends on. Example (`TASK-001`, verbatim):

```json
{
  "task_id": "TASK-001",
  "title": "CSV export to analytics dashboard",
  "prompt": "Add a CSV export button to the analytics dashboard. Users should be able to pick a date range, preview how many records they'll get, and download their data as a CSV file. Put the export button next to the existing filters.",
  "difficulty": "hard",
  "category": "full-stack",
  "gotchas": [
    { "decision_id": "D-002", "weight": 3 },
    { "decision_id": "D-001", "weight": 2 },
    { "decision_id": "D-003", "weight": 1 }
  ]
}
```

The load-bearing field is **`gotchas`**: a list of `{decision_id, weight}` pairs.
These are the governing decisions the correct implementation must honor — and the
prompt itself never names them at depth ≥ 2. In the loader, the governing set is
de-duplicated in first-seen order:

```python
governing = tuple(dict.fromkeys(g["decision_id"] for g in raw["gotchas"]))
```

(`src/membench/datasets/dcbench.py`). The `weight` field is the source repo's
severity hint for its own git-diff scorer; the offline compliance metric used here
treats each governing decision as one unit (see Scoring), so weights are carried as
data but do not skew the offline compliance rate.

## Depth labeling: one dial, the strip-count

Depth in dcbench is **strip-count**, not a different task set. The same 14 tasks are
re-presented at three depths, and depth controls both how much of the spec is left
in the prompt and how the decision corpus is shaped
([`src/membench/datasets/dcbench.py`](../../src/membench/datasets/dcbench.py)):

- **`d=1` (full / `SpecVariant.FULL`)** — the governing constraints are appended to
  the query under "Relevant team decisions you must honour". No memory is needed;
  this is the vanilla control. The loader rejects any other depth for `full`.
- **`d=2` (stripped)** — constraints are removed from the query, and each decision
  is one combined corpus node: `"{decision}\n\nRationale: {rationale}"`
  (`_combined_corpus`, `node_type: "decision"`). A single retrieval hop recovers it.
- **`d=3` (stripped)** — same query stripping, but each decision splits into a
  **constraint node** (`node_type: "constraint"`) and a **justification node**
  (`node_type: "justification"`, `item_id = "{id}-rationale"`). The justification
  carries a `constrains` back-reference to the constraint's id (`_split_corpus`).
  Recovering the whole picture takes a *second* hop — across the `constrains` link.

That `constrains` metadata is exactly what the structured-graph arm turns into a
`CONSTRAINS` edge, so depth-3 dcbench is, by construction, where a link-following
arm's advantage is *designed* to appear.

Which tasks are certified to which depth is recorded in
[`depth_labels.jsonl`](../../src/membench/datasets/data/depth_labels.jsonl); the
loader keeps only tasks whose `max_depth >= depth`. All 14 dcbench tasks are
labeled `max_depth: 3`, so each appears at d=1, d=2, and d=3.

**Honesty note on labeling provenance.** Every dcbench row in `depth_labels.jsonl`
records `dual_annotated: false` and an `annotation_method` of a *single-pass
automated check* (rationale present and distinct from the decision text; no 4-gram
overlap between the raw prompt and the decision text). The label itself flags that
this is **not** the two-independent-annotator protocol and should get a real
dual-annotation pass before being relied on for published numbers.

## Sizes (counted from the JSON)

Counted directly from the committed files in
[`src/membench/datasets/data/dcbench/`](../../src/membench/datasets/data/dcbench/):

| Quantity | Count | Source |
|---|---|---|
| Decisions (`D-001`..`D-015`) | **15** | `decisions.json` |
| Base tasks (after dropping `TASK-014`) | **14** | `tasks.json` |
| Governing-decision references (gotchas) across the 14 tasks | **28** | `tasks.json` |
| Distinct decisions actually referenced | **15** | `tasks.json` |
| Sum of gotcha `weight` values | **60** | `tasks.json` |
| Task instances across the three depths (14 × {d1,d2,d3}) | **42** | loader |

Distributions (counted from the JSON):

- Task difficulty: 8 `medium`, 3 `hard`, 3 `easy`.
- Task category: 6 `full-stack`, 3 `ui-component`, 2 `api-endpoint`, 2 `security`, 1 `configuration`.
- Decision category: 7 `tech`, 5 `design`, 2 `product`, 1 `process`.
- Decision severity: 7 `important`, 6 `info`, 2 `blocking`.

> **Paper-reported (do not conflate with the counts above).** The paper's dataset
> statistics table reports dcbench as **42 tasks (14 per depth)** and **41 weighted
> decision points** (paper-reported; `depth_not_length_FIXED.tex`, `tab:datastats`,
> per the verified paper digest). The "42 tasks / 14 per depth" figure matches the
> 14 base tasks × 3 depths above. The "41 weighted decision points" figure is a
> paper-reported quantity and is **not** identical to any single count in the JSON I
> read (which has 28 gotcha references and a `weight` sum of 60 across 14 tasks);
> they are reported separately here rather than reconciled.

## How compliance and merge-ready are scored

Both metrics are deterministic so the offline pipeline is reproducible; an optional
LLM-judge backend can replace the identifier check for live runs.

**Compliance** ([`src/membench/metrics/compliance.py`](../../src/membench/metrics/compliance.py)).
The `Task` contract carries gold decision *ids*, but a coding agent never emits the
literal string `"D-002"`. So for each governing id, the scorer looks up the
decision's text in the corpus (ground truth, present for every arm) and extracts the
concrete identifiers that decision names — function calls (`withAuditLog()`),
PascalCase/camelCase symbols (`DateRangePicker`, `withAuditLog`), and scoped
packages (`@t3-oss/env-nextjs`). A governing decision is **honored** if any of its
identifiers appears in the response; if a decision names no identifier, the bare id
is the last-resort key. The compliance `rate` is `honored / total` (and is `1.0`
when a task has no governing decisions). Because the identifier surface is exactly
what the agent can only echo if it actually retrieved the decision, compliance is
driven by whether the governing decision was recovered — which is the point.

**Merge-ready** ([`src/membench/metrics/correctness.py`](../../src/membench/metrics/correctness.py)).
A text-only proxy for "would this diff go in as written". A response is merge-ready
**iff** it is non-empty, is not a refusal (no `"i can't"`, `"i cannot"`, `"i won't"`,
`"as an ai"`, `"i'm not able to"` markers), **and** its compliance `rate == 1.0`
(every governing decision honored). Any unmet governing decision makes it not
merge-ready, with the reason `"{missed}/{total} governing decisions unmet"`. The
dependency on the compliance result is deliberate: a real diff+test harness can
replace this binary without touching the compliance contract.

## Repo-measured headline numbers (Claude, dcbench)

All figures in this section are **repo-measured** (our harness), copied verbatim
from [`results/METRICS.md`](../../results/METRICS.md),
[`results/EVALS.md`](../../results/EVALS.md), and
[`results/data/metrics_matrix.csv`](../../results/data/metrics_matrix.csv) — tier
`measured`, note `claude/dcbench`. Arms shown are those carried in `METRICS.md`
(Brief = the typed decision graph; "dense (best baseline)"; BM25; `none`;
`random_context`).

**Pooled over depths (`dcbench`, d=all):**

| metric | Brief | dense (best baseline) | BM25 | none | random_context |
|---|---|---|---|---|---|
| recall@dcbench | 0.7698 | **0.7817** | 0.7421 | 0.0 | 0.1984 |
| compliance@dcbench | 0.3571 | 0.3452 | **0.3651** | 0.1627 | 0.2222 |
| merge_ready@dcbench | 0.1667 | **0.1905** | **0.1905** | 0.0476 | 0.0714 |
| precision@dcbench | **0.4397** | 0.4331 | 0.4092 | 0.0 | 0.1224 |
| return_on_tokens@dcbench | 0.2709 | 0.26 | **0.2751** | 0.151 | 0.1654 |
| tokens_per_query@dcbench | 1304.4048 | 1313.1429 | 1318.7857 | 1103.6429 | 1315.881 |

(Source: `results/METRICS.md` buckets `1_retrieval_quality`, `2_task_outcome`,
`3_token_economics`, `5_context_management`; all `measured`. **Bold = best arm in
that row.**)

**Compliance by depth (`dcbench`):**

| depth | Brief | dense (best baseline) | BM25 | none | random_context |
|---|---|---|---|---|---|
| d=1 | **0.5** | 0.4643 | 0.4643 | 0.4643 | 0.4643 |
| d=2 | 0.2619 | 0.2381 | **0.2976** | 0.0 | 0.0357 |
| d=3 | 0.3095 | **0.3333** | **0.3333** | 0.0238 | 0.1667 |

(Source: `results/METRICS.md` bucket `4_depth_robustness`; all `measured`,
`claude/dcbench/d=1|2|3`.)

**Deep recall (`recall@d3_dcbench`, d=3):** Brief 0.6667, dense (best baseline)
**0.7024**, BM25 0.5952, none 0.0, random_context 0.1548
(`results/data/metrics_matrix.csv`, bucket `1_retrieval_quality`, `measured`).

**Product-Navigator cut (Brief vs none, `results/EVALS.md`, family
`F1_product_navigator`):** at d=all (n=42) Brief compliance 0.3571 vs none 0.1627
(Δ +0.1944); recall 0.7698 vs 0.0; precision 0.4397 vs 0.0; merge_ready 0.1667 vs
0.0476; chain_recovery 0.5476 vs 0.0; return_on_tokens 0.2709 vs 0.151. So against
the no-context control, Brief is a clear win on dcbench — the contrast below is with
*similarity* arms, not the bare baseline.

## Where similarity arms are competitive on dcbench (honest)

dcbench is the dataset where the typed-graph advantage is *muted*, and the
repo-measured numbers say so plainly. On the pooled, repo-measured cuts above:

- **Recall:** dense (best baseline) **0.7817** edges out Brief **0.7698** — dense,
  not the typed graph, leads pooled dcbench recall.
- **Compliance:** BM25 **0.3651** is the top arm, ahead of Brief **0.3571** and
  dense 0.3452 — a spread of ~0.01.
- **Merge-ready:** dense and BM25 tie at **0.1905**, both above Brief **0.1667**.
- **Return-on-tokens:** BM25 **0.2751** leads, with Brief 0.2709 and dense 0.26.
- **By depth:** Brief only wins compliance at **d=1** (0.5 vs 0.4643). At **d=2**,
  BM25 leads (0.2976 vs Brief 0.2619); at **d=3**, dense and BM25 tie at 0.3333 vs
  Brief 0.3095; and dense leads deep recall at d=3 (0.7024 vs 0.6667).

The statistical-confidence bucket of `results/METRICS.md` (all `measured`,
`dcbench`) confirms the ties are not in Brief's favor here:

- `P_brief_gt_dense_dcbench` = **0.3436** and `P_brief_gt_bm25_dcbench` = **0.3436**
  (both below 0.5).
- `cohens_h_vs_dense_dcbench` = **-0.051** and `cohens_h_vs_bm25_dcbench` =
  **-0.051** (negative effect sizes).
- Brief's d=3 compliance Wilson interval is wide: `brief_ci_low_dcbench_d3` =
  **0.1429** to `brief_ci_high_dcbench_d3` = **0.5238**.

The head-to-head `brief_vs_best_similarity` rows in
[`results/EVALS.md`](../../results/EVALS.md) (`measured`, n=14 per depth) make the
deltas explicit — Brief trails the best similarity arm on dcbench compliance at
every depth: d=1 0.5 vs 0.5357 (Δ -0.0357), d=2 0.2619 vs 0.3929 (Δ -0.131), d=3
0.3095 vs 0.4048 (Δ -0.0952); and on recall at d=2 (Δ -0.0238) and d=3 (Δ -0.0357).

> **Paper-reported, for cross-reference.** The paper's unbiased win/loss scorecard
> (paper-reported; `tab:scorecard`, per the verified digest) lists dcbench-specific
> losses among its 5 outright losses: **dcbench d=2 compliance → dense**, **dcbench
> d=3 recall → hybrid_rrf**, and **dcbench d=1 recall → bm25**. The pooled real-code
> story (dcbench+swebench combined) is reported separately in the paper as a typed-
> graph lead (paper-reported recall 0.667, compliance 0.469, use factor κ=0.703;
> `tab:realret`) — but that is a *pooled* figure and should not be read as a
> dcbench-only result. On dcbench alone, the repo-measured numbers above show
> similarity arms at or ahead of the typed graph on most cells.

## Why this matters

dcbench is the honest counterweight to the synthetic suite. On synthetic data the
typed graph dominates (repo-measured `compliance@synthetic` Brief 0.9917 vs dense
0.825; `results/METRICS.md`). On dcbench — a real, compressed Next.js repo with
low-to-medium drift — the gap collapses and similarity baselines are competitive or
better on most cells. Reporting both, side by side and unreconciled, is the point of
keeping dcbench in the suite.

## Sources

- Loader & corpus shaping: [`src/membench/datasets/dcbench.py`](../../src/membench/datasets/dcbench.py)
- Data: [`decisions.json`](../../src/membench/datasets/data/dcbench/decisions.json), [`tasks.json`](../../src/membench/datasets/data/dcbench/tasks.json)
- Depth labels: [`depth_labels.jsonl`](../../src/membench/datasets/data/depth_labels.jsonl)
- Scoring: [`compliance.py`](../../src/membench/metrics/compliance.py), [`correctness.py`](../../src/membench/metrics/correctness.py)
- Repo-measured numbers: [`results/METRICS.md`](../../results/METRICS.md), [`results/EVALS.md`](../../results/EVALS.md), [`results/data/metrics_matrix.csv`](../../results/data/metrics_matrix.csv)
- Paper-reported figures: `depth_not_length_FIXED.tex` (`tab:datastats`, `tab:scorecard`, `tab:realret`), via the verified paper digest.
