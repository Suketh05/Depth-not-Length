# PLAN.md — build order

Each step is one vertical slice. Do not start a step until the previous one runs and
you've read the files yourself. Artifacts in **bold** are what that step must produce.

## Step 0 — Load context (no code)
Drop the 4 PDFs + handoff into `docs/`. Produce **CLAUDE.md** (already drafted).
Review it by hand — every later step inherits its errors.

## Step 1 — Freeze the two contracts (be slow here)
**adapters/base.py** (MemorySystem, RetrievedContext) + **benchmarks/task.py** (Task,
MemoryItem). Plus trivial interface tests. Nothing else compiles until these are right;
nothing downstream is salvageable if they're wrong. IDs survive retrieval; budget is an
input.

## Step 2 — One number, end to end (skeleton proof)
**adapters/none.py**, **adapters/fullcontext.py**, **benchmarks/dcbench.py** (FULL-spec
only, the d=1 control), **agent/runner.py**, **agent/llm_client.py** (Anthropic backend
only; records input/output tokens + dollars per call). Retry policy is a config flag,
default single_shot. Run on 1–2 tasks, inspect JSONL.

## Step 3 — Scoring
**scoring/compliance.py** (honored governing_decisions), **scoring/correctness.py**
(merge-ready), **scoring/cost.py** (cost-to-correct in tokens+dollars; retrieval P/R
from RetrievedContext.ids vs gold; iterations-to-correct). Per-call tokens logged as a
plain reported field. Re-run the step-2 slice; inspect enriched JSONL.

## Step 4 — Remaining arms, one prompt each
**adapters/mem0.py**, then **adapters/brief.py** (env-based endpoint+key — teammate owns
this file), then **adapters/random_context.py** (random nodes to Brief's exact budget).
Diff each against the same dcbench-full run.

## Step 5 — Spec-stripping → unlocks ablation
Add stripping to **dcbench.py** (strip-count = depth), **benchmarks/depth_labels.jsonl**,
and wire the **4-arm ablation** (dcbench + swebench only).

## Step 6 — Remaining loaders
**benchmarks/swebench.py** (same stripping harness), **benchmarks/longmemeval.py**
(temporal + knowledge-update splits only; no stripping).

## Step 7 — Lock fairness + assemble
**configs/** (dataset x arm x model pins; same code-search tool + budget; encode the
two-line grid), **run.py** (loop over tasks x arms x models from configs), scoring emits
the **four tables** as groupbys over results/*.jsonl.

## Open inputs to slot in (don't block steps 0–3)
- Step 4 (brief.py): confirm Brief endpoint + key come from env at run time.
- Step 7 (configs): which GPT model, which open-weight model + host for robustness row.
- Retry policy: keep default single_shot unless you decide to measure realized
  cost-to-correct directly (then enable retry_until_correct for a sub-study).
