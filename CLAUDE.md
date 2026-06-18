# CLAUDE.md — membench invariants

This file is the source of truth. Re-read it before any change. If a prompt would
violate anything here, stop and ask first.

## Thesis (two sentences)
Coding agents fail not because they run out of tokens but because similarity-based
memory cannot recover decisions that sit several reasoning hops away from the current
task. Structured memory (Brief) — typed records joined by explicit links — lets
retrieval follow the dependency instead of guessing by resemblance, so at a roughly
fixed per-call token budget the agent honors past decisions more often and total
cost-to-correct drops.

The headline claim is **depth recovery**, not coding itself. "Better at coding" is the
flagship result; the mechanism being proven is that *similarity* failed, and that the
fix is *structure*, not more tokens.

## The authority order of docs
`docs/handoff.md` is authoritative. Where it conflicts with the older test-plan PDF,
the handoff wins. The four PDFs are background/derivation; the handoff is the spec.

## Locked set — do not expand
**3 datasets, exactly:**
- `dcbench` — decisions invisible in code. Spec-strippable. In 4-arm ablation.
- `swebench` (SWE-bench) — depth d>=2 constraint in a remote module. Spec-strippable.
  In 4-arm ablation.
- `longmemeval` (LongMemEval) — **temporal + knowledge-update splits ONLY**. Not
  stripped; depth read off their existing event/multi-hop structure.

**Explicitly DROPPED — never build loaders for these:**
FRAMES, static HotpotQA, StreamBench, HotpotQA multi-session, DepDec-Bench,
coding-agent-life, SWE-EVO, LoCoMo, RULER.

**5 arms (each a MemorySystem implementation):**
- `none` — empty context (floor)
- `fullcontext` — whole corpus truncated to budget ("just buy tokens" control)
- `mem0` — similarity top-k to budget
- `brief` — graph walk; reads endpoint + key from env, never hardcoded
- `random_context` — random nodes filled to **Brief's** budget (proves structure, not
  budget, is the cause). Its budget must equal Brief's exactly.

**4 output tables (groupbys over results/*.jsonl, no special code):**
1. Headline — model fixed (Claude), rows = memory arms, per dataset.
2. Model-robustness — Brief fixed, rows = models.
3. Depth crossover — rows = arms, cols = d=1 / d=2 / d=3+, cells = compliance.
   Similarity collapses, Brief stays flat, they cross near d=2.
4. Ablation — the 4 arms below, on dcbench + swebench ONLY.

**The 4-arm ablation (dcbench + swebench only):**
- full-spec / none  → ceiling
- stripped-spec / none → floor
- stripped-spec + brief → recovers toward ceiling
- random-context at same token budget → stays at floor
This is the load-bearing piece: it proves similarity failed (not absence of context)
and that recovery comes from structure (not tokens).

## Metrics
- Primary: decision compliance; merge-ready correctness.
- Payoff: cost-to-correct (tokens and dollars), summed over attempts.
- Diagnostic: retrieval precision/recall (retrieved IDs vs gold), iterations-to-correct.
- Control, reported but NEVER optimized: per-call tokens, held ~constant across arms.

## The two contracts — everything depends on these
**MemorySystem** (adapters/base.py):
- `write(items: list[MemoryItem]) -> None`
- `retrieve(query: str, budget_tokens: int) -> RetrievedContext`
- RetrievedContext carries BOTH assembled text AND the list of retrieved item IDs
  (IDs are needed to score retrieval precision/recall — never discard them).
- `budget_tokens` is passed IN by the caller, identical across arms. An arm never
  chooses its own budget.

**Task** (benchmarks/task.py):
task_id, dataset, query, repo_ref (None for LongMemEval), memory_corpus
(list[MemoryItem] that write() ingests, incl. stripped spec), governing_decisions
(gold IDs the output must honor), depth (int d), spec_variant ("full"|"stripped"),
scorer (str).

If a change would touch base.py or task.py, STOP and ask first.

## Fairness rule (encoded in configs/, recorded for reproducibility)
Same model, same code-search tool, same retrieval budget across all arms. Memory is
the only variable. The grid is swept as TWO LINES, not the whole cube:
- main table: model fixed (Claude), sweep memory
- robustness table: memory fixed (Brief), sweep model

## Depth == strip-count (one operation, not two)
Spec-stripping and depth-labeling are the same dial on coding datasets:
- d=1 → full spec at edit site (vanilla control)
- d=2 → strip the immediate constraint into memory_corpus
- d=3+ → strip the constraint AND its justification into memory_corpus
Depth labels live in benchmarks/depth_labels.jsonl (data, not code), per the
two-annotator protocol (two trace the path independently; tasks disagreeing by >1 hop
are dropped or rewritten). LongMemEval is not stripped — depth is read off its
existing structure.

## Repo shape
```
membench/
  adapters/      base.py + none, fullcontext, mem0, brief, random_context
  benchmarks/    task.py, dcbench.py, swebench.py, longmemeval.py, depth_labels.jsonl
  agent/         runner.py, llm_client.py
  scoring/       compliance.py, correctness.py, cost.py
  configs/       dataset x arm x model pins
  results/       raw jsonl + summary tables
  run.py
```

## Standing instruction
Build ONE slice at a time. Do not add arms, loaders, models, or scope beyond what the
current prompt asks. Single-shot vs retry-until-correct is a config flag, default
single_shot. Per-call tokens are reported, never optimized.
