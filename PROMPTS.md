# PROMPTS.md — copy-paste prompts in order

Paste these into Claude Code one at a time, in order. Wait for each to finish and
read the output before the next. Keep `CLAUDE.md` in the repo root the whole time.

---

## Prompt 0 — load context, write CLAUDE.md

```
Read every file in docs/. docs/handoff.md is authoritative and supersedes the older
test-plan PDF wherever they conflict — in particular the locked dataset set is exactly
three: dcbench, SWE-bench, and LongMemEval (temporal + knowledge-update splits only).
DepDec-Bench, SWE-EVO, coding-agent-life, LoCoMo, RULER, HotpotQA, FRAMES, StreamBench
are all explicitly dropped — do not build loaders for them.

Then write a CLAUDE.md capturing, tersely, the invariants any future change must
respect. Do NOT write any application code yet. Include: the thesis in two sentences;
the locked set (3 datasets, 5 arms = none/fullcontext/mem0/brief/random_context,
4 tables, the metrics — compliance + merge-ready correctness primary, cost-to-correct
payoff, retrieval P/R + iters diagnostic, per-call tokens reported-but-never-optimized);
the two contracts everything depends on (MemorySystem.write/retrieve, the Task
dataclass); the fairness rule (same model, same code-search tool, same retrieval budget
across arms, budget passed INTO retrieve()); the repo shape; and a standing instruction:
"Build one slice at a time. Do not add arms, loaders, or scope beyond the current
prompt. If a change would touch the two contracts, stop and ask first."

Show me CLAUDE.md before writing it to disk.
```
(Or skip this and drop the provided CLAUDE.md in directly.)

---

## Prompt 1 — freeze the two contracts

```
Implement ONLY the two contracts, nothing else. No arms, no loaders, no runner.

1. adapters/base.py — abstract MemorySystem with two methods:
   - write(items: list[MemoryItem]) -> None
   - retrieve(query: str, budget_tokens: int) -> RetrievedContext
   RetrievedContext must carry BOTH the assembled text AND the list of item IDs that
   were retrieved (we score retrieval precision/recall against gold IDs, so IDs can't
   be thrown away). budget_tokens is passed in by the caller and is identical across
   arms — an arm never chooses its own budget.

2. benchmarks/task.py — Task dataclass and MemoryItem dataclass. Task fields: task_id,
   dataset, query, repo_ref (None for LongMemEval), memory_corpus (list of MemoryItem
   that write() ingests, including any stripped spec), governing_decisions (gold IDs
   the output must honor), depth (int d), spec_variant ("full"|"stripped"), scorer (str).

Add docstrings explaining why IDs survive retrieval and why budget is an input. Write a
couple of trivial tests that a dummy MemorySystem subclass satisfies the interface.
Stop here.
```

---

## Prompt 2 — one number end to end

```
Build the thinnest vertical slice that produces ONE real number:
- adapters/none.py (returns empty context) and adapters/fullcontext.py (returns the
  whole corpus truncated to budget_tokens)
- benchmarks/dcbench.py — loader for dcbench, FULL-spec variant only for now (the d=1
  control). Emit Task objects per the contract.
- agent/runner.py — the loop: build arm, write(corpus), retrieve(query, budget), call
  the model, score, log one JSONL row per attempt.
- agent/llm_client.py — an LLMClient abstraction with an Anthropic backend only. It must
  record input_tokens, output_tokens, and dollars per call.

Make retry policy a config flag (single_shot vs retry_until_correct), default
single_shot. Don't build other backends or arms. Run on 1–2 dcbench tasks and show me
the JSONL output.
```

---

## Prompt 3 — scoring

```
Implement scoring/ against the Task contract:
- compliance.py — decision compliance: of task.governing_decisions, how many the output
  honors.
- correctness.py — merge-ready correctness for coding tasks.
- cost.py — cost-to-correct in tokens and dollars (sum over attempts); retrieval
  precision/recall from RetrievedContext.ids vs task.governing_decisions;
  iterations-to-correct.
Per-call tokens must be logged as a plain reported field — never an optimization target,
never used to gate anything. Wire these into runner.py's log rows. Re-run the step-2
dcbench slice and show me the enriched JSONL.
```

---

## Prompt 4a — mem0 arm

```
Add adapters/mem0.py only — similarity top-k retrieval filled to budget_tokens. Run it
on the same dcbench-full tasks as the none/fullcontext arms and show me a side-by-side
of compliance + retrieved IDs across the three arms.
```

## Prompt 4b — brief arm

```
Add adapters/brief.py only — a structured-graph memory that walks typed links. Read the
Brief endpoint URL and API key from environment variables (BRIEF_ENDPOINT, BRIEF_API_KEY)
— never hardcode them. retrieve() must return the IDs of the nodes it walked. Run on the
same dcbench-full tasks and show me the side-by-side against the existing arms.
```

## Prompt 4c — random_context control

```
Add adapters/random_context.py only — selects random nodes from the corpus filled to the
SAME budget_tokens Brief uses. This is the control proving the win is structure, not
budget, so its token budget must match Brief's exactly. Run on the same dcbench-full
tasks; show me compliance for all five arms together.
```

---

## Prompt 5 — spec-stripping + ablation

```
Add the spec-stripping harness to dcbench.py. Strip-count IS depth: d=1 full spec at
edit site; d=2 strip the immediate constraint into memory_corpus; d=3+ strip the
constraint AND its justification into memory_corpus. Depth labels live in a checked-in
benchmarks/depth_labels.jsonl (data, not code), per the two-annotator protocol in docs.
Then wire the 4-arm ablation: full-spec/none (ceiling), stripped/none (floor),
stripped+brief (recovery), random_context at the same budget (floor). Ablation runs on
dcbench and SWE-bench only.
```

---

## Prompt 6 — remaining loaders

```
Add benchmarks/swebench.py (with the same spec-stripping harness) and
benchmarks/longmemeval.py (temporal + knowledge-update splits ONLY; do not strip — read
depth off their existing event/multi-hop structure). Both emit the same Task contract.
```

---

## Prompt 7 — fairness lock + assemble

```
Add configs/ as the fairness lock: each config pins dataset x arm x model with the same
code-search tool and same retrieval budget, recorded so runs reproduce. Encode the grid
restriction here — main table sweeps memory with model fixed (Claude); robustness table
sweeps model with Brief fixed. We sweep two lines, not the whole cube. Then make run.py
a loop over tasks x arms x models driven entirely by configs, and have scoring emit the
four tables (headline, model-robustness, depth-crossover, ablation) as groupbys over
results/*.jsonl.
```

---

## Correction prompts (keep handy)

- Drift / over-build: `Re-read CLAUDE.md. You've added scope beyond this prompt — revert X and do only Y.`
- Contract risk: `Does this change touch adapters/base.py or benchmarks/task.py? If so, stop and show me the diff before applying.`
- Fairness check: `Confirm this arm receives budget_tokens from the caller and does not set its own budget.`
