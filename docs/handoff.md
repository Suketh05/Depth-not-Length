# Handoff — membench benchmark lock

**This document is authoritative.** Where it conflicts with the older test-plan or the
background PDFs, this wins. It is the finalized, decided version of the benchmark design.

## The thesis (paste first in any new chat)

Coding agents fail not because they run out of tokens but because similarity-based
memory can't recover decisions that sit several reasoning hops away from the current
task. The fix is structured memory (Brief): typed records joined by explicit links, so
retrieval follows the dependency instead of guessing by resemblance. The claim is not
"fewer tokens." It's "better-spent tokens": at a roughly fixed per-call budget,
structured memory honors past decisions more often, so total cost-to-correct drops.
"Better at coding" is the flagship result, but depth recovery is the headline claim, not
coding itself.

## The locked benchmark set

Three datasets, one distinct failure mode each, no overlap:
- **dcbench** — decisions invisible in code.
- **SWE-bench** — depth d>=2 constraint in a remote module.
- **LongMemEval** — using only the temporal and knowledge-update splits (temporal
  supersession).

dcbench and SWE-bench both run spec-stripped with the 4-arm ablation. Nothing else is
needed.

Explicitly dropped, with reasons:
- FRAMES and static HotpotQA — a smarter retriever fixes them, so they don't isolate
  similarity failure.
- StreamBench and HotpotQA multi-session — redundant unless hand-built.
- DepDec-Bench / coding-agent-life / SWE-EVO — coding breadth, not a new failure mode.
- LoCoMo and RULER — wrong domain / reframes as context-length.

## The arms (model + memory combinations)

Each row is a base model paired with a memory system. Main table holds the model fixed
and sweeps memory: Claude+none, Claude+full-context, Claude+Mem0, Claude+Brief.
Robustness table holds Brief fixed and sweeps the model: Claude+Brief, GPT+Brief,
open-weight+Brief. Plus the random-context arm (model + random nodes at Brief's token
budget) that proves the win is structure, not budget. You sweep two lines through the
grid, not the whole cube.

## The 4-arm ablation (on dcbench and SWE-bench only)

- Full-spec / no-memory (ceiling)
- Stripped-spec / no-memory (floor)
- Stripped-spec + graph (should recover toward ceiling)
- Random-context at the same token budget (should stay at floor)

This is the load-bearing piece. It's what proves *similarity* failed rather than context
simply being absent, and that the recovery comes from structure rather than tokens.

## The metrics

- **Primary:** decision compliance and merge-ready correctness.
- **Payoff:** cost-to-correct in tokens and dollars.
- **Diagnostic:** retrieval precision/recall and iterations-to-correct.
- **Control, reported but never optimized:** per-call tokens, held roughly constant
  across arms.

## The four output tables

- **Headline** — per dataset, model fixed, rows = memory arms.
- **Model-robustness** — Brief fixed, rows = models.
- **Depth crossover** — rows = memory arms, columns = d=1, d=2, d=3+, cells =
  compliance; similarity collapses while Brief stays flat, they cross near d=2.
- **Ablation** — the four arms above.

## The repo shape

```
membench/
  adapters/      base.py + none, fullcontext, mem0, brief, random_context
  benchmarks/    dcbench.py, swebench.py, longmemeval.py
  agent/         runner.py
  scoring/       compliance.py, correctness.py, cost.py
  configs/       which dataset x which arm x which model
  results/       raw jsonl + summary tables
  run.py
```

The one rule that makes it work: every memory system implements the same two methods
(`write`, `retrieve`) and every benchmark emits the same Task object. Then `run.py` is
just a loop over tasks x arms, and the four tables fall out. Your teammate only touches
`configs/` and `brief.py` (API key + endpoint) and runs from the office.

## Build order

1. Freeze the two contracts first: the `MemorySystem` interface and the `Task` format.
   Everything depends on these.
2. Build none + full-context arms and the dcbench loader. Get one number end to end.
   That's your skeleton proof.
3. Write scoring: compliance, correctness, cost-to-correct, with per-call tokens logged
   as a reported field.
4. Add arms one at a time (Mem0, Brief, random-context), each tested against the same
   dcbench run so you can diff fairly.
5. Add the spec-stripping harness and wire the 4-arm ablation.
6. Add the SWE-bench and LongMemEval loaders.
7. Lock fairness in configs: same model, same code-search tool, same retrieval budget
   across arms, recorded so every run reproduces.

## Two open design questions to settle

- **The depth-labeling rule** — how you tag a task's reasoning-chain depth so the
  crossover x-axis isn't subjective.
- **The spec-stripping recipe** — exactly how much spec to remove and where the stripped
  context gets stored.

Both are needed before the ablation is real. (Note: these two collapse into one
operation — strip-count = causal depth = d.)
