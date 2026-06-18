# METHODOLOGY.md — how to drive this yourself next time

This is the transferable method. The membench project is just one instance of it. Next
time you design a system with Claude Code, run this loop.

## The core problem you're managing
Claude Code's main failure mode is **building ahead** — it scaffolds everything at once,
and the pieces don't line up because the interfaces weren't fixed first. Every habit
below exists to prevent that.

## The five moves

### 1. Find the contracts before you find the features
Before any code, ask: *what are the 1–3 interfaces that everything else plugs into?*
Here it was `MemorySystem` (write/retrieve) and `Task`. If you get these right, the rest
is a loop; if you get them wrong, every feature is built on sand. Spend your slow,
careful attention here and nowhere else.

How to find them: look for the place where the spec says "every X does the same thing"
or "then it's just a loop over Y." That sentence names your contract. (In the handoff:
"every memory system implements write/retrieve and every benchmark emits the same Task
object. Then run.py is just a loop.")

### 2. Write a CLAUDE.md of invariants, not instructions
CLAUDE.md is re-read every turn, so it's your real persistent context. Put in it only
things that must stay true across the whole build: the thesis, the locked scope, the
contracts, the fairness/correctness rules, and a standing "build one slice, ask before
touching contracts" instruction. Keep it terse — it competes for attention every turn.
Don't put step-by-step build instructions here; those go in PLAN.md.

### 3. Establish an authority order for your inputs
You had four PDFs that partly contradicted each other and one handoff that superseded
them. State explicitly which document wins. Without this, Claude Code averages your
sources and reintroduces dropped scope (the old test-plan still listed DepDec, SWE-EVO,
etc.). One sentence — "X supersedes Y on conflict" — saves hours.

### 4. One vertical slice per prompt, end to end
Don't build all loaders, then all arms, then scoring. Build the thinnest path that
produces ONE real output (Step 2: none + fullcontext + one loader + runner →
one number), then widen. Each later prompt adds exactly one arm / one loader / one
scorer and is tested against the same earlier run so differences are attributable.
A prompt that says "add X only" and ends with "show me the output" is the right shape.

### 5. Parameterize open questions instead of blocking on them
You had two unresolved design questions (retry policy; depth-labeling). Don't stall.
Make retry a config flag with a default; make depth a checked-in data file. The build
proceeds and the decision slots in later as data/config, not a rewrite.

## The per-prompt template
Every build prompt should have these four parts:
1. **Scope fence** — "Implement ONLY X. No Y, no Z."
2. **The contract reference** — "per the Task/MemorySystem contract."
3. **The invariant reminders** that this slice could violate (budget is an input; IDs
   survive; per-call tokens reported not optimized).
4. **A check** — "run it on N cases and show me the output."

## The two reflexes that save you
- When it drifts: `Re-read CLAUDE.md. Revert the extra scope; do only what this prompt asked.`
- Before contract-adjacent edits: `Does this touch base.py/task.py? Show me the diff first.`

## Order of operations, abstracted
0. Load context + authority order → write CLAUDE.md (invariants only).
1. Freeze contracts (slow, careful, tested).
2. One vertical slice → one real number (skeleton proof).
3. Scoring / measurement layer.
4. Add interchangeable pieces one at a time, diffed against the skeleton.
5. Add the harness that unlocks your hardest test.
6. Add remaining data sources.
7. Lock reproducibility (configs) and assemble the final loop + outputs.

## What you still owe on membench specifically
- Which GPT model + which open-weight model & host (robustness row, Step 7).
- Confirm Brief endpoint + key via env at run time (Step 4b).
- Whether to ever flip retry_until_correct on for a realized-cost sub-study.
None block Steps 0–3, so you can start building today.
