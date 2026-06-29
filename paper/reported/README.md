# Paper-reported results (source of truth)

These artifacts are transcribed **verbatim** from the paper *"Depth, Not Length: A
Benchmark and Theory of Memory, Retrieval, and Decision Compliance in LLM Coding
Agents"* — the single source of truth for this repository's results. They **replace** the
previous offline measured-reproduction dossier; this repo no longer ships a divergent
measured run.

## What's here

- `summary.md` — the marquee results.
- `dossier.md` — the full per-arm / per-benchmark breakdown, each block tied to a paper
  table label.

The same numbers are also available as CSV under
[`../../results/data/`](../../results/data) (`paper_*.csv`).

## Provenance

Every value is copied verbatim from a labelled table in the manuscript and attributed to
it. **Nothing here is "measured" by the code in this repository.** The paper reports
**aggregate tables only** (no row-level attempt log), so there is no `rows.jsonl` in this
directory; the harness can still generate fresh rows on demand, but those are a separate
offline reproduction and are intentionally not committed as results.
