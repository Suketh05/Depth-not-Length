# Headline results (paper-reported)

All figures are transcribed verbatim from the paper (the source of truth). "Brief" = the
`brief_graph_3hop` typed decision-graph arm; answer model = Claude (Sonnet) unless noted.

- **Real code (pooled dcbench + swebench)** — the typed store leads: recall **0.667**,
  compliance **0.469**, precision **0.385**, use-factor **κ = 0.703** — the only arm to
  clear the **0.56–0.64** κ band that ceilings every similarity arm. (`tab:realret`)
- **Synthetic depth suite** — compliance: Brief **0.933** vs tfidf 0.883, bm25 0.875,
  dense 0.858, raptor 0.817, none 0.025. (`tab:synthall`)
- **Crossover** — a provable crossover depth **d⋆ = 2** (closed-form smallest `d` with
  `g(d) > 0` is 1; **d⋆ = 2** is the calibrated value for an overhead margin
  `τ ∈ (0.50, 0.79]`, at `s₀ = 0.70, ρ = 0.67, q = 0.97`; `g(1) ≈ 0.50`, `g(2) ≈ 0.79`).
  (`prop:cross`)
- **Supersession** — Brief ranks the current (non-superseded) decision first **92.3%** of
  the time vs the similarity band **64–69%** (bm25 64.1, tfidf 65.8, dense 68.7; chance
  50%). (`tab:super`)
- **Public benchmarks** — LoCoMo **87.6** (next best RAPTOR 84.7); DMR **94.2** (next best
  Kluris 87.0); SWE-ContextBench **47.3%** (next best Unabyss 37.6%); HotpotQA recall@5
  **0.783**, nDCG@10 **0.987**, MRR **0.981** (the one row Brief loses is HotpotQA
  supporting-fact recall, 0.503 vs Zep 0.529). (`tab:stdbench`, `tab:t101pub`)
- **Token economics** — Brief is the cheaper session in **80.0%** of matchups
  (2880 / 3600, 0 ties), 68.6%–86.7% per competitor. (`tab:tok_context_winrate`,
  `tab:tok_context_by_competitor`)
- **Unbiased 41-axis scorecard** — **22 W / 14 m (near-tie) / 5 l**. (`tab:scorecard`)

**Honest boundary.** The harness supplies a common pre-extracted decision corpus, so it
measures retrieval and *use* of typed links, not extraction; the paper states this is not
an end-to-end product win.
