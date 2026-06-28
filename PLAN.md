# PLAN — 50 research-grade, core-codebase PRs for BriefBench (`membench`)

Live checklist for taking the repository forward as **exactly 50 substantial pull requests**,
each branched **independently off `main`**. Every PR adds *real research/engineering value to the
core codebase* — a new retrieval method, statistical estimator, theory model, metric, dataset
generator, or harness internal — implemented to the repo's actual interfaces, with tests that
assert known closed-form / textbook golden values. **No docs-only or CI-only filler.**

## Operating rules (enforced for every PR)

- **Branch fresh off main:** `git switch main && git fetch origin && git reset --hard origin/main`
  then `git switch -c <type>/<slug>`. Independent, non-stacked branches.
- **File-disjoint by construction:** every PR adds *new* modules + *new* test files that no other
  PR touches. The only edits to existing shared files are `src/membench/harness.py` (PR #47 only)
  and `src/membench/cli.py` (PR #49 only) — each owned by exactly one PR. Any merge order applies
  with zero/near-zero conflicts.
- **Conforms to real interfaces** (verified): retrieval arms subclass `MemorySystem(ABC)`
  (`write`, `retrieve(query, budget_tokens) -> RetrievedContext`), are registered with
  `@register_arm(...)`, pack with `pack_to_budget`, and run through `run_benchmark(arms=[name...])`;
  stats/theory return bare `float` or frozen `@dataclass(slots=True)`; `Config` is pydantic. New
  arms/datasets are exercised end-to-end through the real harness in their tests (importing the
  module triggers registration — no edit to `builtin.py`/`harness.py` required).
- **Correctness bar:** each algorithm is the standard, cited form; the module carries NumPy-style
  docstrings (ruff `D`/`NPY` families) and full type hints (mypy `strict`). Tests assert a
  hand-computed / textbook golden value **and** invariants. No fabricated benchmark numbers.
- **Green gate before opening:** `uv run ruff check <files>`, `uv run ruff format --check <files>`,
  `uv run mypy src`, `uv run pytest <new test file>` — all pass. Baseline on `main`:
  **487 passed, 7 skipped**.
- **Behavior-preserving:** new modules are additive (off by default); existing results/defaults are
  never changed. The 50 PRs are left **OPEN** for the maintainer. No force-push to `main`.

`[ ]` not opened · `[x]` PR open & green

---

## PR #1 — roadmap
- [x] **1.** `docs: PLAN.md (this roadmap)` — `PLAN.md`

## Retrieval methods — new arms conforming to `MemorySystem` + `@register_arm` (tested through `run_benchmark`)
- [ ] **2.** `feat(retrieval): MMR diversity reranker (Carbonell & Goldstein 1998)` — `retrieval/mmr.py` + `tests/test_mmr.py`
- [ ] **3.** `feat(retrieval): CombSUM/CombMNZ/CombANZ score fusion (Fox & Shaw 1994)` — `retrieval/fusion.py` + `tests/test_fusion.py`
- [ ] **4.** `feat(retrieval): RM3 / Rocchio pseudo-relevance feedback expansion` — `retrieval/rm3.py` + `tests/test_rm3.py`
- [ ] **5.** `feat(retrieval): fielded BM25F (Robertson et al.)` — `retrieval/bm25f.py` + `tests/test_bm25f.py`
- [ ] **6.** `feat(retrieval): BM25+ lower-bounded TF (Lv & Zhai 2011)` — `retrieval/bm25_plus.py` + `tests/test_bm25_plus.py`
- [ ] **7.** `feat(retrieval): query-likelihood LM, Dirichlet & Jelinek-Mercer (Zhai & Lafferty)` — `retrieval/lmdir.py` + `tests/test_lmdir.py`
- [ ] **8.** `feat(retrieval): ColBERT-style late-interaction MaxSim (offline)` — `retrieval/colbert_lite.py` + `tests/test_colbert_lite.py`
- [ ] **9.** `feat(retrieval): deterministic learned-sparse term expansion (SPLADE-style)` — `retrieval/splade_lite.py` + `tests/test_splade_lite.py`
- [ ] **10.** `feat(retrieval): personalized PageRank over the typed graph (similarity-seeded)` — `retrieval/ppr.py` + `tests/test_ppr.py`
- [ ] **11.** `feat(retrieval): spreading activation over the context graph` — `retrieval/spreading_activation.py` + `tests/test_spreading_activation.py`
- [ ] **12.** `feat(retrieval): content-addressed LRU cache wrapper for a MemorySystem` — `retrieval/cache.py` + `tests/test_retrieval_cache.py`
- [ ] **13.** `feat(retrieval): RRF fusing similarity ranks with graph-hop ranks` — `retrieval/rank_fusion_graph.py` + `tests/test_rank_fusion_graph.py`

## Statistics — new estimators (conforming return types) + tests with golden values
- [ ] **14.** `feat(stats): TOST equivalence testing (Schuirmann)` — `stats/equivalence.py` + `tests/test_stats_equivalence.py`
- [ ] **15.** `feat(stats): DerSimonian-Laird random-effects meta-analysis (+ I², Q)` — `stats/meta_analysis.py` + `tests/test_stats_meta_analysis.py`
- [ ] **16.** `feat(stats): Jeffreys/BIC Bayes factor for proportions` — `stats/bayes_factor.py` + `tests/test_stats_bayes_factor.py`
- [ ] **17.** `feat(stats): split conformal prediction intervals` — `stats/conformal.py` + `tests/test_stats_conformal.py`
- [ ] **18.** `feat(stats): always-valid confidence sequence / e-value SPRT` — `stats/sequential.py` + `tests/test_stats_sequential.py`
- [ ] **19.** `feat(stats): isotonic regression (PAVA) + monotone depth-trend test` — `stats/isotonic.py` + `tests/test_stats_isotonic.py`
- [ ] **20.** `feat(stats): Storey q-values + Benjamini-Yekutieli FDR` — `stats/fdr_extra.py` + `tests/test_stats_fdr_extra.py`
- [ ] **21.** `feat(stats): partial & semipartial correlation` — `stats/partial_correlation.py` + `tests/test_stats_partial_correlation.py`
- [ ] **22.** `feat(stats): aligned rank transform for factorial nonparametrics (Wobbrock 2011)` — `stats/aligned_rank_transform.py` + `tests/test_stats_aligned_rank_transform.py`
- [ ] **23.** `feat(stats): studentized (bootstrap-t) & subsampling CIs` — `stats/bootstrap_studentized.py` + `tests/test_stats_bootstrap_studentized.py`
- [ ] **24.** `feat(stats): quantile regression of compliance on depth` — `stats/quantile_regression.py` + `tests/test_stats_quantile_regression.py`
- [ ] **25.** `feat(stats): g-computation / standardization estimator` — `stats/g_computation.py` + `tests/test_stats_g_computation.py`

## Theory — new models extending the depth/recovery framework + tests
- [ ] **26.** `feat(theory): closed-form bounds & monotone win-window for d⋆` — `theory/bounds.py` + `tests/test_theory_bounds.py`
- [ ] **27.** `feat(theory): Fano / channel-capacity context-free floor` — `theory/capacity.py` + `tests/test_theory_capacity.py`
- [ ] **28.** `feat(theory): scatter-penalty p^σ model` — `theory/scatter.py` + `tests/test_theory_scatter.py`
- [ ] **29.** `feat(theory): sensitivity / elasticity of recovery & d⋆ wrt (s0,ρ,q)` — `theory/sensitivity.py` + `tests/test_theory_sensitivity.py`
- [ ] **30.** `feat(theory): supersession-edge recovery model` — `theory/supersession.py` + `tests/test_theory_supersession.py`
- [ ] **31.** `feat(theory): multi-hop decay composition algebra` — `theory/composition.py` + `tests/test_theory_composition.py`

## Metrics — new scorers (operate on `ScoredRow`/arrays) + tests
- [ ] **32.** `feat(metrics): explicit P_comply = P_ret · κ factorization scorer (+ κ estimation)` — `metrics/factorization.py` + `tests/test_metrics_factorization.py`
- [ ] **33.** `feat(metrics): area-under-depth-curve + depth half-life` — `metrics/area_under_depth.py` + `tests/test_metrics_area_under_depth.py`
- [ ] **34.** `feat(metrics): recency-correct supersession rate` — `metrics/supersession.py` + `tests/test_metrics_supersession.py`
- [ ] **35.** `feat(metrics): normalized regret vs oracle` — `metrics/regret.py` + `tests/test_metrics_regret.py`
- [ ] **36.** `feat(metrics): accuracy-cost Pareto frontier + hypervolume` — `metrics/pareto.py` + `tests/test_metrics_pareto.py`
- [ ] **37.** `feat(metrics): RBP / ERR / MAP / R-precision` — `metrics/ir_extra.py` + `tests/test_metrics_ir_extra.py`
- [ ] **38.** `feat(metrics): Brier score + Murphy decomposition` — `metrics/brier.py` + `tests/test_metrics_brier.py`
- [ ] **39.** `feat(metrics): inter-arm agreement (Fleiss' κ, Krippendorff's α)` — `metrics/agreement.py` + `tests/test_metrics_agreement.py`

## Dataset generators / loaders — produce valid `Task`s, exercised through an arm
- [ ] **40.** `feat(datasets): HotpotQA multi-hop loader` — `datasets/hotpotqa.py` + `tests/test_hotpotqa.py`
- [ ] **41.** `feat(datasets): LoCoMo long-memory loader` — `datasets/locomo.py` + `tests/test_locomo.py`
- [ ] **42.** `feat(datasets): DMR (Deep Memory Retrieval) loader` — `datasets/dmr.py` + `tests/test_dmr.py`
- [ ] **43.** `feat(datasets): configurable supersession-chain generator` — `datasets/supersession_synth.py` + `tests/test_supersession_synth.py`
- [ ] **44.** `feat(datasets): distractor-scatter depth generator` — `datasets/scatter_synth.py` + `tests/test_scatter_synth.py`
- [ ] **45.** `feat(datasets): parametric N-hop reasoning-chain generator` — `datasets/depth_chain.py` + `tests/test_depth_chain.py`

## Core / harness / analysis internals
- [ ] **46.** `feat(core): centralized determinism/seed context manager` — `membench/seeding.py` + `tests/test_seeding.py`
- [ ] **47.** `feat(harness): accept ad-hoc arm instances in run_benchmark (additive)` — `src/membench/harness.py` + `tests/test_harness_registry.py`
- [ ] **48.** `feat(analysis): depth-curve fitting & crossover extraction helpers` — `analysis/effect_curves.py` + `tests/test_analysis_effect_curves.py`
- [ ] **49.** `feat(cli): frontier subcommand (Pareto / AUDC over a results file, self-contained)` — `src/membench/cli.py` + `tests/test_cli_frontier.py`
- [ ] **50.** `feat(analysis): parallel sweep runner matching serial output` — `analysis/sweep_parallel.py` + `tests/test_sweep_parallel.py`

---

### Prior open PRs (reviewed, left OPEN for the maintainer)
- **#45** feat/dashboard — *changes needed* (CI red: ruff). **#46/#47/#48** dependabot — *merge* (CI green).

The 50 new branches avoid all paths touched by these PRs.
