# PLAN — 50 independent PRs to harden BriefBench (`membench`)

This is the live checklist for taking the repository from its current state to a polished,
reproducible, well-documented project, delivered as **exactly 50 substantial pull requests**,
each branched **independently off `main`**.

## Operating rules (enforced for every PR)

- **Branch fresh off main:** `git switch main && git fetch origin && git reset --hard origin/main`
  then `git switch -c <type>/<slug>`. No stacked/dependent branches.
- **File-disjoint by construction:** the work is **additive-first** — most PRs add *new* files
  that no other PR touches. Each shared file (`README.md`, `CHANGELOG.md`, `pyproject.toml`,
  `src/membench/cli.py`) is edited by **exactly one** PR. Result: any merge order applies with
  zero/near-zero conflicts.
- **Green & validated:** every PR passes the repo's own gates before it opens. Python-touching
  PRs run `uv run ruff check`, `uv run mypy src`, and the relevant `uv run pytest`; docs/CI PRs
  validate markdown/YAML. Baseline on `main`: **487 passed, 7 skipped** (native/Rust kernels
  unbuilt), build green.
- **Integrity guardrail:** no benchmark number is invented, estimated, or reconstructed. Every
  metric in docs is copied **verbatim** from the paper (`/Users/kasyapvaranasi/Desktop/v1/depth_not_length_FIXED.tex`)
  or from committed result files (`results/data/*.csv`, `results/*.md`, `paper/measured/`). The
  measured / modeled / vendor provenance tiering is preserved.
- **Behavior-preserving** unless a PR explicitly states otherwise and ships tests for the new
  behavior. `main` stays green; no force-push to `main`; **the 50 PRs are left OPEN** for the
  maintainer to merge.
- **Labels:** apply existing repo labels (`documentation`, `enhancement`, `github_actions`,
  `bug`, `dependencies`, `rust`, `javascript`). PR #32 adds a path-based auto-labeler.

## Status legend

`[ ]` not opened · `[x]` PR open & green

---

## PR #1 — this plan
- [ ] **1.** `docs: add PLAN.md (50-PR roadmap & checklist)` — *documentation* — files: `PLAN.md`

## Documentation — new files (each a distinct topic, paper-grounded)
- [ ] **2.** `docs: glossary of core terms` — *documentation* — `docs/GLOSSARY.md` (depth, κ/use-factor, supersession, P_comply=P_ret·κ, d⋆, arms)
- [ ] **3.** `docs: architecture overview with mermaid diagrams` — *documentation* — `docs/ARCHITECTURE.md`
- [ ] **4.** `docs: reproduce-the-paper map (figures/tables → commands)` — *documentation* — `docs/REPRODUCE.md`
- [ ] **5.** `docs: theory derivation walkthrough` — *documentation* — `docs/THEORY.md`
- [ ] **6.** `docs: statistics menu explained` — *documentation* — `docs/STATISTICS.md`
- [ ] **7.** `docs: retrieval-arms catalog (22 arms)` — *documentation* — `docs/RETRIEVAL_ARMS.md`
- [ ] **8.** `docs: provenance tiers (measured/modeled/vendor)` — *documentation* — `docs/PROVENANCE.md`
- [ ] **9.** `docs: FAQ` — *documentation* — `docs/FAQ.md`
- [ ] **10.** `docs: DC-bench dataset deep-dive` — *documentation* — `docs/benchmarks/DCBENCH.md`
- [ ] **11.** `docs: SWE-bench dataset deep-dive` — *documentation* — `docs/benchmarks/SWEBENCH.md`
- [ ] **12.** `docs: LongMemEval dataset deep-dive` — *documentation* — `docs/benchmarks/LONGMEMEVAL.md`
- [ ] **13.** `docs: synthetic depth-crossover suite deep-dive` — *documentation* — `docs/benchmarks/SYNTHETIC.md`
- [ ] **14.** `docs: verbatim headline results from the paper` — *documentation* — `docs/RESULTS.md`
- [ ] **15.** `docs: competitor landscape (cited, two-tier)` — *documentation* — `docs/COMPETITORS.md`
- [ ] **16.** `docs: full CLI reference (all subcommands & flags)` — *documentation* — `docs/CLI.md`
- [ ] **17.** `docs: data-format reference (rows.jsonl / manifest / CSV schemas)` — *documentation* — `docs/DATA_FORMAT.md`

## Tests — new files (target real coverage gaps, not duplicate unit tests)
- [ ] **18.** `test: results-integrity (docs numbers == committed CSV/data)` — *enhancement* — `tests/test_results_integrity.py`
- [ ] **19.** `test: property-based theory invariants (Hypothesis)` — *enhancement* — `tests/test_property_theory.py`
- [ ] **20.** `test: property-based statistics invariants (Hypothesis)` — *enhancement* — `tests/test_property_stats.py`
- [ ] **21.** `test: CLI golden/smoke across all subcommands` — *enhancement* — `tests/test_cli_golden.py`
- [ ] **22.** `test: run determinism (same seed → identical rows)` — *enhancement* — `tests/test_determinism_seed.py`
- [ ] **23.** `test: compliance factorization identity P_comply=P_ret·κ` — *enhancement* — `tests/test_factorization_identity.py`
- [ ] **24.** `test: fairness-lock config validation (equal budget across arms)` — *enhancement* — `tests/test_fairness_lock_config.py`
- [ ] **25.** `test: provenance single-tier guard (no measured/vendor mixing)` — *enhancement* — `tests/test_provenance_tiers.py`
- [ ] **26.** `test: figure inventory (referenced figures exist on disk)` — *enhancement* — `tests/test_figure_inventory.py`
- [ ] **27.** `test: cross-language golden agreement (nDCG/BM25)` — *enhancement* — `tests/test_xlang_reference_golden.py`

## CI / GitHub automation — new workflow & config files
- [ ] **28.** `ci: CodeQL security scanning` — *github_actions* — `.github/workflows/codeql.yml`
- [ ] **29.** `ci: markdown link checker` — *github_actions* — `.github/workflows/docs-linkcheck.yml`
- [ ] **30.** `ci: spell-check (codespell)` — *github_actions* — `.github/workflows/spellcheck.yml`, `.codespellrc`
- [ ] **31.** `ci: stale issue/PR bot` — *github_actions* — `.github/workflows/stale.yml`
- [ ] **32.** `ci: path-based PR auto-labeler (+ create test/ci/chore/perf labels)` — *github_actions* — `.github/labeler.yml`, `.github/workflows/labeler.yml`
- [ ] **33.** `ci: release-drafter` — *github_actions* — `.github/release-drafter.yml`, `.github/workflows/release-drafter.yml`
- [ ] **34.** `ci: run pre-commit hooks in CI` — *github_actions* — `.github/workflows/pre-commit.yml`
- [ ] **35.** `chore: benchmark-result issue template` — *documentation* — `.github/ISSUE_TEMPLATE/benchmark_result.md`
- [ ] **36.** `ci: figures smoke-render workflow` — *github_actions* — `.github/workflows/figures-smoke.yml`
- [ ] **37.** `ci: shellcheck for shell scripts` — *github_actions* — `.github/workflows/shellcheck.yml`

## Tooling / scripts / config — new files
- [ ] **38.** `feat: standalone results-integrity checker script` — *enhancement* — `scripts/check_integrity.py`
- [ ] **39.** `feat: machine-readable results.json exporter` — *enhancement* — `scripts/export_results_json.py`
- [ ] **40.** `feat: Okabe-Ito colorblind palette module + tests` — *enhancement* — `src/membench/analysis/palette.py`, `tests/test_palette.py`
- [ ] **41.** `build: conda environment.yml` — *enhancement* — `environment.yml`
- [ ] **42.** `feat: data-manifest checksum generator + SHA256SUMS` — *enhancement* — `scripts/make_manifest.py`, `results/data/SHA256SUMS`
- [ ] **43.** `docs: contributor development guide` — *documentation* — `docs/DEVELOPMENT.md`
- [ ] **44.** `feat: documented full-sweep config (+ configs README)` — *enhancement* — `configs/full.yaml`, `configs/README.md`
- [ ] **45.** `perf: native-vs-python kernel micro-benchmark script` — *enhancement* — `scripts/bench_native.py`

## Single-owner edits to shared files (each file edited by exactly one PR)
- [ ] **46.** `docs: README — badges, reproduce-the-paper, verbatim results` — *documentation* — `README.md`
- [ ] **47.** `docs: CHANGELOG — Unreleased hardening entries` — *documentation* — `CHANGELOG.md`
- [ ] **48.** `feat: membench verify subcommand (offline integrity check)` — *enhancement* — `src/membench/cli.py`, `tests/test_cli_verify.py`
- [ ] **49.** `build: pyproject — coverage config + ruff rule expansion` — *enhancement* — `pyproject.toml`
- [ ] **50.** `docs: citation/DOI metadata (Zenodo) + citing guide` — *documentation* — `.zenodo.json`, `docs/CITING.md`

---

### Prior open PRs (reviewed, left OPEN for the maintainer)
- **#45** feat/dashboard — *changes needed* (CI red: ruff import-sort + E501; fix is mechanical).
- **#46** bump actions/setup-dotnet 4→5 — *merge* (CI green; Node24 satisfied by runner).
- **#47** bump Jimver/cuda-toolkit — *merge* (consumer CUDA job green).
- **#48** bump @types/node 25→26 in /tools/seed — *merge* (dev-only; typecheck green).

The new 50 branches avoid all paths touched by these PRs.
