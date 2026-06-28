# Statistical methods reference

This document maps every statistical method used in *Depth, Not Length* to its
implementing module under [`src/membench/stats/`](../src/membench/stats), and to the
paper claim and figure each method backs. The claim → code → artifact spine lives in
[`paper/CLAIMS.md`](../paper/CLAIMS.md); this file is the per-method companion to it.

For each module you get: **what question it answers**, its **inputs and outputs** (the
public functions, transcribed from the module's `__all__`), and **which claim/figure it
supports**. Read the modules themselves for the full docstrings and the exact math.

## Provenance of numbers in this document

Numbers below are quoted from two strictly separate sources, never merged:

- **repo-measured** — produced by real-backend runs and kept as durable artifacts.
  Cited to [`paper/measured/summary.md`](../paper/measured/summary.md),
  [`results/METRICS.md`](../results/METRICS.md), [`results/EVALS.md`](../results/EVALS.md),
  or [`results/FIGURES.md`](../results/FIGURES.md).
- **paper-reported** — figures transcribed verbatim from the paper LaTeX
  (`depth_not_length_FIXED.tex`). These are the values that appear in the manuscript's
  tables and figures.

Where a method's headline statistic is genuinely a measured value it is labelled
*repo-measured*; where it is a manuscript figure it is labelled *paper-reported*. The two
are kept in distinct sentences, consistent with the provenance rules in
[`paper/measured/README.md`](../paper/measured/README.md) (measured and vendor/paper
numbers never share a cell).

## The design these methods assume

Two facts about the benchmark design drive nearly every choice here, and are stated in
[`docs/METHODOLOGY.md`](./METHODOLOGY.md):

1. **Paired / clustered observations.** Every memory arm is evaluated on the *same*
   tasks (a repeated-measures design), and rows within a task are dependent. Comparisons
   are therefore paired, and resampling is done over whole tasks (clusters), not rows.
2. **Bounded, often-binary outcomes.** Compliance and merge-ready correctness are
   Bernoulli, measured on a few dozen tasks per cell — small *n*, where normal
   approximations misbehave near 0 and 1. This motivates exact/non-parametric tools
   (Beta–Binomial, BCa, permutation, Cliff's δ).

## Method → module → claim map

| Method | Module | Primary claim / figure it backs |
|---|---|---|
| BCa & cluster bootstrap CIs | [`bootstrap.py`](../src/membench/stats/bootstrap.py) | Every headline number ships with an interval (forest plot `fig:forest`) |
| Bayesian superiority `P(Brief > comp)` | [`bayes.py`](../src/membench/stats/bayes.py) | "Brief beats competitor X with probability *p*" → competitor matrix / posteriors `fig:posteriors` |
| Paired permutation / Wilcoxon / McNemar | [`hypothesis.py`](../src/membench/stats/hypothesis.py) | Per-contrast significance → forest plot `fig:forest` |
| Holm / BH / Friedman / Nemenyi | [`multiple_comparisons.py`](../src/membench/stats/multiple_comparisons.py) | "Rankings differ significantly across arms" → critical-difference diagram `fig:cd` |
| Cluster-robust logistic `comply ~ structured*depth` | [`regression.py`](../src/membench/stats/regression.py) | The depth thesis in one coefficient → depth-slope `fig:slopebar` |
| Empirical crossover depth `d⋆` with CI | [`changepoint.py`](../src/membench/stats/changepoint.py) | "Predicted `d⋆` ≈ measured `d⋆`" → crossover figure `fig:crossover` |
| Mediation (causal decomposition) | [`mediation.py`](../src/membench/stats/mediation.py) | "The win runs *through* retrieval" → mediation result `fig:mediation` |
| Stochastic dominance & Pareto frontier | [`dominance.py`](../src/membench/stats/dominance.py) | "Joint accuracy + cost win" → cost-frontier `fig:pareto` |
| Effect sizes (Cohen's h, Cliff's δ, …) | [`effect_size.py`](../src/membench/stats/effect_size.py) | Magnitude of the win → Cohen's-h bars `fig:cohensh` |
| Decay-law model selection (AIC/BIC) | [`model_selection.py`](../src/membench/stats/model_selection.py) | Justifies `s_i = s₀·ρ^i` over alternatives → phase diagram `fig:phase` |
| Power / sample size | [`power.py`](../src/membench/stats/power.py) | Whether the design can detect the reported effects |

---

## Bootstrap confidence intervals — [`bootstrap.py`](../src/membench/stats/bootstrap.py)

**Question it answers.** How uncertain is a reported point estimate (a rate, an RoT, a
cost-to-correct ratio), given the clustered design?

**Inputs / outputs.**

- `bca_ci(values, statistic=mean, *, confidence=0.95, n_resamples=2000, seed=0)` →
  `ConfidenceInterval(point, low, high, confidence)`. The bias-corrected and accelerated
  bootstrap: it corrects the percentile interval for median bias (`z0`) and skew (the
  jackknife acceleration `a`), giving second-order-accurate bounds — important for rates
  pinned near 0/1 and for ratios. Falls back to percentile bounds if the acceleration is
  degenerate.
- `cluster_bootstrap_ci(values, clusters, statistic=mean, …)` → `ConfidenceInterval`.
  Resamples whole *clusters* (tasks) with replacement rather than individual rows — the
  correct bootstrap when within-task rows are dependent.
- `percentile_ci(...)` → `ConfidenceInterval`. The basic percentile interval (baseline).

**What it backs.** This is the interval machinery behind "every headline number ships
with an interval and a test" (module docstring of
[`src/membench/stats/__init__.py`](../src/membench/stats/__init__.py)). It feeds the
forest plot of per-contrast effects (`fig:forest`). The paper's stated protocol uses
*BCa bootstrap with 10⁴ resamples* alongside Wilson intervals (paper-reported
methodology). It is also the resampling primitive reused by the crossover CI below.

## Bayesian superiority — [`bayes.py`](../src/membench/stats/bayes.py)

**Question it answers.** With what probability is Brief's rate actually higher than a
competitor's, given the small per-cell *n*? This is the directly defensible comparison
statement `P(rate_Brief > rate_competitor)`, rather than a *p*-value.

**Inputs / outputs.**

- `beta_binomial_posterior(successes, trials, *, prior=(0.5, 0.5))` → `BetaPosterior`.
  Conjugate update: a `Beta(a, b)` prior with `s` successes in `n` trials gives a
  `Beta(a+s, b+n−s)` posterior. The default is Jeffreys' `Beta(½, ½)` objective prior.
- `BetaPosterior.mean`, `.credible_interval(cred=0.95)`, `.sample(n, seed)` — posterior
  mean, equal-tailed credible interval, and seeded draws.
- `prob_superiority(a, b, *, n_samples=200_000, seed=0)` → `float`. Monte-Carlo
  `P(rate_a > rate_b)` from the two posteriors.

**What it backs.** The claim row "Brief beats competitor X with probability *p*"
([`paper/CLAIMS.md`](../paper/CLAIMS.md)) and the Beta–Binomial posterior-density figure
`fig:posteriors`. The manuscript reports (paper-reported) a synthetic depth-3 compliance
superiority of `P(θ_B > θ_C) ≈ 0.81` and uses a Jeffreys `Beta(½, ½)` prior throughout.

## Paired hypothesis tests — [`hypothesis.py`](../src/membench/stats/hypothesis.py)

**Question it answers.** For a single arm-vs-arm contrast on the same tasks, is the
difference statistically significant?

**Inputs / outputs.**

- `paired_permutation_test(a, b, *, n_resamples=10_000, seed=0)` → `TestResult`. The
  exact sign-flip randomization test on paired differences (the IR-standard test; no
  distributional assumption). Two-sided *p* = `(1 + #{|t*| ≥ |t_obs|}) / (1 + R)`.
- `wilcoxon_signed_rank(a, b)` → `TestResult`. Non-parametric paired rank test (SciPy).
- `mcnemar_test(correct_a, correct_b)` → `TestResult`. Exact test for paired *binary*
  outcomes (each arm correct/incorrect per task), on the discordant pairs only.

Each returns `TestResult(statistic, p_value, method)`.

**What it backs.** The per-contrast significances in the forest plot `fig:forest`. The
manuscript reports (paper-reported) the depth-3 Brief-vs-dense contrast as 40/40 vs 27/40
correct, `z = 3.94`, `p < 10⁻⁴`.

## Multiple comparisons & the Demšar protocol — [`multiple_comparisons.py`](../src/membench/stats/multiple_comparisons.py)

**Question it answers.** With 20+ arms across several datasets, (a) do the arms differ at
all, (b) which pairwise rank gaps are real, and (c) how do we keep the family-wise / FDR
error under control?

**Inputs / outputs.**

- `holm_adjust(pvalues)` → adjusted *p*-values (Holm–Bonferroni step-down, FWER control).
- `benjamini_hochberg_adjust(pvalues)` → adjusted *p*-values (BH step-up, FDR control).
- `friedman_test(scores)` → `FriedmanResult(statistic, p_value)`. The non-parametric
  omnibus over a (blocks × treatments) matrix — do the arms differ across datasets/tasks?
- `average_ranks(scores, *, higher_is_better=True)` → mean rank per arm (rank 1 = best).
- `nemenyi_critical_difference(k, n_blocks, *, alpha=0.05)` → the critical difference
  `CD = q_α·√(k(k+1)/(6N))`; mean-rank gaps larger than `CD` are significant. Nemenyi
  `q₀.₀₅` values are tabulated for `k = 2..20` (Demšar 2006, Table 5).

**What it backs.** The claim "Rankings differ significantly across arms"
([`paper/CLAIMS.md`](../paper/CLAIMS.md)) and the critical-difference diagram `fig:cd`.
The manuscript reports (paper-reported) a synthetic Friedman test with `k = 8` arms
(7 df) exceeding a critical value of `14.07`, with Brief at mean rank ≈ `2.55`.

## Cluster-robust logistic regression — [`regression.py`](../src/membench/stats/regression.py)

**Question it answers.** Two at once: how large is the structured-memory advantage, and —
the headline — *does that advantage grow with depth?*

**Inputs / outputs.**

- `fit_compliance_on_arm_depth(compliance, structured, depth, task_ids)` → `LogisticFit`.
  Fits `logit Pr(comply) = β₀ + β₁·structured + β₂·depth + β₃·(structured×depth)` with
  **task-cluster-robust** standard errors (robust to within-task correlation). `structured`
  is 1 for the typed-graph arm and 0 for a similarity arm.
- `LogisticFit` exposes `params`, `pvalues`, `n_obs`, `n_clusters`, plus convenience
  properties: `.structured_advantage` (β₁), `.depth_interaction` (β₃, the per-hop growth
  of the advantage), and `.interaction_p` (the depth-thesis test).

> Note on naming: the paper phrases this model as `compliance ~ arm * depth`; the code
> operationalises "arm" as the binary `structured` indicator (typed graph vs. similarity),
> so the interaction term is `structured:depth`.

**What it backs.** A positive, significant `β₃` *is* the depth thesis in one coefficient.
It underpins the depth-slope result drawn in `fig:slopebar` / the arm×depth heatmap
`fig:armdepth`. The manuscript reports (paper-reported) Brief's compliance slope from
`d=1` to `d=3` as `−0.075` (the least decay of any arm) versus similarity arms at `−0.125`
to `−0.200`.

## Empirical crossover depth — [`changepoint.py`](../src/membench/stats/changepoint.py)

**Question it answers.** At what depth does structured memory's recovery *overtake*
similarity's, and how certain is that location? It turns the theory's predicted `d⋆` into
a measured estimate with a confidence interval.

**Inputs / outputs.**

- `interpolated_crossing(x, y, *, level=0.0)` → `float | None`. The first `x` at which the
  advantage curve `y` crosses `level` upward, by linear interpolation (or `None`).
- `bootstrap_crossover_ci(depths, structured_by_depth, similarity_by_depth, *,
  confidence=0.95, n_resamples=2000, seed=0)` → `CrossoverEstimate(d_star, ci_low,
  ci_high, exists_fraction)`. The point estimate locates the crossing of the *mean*
  structured-minus-similarity advantage curve; the CI resamples tasks within each depth
  and relocates the crossing on each replicate. `exists_fraction` is the share of
  replicates that actually cross (so a fragile crossover is visible, not hidden).

**What it backs.** The claim "Predicted `d⋆` ≈ measured `d⋆`"
([`paper/CLAIMS.md`](../paper/CLAIMS.md)) and the crossover figure `fig:crossover` (the
theory side is the phase diagram `fig:phase`). The manuscript reports (paper-reported) a
predicted `d⋆ = 2` (calibrated; the closed-form smallest-crossing depth is `d⋆ = 1`). The
controlled depth result is repo-measured: at `d=3` on synthetic, **Brief 1.00 vs best
similarity 0.70** ([`paper/measured/summary.md`](../paper/measured/summary.md)).

## Mediation analysis — [`mediation.py`](../src/membench/stats/mediation.py)

**Question it answers.** Is Brief's advantage *mechanistic* — does it win **because** it
retrieves the governing chain, rather than for some incidental reason?

**Inputs / outputs.**

- `mediation_analysis(treatment, mediator, outcome)` → `MediationResult(total_effect,
  direct_effect, indirect_effect, proportion_mediated)`. Linear-probability (OLS)
  mediation appropriate for binary outcomes: with treatment `T` (structured vs.
  similarity), mediator `M` (governing chain recovered), outcome `Y` (complied), the
  total effect is `E[Y|T=1]−E[Y|T=0]`, the indirect effect is `a·b` (with `a` the
  effect of `T` on `M` and `b` the coefficient of `M` in `Y ~ T + M`), and the
  proportion mediated is `indirect / total`. A proportion near 1 means the advantage runs
  almost entirely through retrieval.
- `mutual_information(x, y)` → bits — how much knowing whether the chain was retrieved
  tells you about compliance.

**What it backs.** The claim "The win runs *through* retrieval (mechanism)"
([`paper/CLAIMS.md`](../paper/CLAIMS.md)) and the mediation figure `fig:mediation`. That
figure's total effect is the synthetic Brief-vs-dense compliance gap (paper-reported:
Brief `0.933` vs dense `0.858`, i.e. `+0.075`).

## Stochastic dominance & Pareto frontier — [`dominance.py`](../src/membench/stats/dominance.py)

**Question it answers.** Is the win *joint* — more accurate **and** cheaper — and is it
robust to looking past the means at the whole distribution?

**Inputs / outputs.**

- `dominates(a, b)` and `pareto_frontier(points)` over `(accuracy, cost)` points → the
  non-dominated set (higher accuracy, lower cost). An arm alone in the high-accuracy /
  low-cost corner wins outright.
- `first_order_dominates(a, b)` → whether arm `a`'s accuracy distribution is
  stochastically larger than `b`'s (CDF everywhere below) — stronger than comparing means.
- `second_order_dominates(a, b)` → the weaker risk-averse criterion on the integrated CDF.

**What it backs.** The claim "Joint accuracy + cost win"
([`paper/CLAIMS.md`](../paper/CLAIMS.md)) and the accuracy–cost Pareto frontier
`fig:pareto`. On that frontier the typed store sits at the corner (paper-reported:
compliance `0.933` at ≈ `1.39k` retrieval tokens, synthetic).

## Effect sizes — [`effect_size.py`](../src/membench/stats/effect_size.py)

**Question it answers.** Not *whether* a difference exists but *how large* it is — what a
"wins by a wide margin" claim actually rests on.

**Inputs / outputs.**

- `cliffs_delta(a, b)` → `CliffsDelta(delta, magnitude)`: ordinal dominance in `[−1, 1]`
  with Romano magnitude labels (negligible `<0.147`, small `<0.33`, medium `<0.474`,
  else large) — robust to the capped, non-normal outcomes here.
- `cohens_h(p1, p2)` → arcsine effect size for two proportions, `2·asin√p₁ − 2·asin√p₂`.
- `cohens_d(a, b)` → standardized mean difference (pooled SD).
- `common_language_effect_size(a, b)` → `P(X > Y)` (ties = 0.5).
- `odds_ratio(...)` → Haldane-corrected odds ratio of two success counts.

**What it backs.** The magnitude side of the win, drawn as the Cohen's-h bars
`fig:cohensh`. The manuscript reports (paper-reported) Brief-vs-best-similarity synthetic
Cohen's h ≈ `1.2`.

## Decay-law model selection — [`model_selection.py`](../src/membench/stats/model_selection.py)

**Question it answers.** Is the theory's assumed geometric decay `s_i = s₀·ρ^i` actually
the best description of measured similarity-vs-distance, or merely assumed?

**Inputs / outputs.**

- `compare_decay_models(distances, similarities)` → `dict[str, ModelFit]` for three
  candidates fitted to measured `(distance, similarity)` pairs and scored on a common
  Gaussian log-likelihood of log-space residuals (so AIC/BIC are comparable):
  **geometric** `s₀·ρ^i`, **power-law** `s₀·(i+1)^{−α}`, and **stretched-exponential**
  `s₀·e^{−(λi)^β}`.
- `ModelFit(name, params, rss, aic, bic)`; helpers `aic(rss, n, k)` and `bic(rss, n, k)`.

**What it backs.** Geometric winning (or tying) is the honest justification for the law
the theory uses; per [`docs/METHODOLOGY.md`](./METHODOLOGY.md), "AIC/BIC select the
geometric law over power-law / stretched-exponential alternatives". This supports the
`s_i = s₀·ρ^i` claim row in [`paper/CLAIMS.md`](../paper/CLAIMS.md) (the decay-fit figure)
and the phase diagram `fig:phase` whose axes are the fitted `(ρ, τ)`. The synthetic
operating point is paper-reported as `s₀ = 0.70`, `ρ = 0.67`, `q = 0.97`.

## Power & sample size — [`power.py`](../src/membench/stats/power.py)

**Question it answers.** Can the design actually detect the effects it reports — and how
many tasks per arm would a given effect need?

**Inputs / outputs.** (Two-proportion comparisons under a normal approximation with a
pooled null variance.)

- `two_proportion_power(p1, p2, n, *, alpha=0.05)` → power of a two-sided z-test at `n`
  per group.
- `required_sample_size(p1, p2, *, power=0.8, alpha=0.05)` → per-group `n` for the target
  power.
- `minimum_detectable_effect(p_baseline, n, *, power=0.8, alpha=0.05)` → smallest absolute
  rate gap detectable at `n` per group.

**What it backs.** The sample-size honesty of the protocol — stating "with N tasks per arm
we have 80% power to detect a 12-point compliance gap" rather than hand-waving — and the
explicit under-power caveat the paper flags for the swebench `d=3` cell (paper-reported:
`n = 12`, read as inconclusive). It does not back a headline accuracy claim; it bounds how
much the others can claim.

---

## See also

- [`paper/CLAIMS.md`](../paper/CLAIMS.md) — the full claim → code → artifact table.
- [`docs/METHODOLOGY.md`](./METHODOLOGY.md) — design, depth definition, and how
  measurement becomes theory.
- [`paper/measured/summary.md`](../paper/measured/summary.md) and
  [`paper/measured/README.md`](../paper/measured/README.md) — repo-measured headline
  results and their provenance rules.
- [`results/METRICS.md`](../results/METRICS.md), [`results/EVALS.md`](../results/EVALS.md),
  [`results/FIGURES.md`](../results/FIGURES.md) — metric definitions, eval breakdowns, and
  the figure index.

> Integrity: no number in this document was computed here. Repo-measured values are quoted
> from `paper/measured/` and `results/`; paper-reported values are transcribed from the
> manuscript. The two are kept separate. No fabricated data.
