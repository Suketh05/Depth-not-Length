# Glossary of core terms

A precise, source-grounded reference for the vocabulary used throughout this
project — the paper *"Depth, Not Length: A Benchmark and Theory of Memory,
Retrieval, and Decision Compliance in LLM Coding Agents"*, the
[methodology note](./METHODOLOGY.md), the
[claim → code → artifact map](../paper/CLAIMS.md), and the `membench` source tree.

Each entry links to where the term is defined or computed. Every benchmark number
is copied verbatim from an integrity source and labelled by provenance:

- **paper-reported** — transcribed from the paper LaTeX / the verified paper digest.
- **repo-measured** — read from a committed results file ([`results/METRICS.md`](../results/METRICS.md)).

Paper-reported and repo-measured figures are kept separate on purpose: the paper's
headline "real code" numbers pool **dcbench + swebench only**, whereas the
repo-measured `@all` cut in `results/METRICS.md` pools **synthetic + dcbench +
swebench**, so the two are *not* interchangeable.

---

## Causal depth `d`

The number of causal hops between the task surface (the code or question the agent
is handed) and the governing decision it must honour. Operationally the benchmark
defines depth as a **strip-count**: `d=1` states the constraint in the query
itself (a control), `d=2` strips it into one corpus node, and `d=3` splits it into
a constraint node plus a linked justification node (see "Depth = strip-count" in
[METHODOLOGY.md](./METHODOLOGY.md)). Depth is the independent variable of the whole
study — every depth-resolved metric is reported at `d=1,2,3`. The synthetic suite
dials depth directly and makes node→query similarity decay geometrically along the
chain.

## Decision-compliance factorization `P_comply(d) = P_ret(d) · κ(d)`

The central identity of the theory: the probability that an agent's edit complies
with the governing decision factors into the probability that retrieval *surfaces*
that decision, `P_ret(d)`, times the probability the agent *acts on* it once
surfaced, `κ(d)`. It cleanly separates a retrieval failure from a use failure, and
because `κ ≤ 1` it implies a hard ceiling `P_comply ≤ κ` (the attenuated-transfer
corollary, `∂P_comply/∂P_ret = κ`). This is equation `eq:factor` of the paper; the
two factors are measured by [`metrics/compliance.py`](../src/membench/metrics/compliance.py)
and [`metrics/retrieval.py`](../src/membench/metrics/retrieval.py).

## Retrieval recall `P_ret`

The probability that the governing decision (the whole causal chain, end to end)
is present in the retrieved context at a fixed budget. It is the *first* factor of
the compliance factorization and the axis on which a typed store is meant to win.
On real code (paper-reported, pooled dcbench+swebench, Claude) the typed store
ranks first at recall **0.667**, 0.032 above dense at 0.635. Recall is scored by
[`metrics/retrieval.py`](../src/membench/metrics/retrieval.py); see the per-dataset
repo-measured values below under *provenance tiers*.

## Use-factor / compliance ceiling `κ`

The "use factor": the conditional probability that the agent complies *given that*
the governing decision was retrieved, `κ = P(comply | retrieved)` — equivalently
`κ ≈ P_comply / P_ret`. It quantifies how much surfaced context an agent actually
converts into a correct edit, and it imposes the compliance ceiling `P_comply ≤ κ`.
The paper reserves the phrase "use ceiling" for the empirical finding that every
similarity retriever's `κ` is trapped in a narrow band on real code, while the
typed store clears it (see the next entries). Synthetic `κ` is near 1 (≈0.95–0.99,
paper-reported), so use is not the bottleneck there — retrieval is.

## Similarity decay ceiling `ρ^{d(d+1)/2}`

The super-geometric ceiling on similarity retrieval's full-chain recall. Under a
geometric per-hop decay `s_i = s_0·ρ^i` (`0 < ρ < 1`), recovering all `d` hops
gives `P_ret^sim(d) = s_0^d · ρ^{d(d+1)/2}`; the triangular exponent
`d(d+1)/2` means recall falls off faster than any fixed exponential rate, so on
deep chains a similarity retriever is reduced to guessing the very links that hold
the task together. This is Proposition 1, the *depth ceiling*, derived and tested
in [`theory/recovery.py`](../src/membench/theory/recovery.py); the decay law
`s_i = s_0·ρ^i` itself lives in [`theory/decay.py`](../src/membench/theory/decay.py)
and is fit from real embeddings in `analysis/decay_fit.py`.

## Structured recovery `q^d`

The recovery curve for a store that *follows stored links* instead of guessing by
resemblance: each typed edge is traversed correctly with per-edge fidelity
`q ∈ (0,1]`, so the whole chain is recovered with probability `q^d`. Because `q` is
close to one and independent of whether two linked facts resemble each other,
`q^d` decays only gently, and its ratio to the similarity curve grows without bound
with depth. Defined alongside the similarity ceiling in
[`theory/recovery.py`](../src/membench/theory/recovery.py).

## Crossover depth `d⋆`

The smallest causal depth at which structured recovery's net gain clears the
per-chain overhead of building and maintaining typed links — formally the smallest
integer `d` with `g(d) = q^d − s_0^d·ρ^{d(d+1)/2} > τ`, where `τ = W/V` is the
overhead-to-value ratio. Because `g` rises, peaks, then falls for `q < 1`, the win
is an interval `[d⋆, d_upper]`, not a half-line, so the project solves for it
numerically rather than asserting monotonicity (see the honesty note in
[`theory/crossover.py`](../src/membench/theory/crossover.py)). At the calibrated
synthetic operating point (`s_0=0.70, ρ=0.67, q=0.97`, paper-reported) the
closed-form crossover is `d⋆=1`, and the reported `d⋆=2` is flagged as calibrated
post-hoc for the margin band `0.50 < τ ≤ 0.79`.

## Supersession edges

A typed `supersedes` edge marks one decision as having replaced an earlier one, so
"which decision is currently in force" becomes a graph dereference rather than a
similarity guess. The supersession benchmark scores how often the *current*
decision is ranked first among a superseded pair (chance = 50%). Paper-reported:
the typed store ranks the current decision first **92.3%** of the time, versus a
similarity band of **64–69%** (a 23–28 point lead). `SUPERSEDES` is one of the
relationship types in the typed store,
[`retrieval/graph/store.py`](../src/membench/retrieval/graph/store.py).

## Typed links / context graph

The typed-edge property graph that mirrors Brief's production context graph: nodes
are typed records (`entity_objects`) and edges are *typed, directed* links
(`entity_link`) drawn from a fixed catalogue — `constrains`, `supersedes`,
`implements` — each with a confidence weight. Endpoints are FK-enforced so the
graph can never hold a dangling link, exactly as Postgres enforces in Brief. The
store is in [`retrieval/graph/store.py`](../src/membench/retrieval/graph/store.py)
and the bounded multi-hop walk over it (Brief's `traverseEntityGraph`, default two
hops, clamped 1–3) is in
[`retrieval/graph/traversal.py`](../src/membench/retrieval/graph/traversal.py).
Following the `constrains` link is precisely the hop a flat retriever cannot make.

## The fairness lock

The experimental control that makes the comparison about the memory mechanism and
nothing else: every memory **arm** implements one contract — `write` and
`retrieve(query, budget)` — and the harness swaps the arm while holding everything
else fixed (the same tasks, the same per-call token budget matched across arms to
~1%, the same deterministic stub agent). See "Arms" and "Budget" in
[METHODOLOGY.md](./METHODOLOGY.md). Without the lock, a win could be attributed to
a bigger budget or a different agent rather than to typed retrieval.

## Measured / modeled / vendor provenance tiers

The three-way labelling every cell in the results matrix carries, so no derived or
cited figure is ever shown as a fresh measurement.
[`results/METRICS.md`](../results/METRICS.md) tags each of its 387 cells as one of:
**measured** (our harness data; 349 cells), **vendor** (a competitor's published
number, cited; 25 cells), or **modeled** (derived by a stated method, not run; 13
cells). For example, the repo-measured real-code recall cells are
`recall@dcbench` Brief **0.7698** and `recall@swebench` Brief **0.5556** (both
*measured*, Claude); a competitor's `Mem0_headline_score` **66.9** is *vendor*
(arXiv:2504.19413); and `retrieval_latency_ms_Brief` **120** is *modeled*. Keeping
the tiers visible is how the project keeps repo-measured and paper-reported numbers
from being conflated.

## Merge-ready correctness

A stricter task outcome than per-decision compliance: a response is *merge-ready*
only if it is non-empty, is not a refusal, and honours **every** governing
decision the task names — one unmet decision makes it not merge-ready. It is the
binary the cost-to-correct metric treats as success, defined in
[`metrics/correctness.py`](../src/membench/metrics/correctness.py). (The harness
scores the identifier contract, not a real test suite or diff review, so it is a
proxy for true merge-readiness.) Repo-measured pooled `merge_ready@all`: Brief
**0.6574**, dense 0.5694, BM25 0.588 (Claude).

## Return on Tokens (RoT)

The token-economics payoff metric: correct, shipped-right work produced per token
spent (per-task `RoT_i = comply_i / T_i`), the empirical counterpart of the
theoretical token-economics model. Per-call tokens are a *reported control*, never
an optimisation target — [`metrics/cost.py`](../src/membench/metrics/cost.py)
measures cost, it does not decide retries. Repo-measured pooled
`return_on_tokens@all`: Brief **0.5725**, dense 0.496, BM25 0.5214 (Claude).

## The memory "arms"

The nine interchangeable memory conditions the harness swaps under the fairness
lock: `none`, `random_context`, `bm25`, `tfidf`, `dense`, `hybrid_rrf`,
`rerank_ce`, `raptor`, and `brief_graph_3hop` (the typed decision graph with
governance edges and 3-hop traversal, referred to as "Brief"). `none` is the
context-free control that exposes the irreducible compliance floor; the similarity
arms (`bm25` through `raptor`) are the retrievers that hit the use ceiling; the
typed-graph arm is the structured alternative under test. The retrieval arms live
under [`src/membench/retrieval/`](../src/membench/retrieval/) and the typed-graph
arm under [`src/membench/retrieval/graph/`](../src/membench/retrieval/graph/).

---

### Source map

| Concept | Theory / code | Paper map |
|---|---|---|
| Compliance factorization, recall, `κ` | [`metrics/compliance.py`](../src/membench/metrics/compliance.py), [`metrics/retrieval.py`](../src/membench/metrics/retrieval.py) | [`paper/CLAIMS.md`](../paper/CLAIMS.md) |
| `ρ^{d(d+1)/2}`, `q^d` | [`theory/recovery.py`](../src/membench/theory/recovery.py), [`theory/decay.py`](../src/membench/theory/decay.py) | depth-ceiling claim |
| Crossover `d⋆` | [`theory/crossover.py`](../src/membench/theory/crossover.py) | crossover claim |
| Typed links, supersession, traversal | [`retrieval/graph/`](../src/membench/retrieval/graph/) | governing-decision-via-link claim |
| Merge-ready, Return on Tokens | [`metrics/correctness.py`](../src/membench/metrics/correctness.py), [`metrics/cost.py`](../src/membench/metrics/cost.py) | Return-on-Tokens claim |
| Provenance tiers | [`results/METRICS.md`](../results/METRICS.md) | — |

See also the [methodology note](./METHODOLOGY.md) for how measurement feeds the
theory (`s_i = s_0·ρ^i` → `q` → predicted `d⋆`), and the
[claim → code → artifact map](../paper/CLAIMS.md) for the full traceability table.
