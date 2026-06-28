r"""Distractor-scatter depth generator: spread a reasoning chain across areas.

The plain synthetic dataset (:mod:`membench.datasets.synthetic`) keeps a whole
depth-``d`` chain inside a *single* engineering area, so a similarity arm at least
retrieves the right neighbourhood (it just packs the wrong nodes from it). Real
multi-hop reasoning is rarely so tidy: the fact that justifies hop ``k`` often
lives in a *different* domain than hop ``k - 1`` (the auth rule that governs an
export feature, the billing invariant a notification touches). This module dials
that **scatter** -- how many distinct distractor areas a single chain is sprinkled
across -- as an explicit, depth-orthogonal axis.

Model
-----
A chain of depth ``d`` has ``d + 1`` nodes (entry, ``d - 1`` policies, governing
rule). With ``scatter = s`` distinct distractor areas, the nodes are assigned
**round-robin** to a pool of ``s + 1`` distinct areas (the target area first, so
the entry node still resembles the query). The number of distinct areas a chain
spans is therefore

.. math::

    A(d, s) = \min(s + 1,\; d + 1),

monotonically non-decreasing in ``s`` until it saturates at ``d + 1`` (one area
per hop). The number of *domain boundaries* a link-follower must bridge is
``sigma = A(d, s) - 1``.

Under the benchmark's structured-recovery model (each independent hop is recovered
with probability ``q``, giving ``p_struct(d) = q^d`` -- see
:mod:`membench.theory.recovery` and ``structured_recovery``), scattering adds an
independent per-boundary recovery factor ``p in (0, 1]`` for each domain crossed:

.. math::

    p_{\mathrm{struct}}(d, s) = q^{\,d} \cdot p^{\,\sigma},
    \qquad \sigma = \min(s, d).

So ``scatter`` is the empirical knob for the ``p^\sigma`` penalty: ``s = 0``
reproduces the single-area baseline (``sigma = 0``), and each extra area multiplies
recovery by another ``p``. This multiplicative compounding of independent hop /
boundary success probabilities is the standard error-compounding model for
multi-hop reasoning, and the distractor-scatter construction follows the
distractor setting of multi-hop QA benchmarks where supporting facts are dispersed
among same-domain distractors.

References
----------
.. [1] Z. Yang, P. Qi, S. Zhang, et al. "HotpotQA: A Dataset for Diverse,
   Explainable Multi-hop Question Answering." EMNLP 2018. (Distractor setting:
   gold supporting facts dispersed among topically similar distractor paragraphs.)
.. [2] H. Trivedi, N. Balasubramanian, T. Khot, A. Sabharwal. "MuSiQue: Multihop
   Questions via Single-hop Question Composition." TACL 2022. (Connected reasoning
   across documents; difficulty rises as hops cross documents.)
.. [3] N. F. Liu, K. Lin, J. Hewitt, et al. "Lost in the Middle: How Language
   Models Use Long Contexts." TACL 2024. (Dispersed evidence degrades recovery.)
"""

from __future__ import annotations

import random

from membench.datasets.synthetic import AREAS, Area
from membench.types import MemoryItem, Scorer, SpecVariant, Task

__all__ = ["generate_scatter_tasks"]

_DATASET = "synthetic-scatter"
_REPO_REF = "synthetic://scatter-bench"

# Query-irrelevant filler nodes (a shared haystack independent of any area).
_FILLER: tuple[str, ...] = (
    "telemetry pipeline ingestion buffer",
    "design system token palette spacing",
    "onboarding checklist progress milestone",
    "localization string catalogue plural",
    "infrastructure terraform module registry",
    "changelog release notes versioning",
)

_STOP = {"the", "a", "to", "so", "can", "their", "with", "and", "of", "in", "for"}


def _keywords(area: Area) -> list[str]:
    """Return the query-overlapping keywords for an area, de-duplicated.

    Parameters
    ----------
    area
        The engineering area to draw feature/name/action vocabulary from.

    Returns
    -------
    list of str
        Order-preserving, de-duplicated content words (stopwords and very short
        tokens removed).
    """
    raw = f"{area.feature} {area.name} {area.action}".lower().split()
    seen: list[str] = []
    for word in raw:
        if word not in _STOP and len(word) > 2 and word not in seen:
            seen.append(word)
    return seen


def _node_text(area: Area, hop: int, depth: int, *, governing: bool) -> str:
    """Render a chain node's text with geometrically decaying query overlap.

    Mirrors :func:`membench.datasets.synthetic._node_text`: hop 0 carries the
    full feature vocabulary; the share falls as ``(depth - hop + 1) / (depth + 1)``
    toward the governing node, instantiating the similarity-decay model
    ``s_i = s_0 rho^i``. Because a scattered node uses *its own* (possibly
    distractor) area's vocabulary, hops dropped into a foreign area are naturally
    dissimilar to the target-area query.

    Parameters
    ----------
    area
        The area whose vocabulary the node is written in.
    hop
        Zero-based position along the chain (0 is the entry node).
    depth
        Chain depth ``d``; the chain has ``depth + 1`` nodes.
    governing
        Whether this node is the terminal governing rule (carries the guard
        identifier compliance scoring targets).

    Returns
    -------
    str
        The node's natural-language content.
    """
    kws = _keywords(area)
    fraction = (depth - hop + 1) / (depth + 1)
    floor = 0 if governing else 1
    count = max(floor, round(len(kws) * fraction))
    sel = " ".join(kws[:count])
    if governing:
        head = f"{sel}. " if sel else ""
        return (
            f"{head}RULE: every operation in this path must call {area.guard}() before proceeding."
        )
    if hop == 0:
        return f"The {sel} surface is where users begin."
    policy = area.policies[hop % len(area.policies)]
    return f"{sel} {policy} policy layer, refining an upstream constraint it derives from."


def _scatter_chain(
    target: Area,
    scatter_areas: list[Area],
    depth: int,
    prefix: str,
) -> tuple[list[MemoryItem], str]:
    """Build a depth-``d`` chain whose hops are scattered round-robin across areas.

    Parameters
    ----------
    target
        The query-aligned area; always carries the entry node (hop 0).
    scatter_areas
        ``scatter`` further distinct areas the chain is sprinkled across.
    depth
        Chain depth ``d``; the chain has ``depth + 1`` nodes.
    prefix
        Per-task id prefix; chain node ids are ``f"{prefix}-hop{h}"``.

    Returns
    -------
    items : list of MemoryItem
        The ``depth + 1`` chain nodes, each tagged with its ``area`` slug and an
        explicit forward ``edges`` link (the last edge is ``constrains``).
    governing_id : str
        The terminal governing rule node's id (the gold decision).
    """
    pool = [target, *scatter_areas]
    ids = [f"{prefix}-hop{h}" for h in range(depth + 1)]
    items: list[MemoryItem] = []
    for hop in range(depth + 1):
        area = pool[hop % len(pool)]
        governing = hop == depth
        edges: list[tuple[str, str]] = []
        if hop < depth:
            rel = "constrains" if hop == depth - 1 else "derived_from"
            edges = [(ids[hop + 1], rel)]
        node_type = "entry" if hop == 0 else ("governing" if governing else "policy")
        items.append(
            MemoryItem(
                ids[hop],
                _node_text(area, hop, depth, governing=governing),
                metadata={"node_type": node_type, "area": area.slug, "edges": edges},
            )
        )
    return items, ids[-1]


def generate_scatter_tasks(
    depths: tuple[int, ...] = (1, 2, 3),
    scatter: int = 0,
    per_depth: int = 10,
    seed: int = 0,
    distractor_areas: int = 6,
) -> list[Task]:
    """Generate depth-controlled tasks whose chains are scattered across areas.

    For each depth and task index one area's chain carries the governing rule, but
    the chain's ``depth + 1`` hops are assigned round-robin to a pool of
    ``scatter + 1`` distinct areas (so the chain spans
    ``min(scatter + 1, depth + 1)`` areas). Same-area noise nodes and shared filler
    form the haystack, and the whole corpus is shuffled with a seeded RNG.

    Parameters
    ----------
    depths
        Chain depths to emit (each must be ``>= 1``).
    scatter
        Number of distinct *distractor* areas to spread each chain across. ``0``
        keeps every hop in the single target area (the baseline); larger values
        raise the ``p^sigma`` scatter penalty with ``sigma = min(scatter, depth)``.
        Must satisfy ``0 <= scatter <= len(AREAS) - 1`` so the pool stays distinct.
    per_depth
        Number of tasks to emit per depth.
    seed
        Seed for the corpus-shuffling RNG; identical seeds give identical output.
    distractor_areas
        How many *other* (non-chain) areas contribute single distractor nodes to
        the haystack.

    Returns
    -------
    list of Task
        Depth-labelled ``STRIPPED`` / ``COMPLIANCE`` tasks on the
        ``"synthetic-scatter"`` dataset.

    Raises
    ------
    ValueError
        If ``scatter`` is negative or exceeds ``len(AREAS) - 1``, or if any depth
        is ``< 1``.
    """
    if scatter < 0:
        raise ValueError("scatter must be >= 0")
    if scatter > len(AREAS) - 1:
        raise ValueError(f"scatter must be <= {len(AREAS) - 1} (one node per distinct area)")
    rng = random.Random(seed)
    tasks: list[Task] = []
    for depth in depths:
        if depth < 1:
            raise ValueError("depth must be >= 1")
        for i in range(per_depth):
            target_idx = (i + depth) % len(AREAS)
            target = AREAS[target_idx]
            scatter_areas = [AREAS[(target_idx + 1 + k) % len(AREAS)] for k in range(scatter)]
            prefix = f"scatter-d{depth}-{i:02d}"
            chain_items, governing_id = _scatter_chain(target, scatter_areas, depth, prefix)
            corpus: list[MemoryItem] = list(chain_items)
            # Same-area noise so a similarity arm is genuinely lured into each
            # area the chain touches (the source of the empirical scatter penalty).
            for area in (target, *scatter_areas):
                for noise in range(2):
                    corpus.append(
                        MemoryItem(
                            f"{prefix}-noise-{area.slug}-{noise}",
                            _node_text(area, max(1, noise + 1), depth, governing=False),
                            metadata={"node_type": "distractor", "area": area.slug},
                        )
                    )
            # Single distractor nodes from areas the chain does not touch.
            used = {target.slug, *(a.slug for a in scatter_areas)}
            others = [a for a in AREAS if a.slug not in used]
            rng.shuffle(others)
            for n, area in enumerate(others[:distractor_areas]):
                corpus.append(
                    MemoryItem(
                        f"{prefix}-distractor-{area.slug}-{n}",
                        _node_text(area, 0, depth, governing=False),
                        metadata={"node_type": "distractor", "area": area.slug},
                    )
                )
            for n, filler in enumerate(_FILLER):
                corpus.append(MemoryItem(f"{prefix}-filler-{n}", filler))
            rng.shuffle(corpus)
            query = (
                f"Add a {target.feature} to the {target.name} module so users can {target.action}."
            )
            tasks.append(
                Task(
                    task_id=f"scatter-d{depth}-{i:02d}",
                    dataset=_DATASET,
                    query=query,
                    repo_ref=_REPO_REF,
                    memory_corpus=tuple(corpus),
                    governing_decisions=(governing_id,),
                    depth=depth,
                    spec_variant=SpecVariant.STRIPPED,
                    scorer=Scorer.COMPLIANCE,
                )
            )
    return tasks
