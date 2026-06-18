"""Synthetic depth-controlled benchmark: the clean test of the depth thesis.

Real codebases hide decisions the way this dataset models them: you are handed a
feature request that names an *area* ("add CSV export to the analytics dashboard"),
and the governing rule ("every export must call ``withAuditLog()``") is terse,
phrased in unrelated vocabulary, and discoverable only because it is *linked* to
that area through one or more hops of derived policy. Similarity retrieval finds the
area-relevant intermediate notes (they look like the task) but, under a fixed token
budget, those intermediates crowd the dissimilar governing rule out of the context;
the more hops of policy sit between the task and the rule, the further the rule is
pushed down the ranking. A link-following memory walks the chain straight to the
rule regardless of depth.

This is a controlled experiment that isolates the mechanism the theory predicts
(``s_i = s_0 rho^i``): it is depth-dialed, fair (every arm sees identical tasks,
corpus, and budget), and not specific to Brief -- *any* link-following system wins
on it, which is exactly why it is an independent test rather than a Brief-tuned one.
The governing node carries a code identifier so compliance scoring has a concrete
target, and a ``constrains``/explicit-``edges`` chain so the graph arm can traverse.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from membench.types import MemoryItem, Scorer, SpecVariant, Task

__all__ = ["AREAS", "Area", "load_synthetic_tasks"]


@dataclass(frozen=True, slots=True)
class Area:
    """A realistic engineering area with a feature, action, guard rule, and policies."""

    slug: str
    name: str
    feature: str
    action: str
    guard: str  # the code identifier the governing rule mandates
    policies: tuple[str, ...]  # intermediate-policy vocabulary for this area


AREAS: tuple[Area, ...] = (
    Area(
        "export",
        "data export",
        "CSV export button",
        "download their data",
        "withAuditLog",
        ("retention", "streaming", "compliance", "redaction"),
    ),
    Area(
        "auth",
        "user authentication",
        "social sign-in flow",
        "log in with Google",
        "requireMfa",
        ("session", "tenancy", "revocation", "lockout"),
    ),
    Area(
        "billing",
        "billing",
        "usage-based plan selector",
        "upgrade their plan",
        "withIdempotencyKey",
        ("proration", "dunning", "tax", "webhook"),
    ),
    Area(
        "search",
        "search",
        "faceted filter sidebar",
        "filter results",
        "sanitizeQuery",
        ("relevance", "sharding", "throttle", "synonym"),
    ),
    Area(
        "uploads",
        "file uploads",
        "drag-and-drop uploader",
        "attach files",
        "scanForMalware",
        ("chunking", "quota", "mimecheck", "presign"),
    ),
    Area(
        "notifications",
        "notifications",
        "digest preferences page",
        "manage alerts",
        "respectQuietHours",
        ("batching", "channel", "optout", "locale"),
    ),
    Area(
        "caching",
        "caching",
        "dashboard widget cache",
        "load faster",
        "withCacheStampedeGuard",
        ("ttl", "invalidation", "warming", "tiering"),
    ),
    Area(
        "ratelimit",
        "rate limiting",
        "public API key form",
        "call the API",
        "withTokenBucket",
        ("burst", "quota", "backoff", "fairness"),
    ),
    Area(
        "reporting",
        "reporting",
        "scheduled report builder",
        "schedule a report",
        "withRowLevelSecurity",
        ("aggregation", "timezone", "delivery", "retention"),
    ),
    Area(
        "flags",
        "feature flags",
        "experiment targeting UI",
        "target a cohort",
        "logExposure",
        ("rollout", "sticky", "killswitch", "segment"),
    ),
)

# Filler vocabulary for distractor nodes (query-irrelevant haystack).
_FILLER = (
    "telemetry pipeline ingestion buffer",
    "design system token palette spacing",
    "onboarding checklist progress milestone",
    "localization string catalogue plural",
    "infrastructure terraform module registry",
    "changelog release notes versioning",
)


_STOP = {"the", "a", "to", "so", "can", "their", "with", "and", "of", "in", "for"}


def _keywords(area: Area) -> list[str]:
    """Query-overlapping keywords for an area (from feature/name/action), de-duped."""
    raw = f"{area.feature} {area.name} {area.action}".lower().split()
    seen: list[str] = []
    for word in raw:
        if word not in _STOP and len(word) > 2 and word not in seen:
            seen.append(word)
    return seen


def _node_text(area: Area, hop: int, depth: int, *, governing: bool) -> str:
    """Node text whose query-keyword overlap decays geometrically along the chain.

    Hop 0 (entry) carries the full feature vocabulary; the share falls as
    ``(depth - hop + 1)/(depth + 1)`` toward the governing node, instantiating the
    similarity-decay model ``s_i = s_0 rho^i`` that the theory assumes.
    """
    kws = _keywords(area)
    fraction = (depth - hop + 1) / (depth + 1)
    floor = 0 if governing else 1
    k = max(floor, round(len(kws) * fraction))
    sel = " ".join(kws[:k])
    if governing:
        head = f"{sel}. " if sel else ""
        return (
            f"{head}RULE: every operation in this path must call {area.guard}() before proceeding."
        )
    if hop == 0:
        return f"The {sel} surface is where users begin."
    policy = area.policies[hop % len(area.policies)]
    return f"{sel} {policy} policy layer, refining an upstream constraint it derives from."


def _chain(area: Area, depth: int) -> tuple[list[MemoryItem], str]:
    """Build a depth-``d`` chain entry -> policy* -> governing for one area."""
    items: list[MemoryItem] = []
    ids = [f"{area.slug}-entry"] + [f"{area.slug}-policy{h}" for h in range(1, depth)]
    ids.append(f"{area.slug}-rule")
    texts = [_node_text(area, hop, depth, governing=(hop == depth)) for hop in range(depth + 1)]
    # link each node to the next (entry ... -> rule); the last edge is CONSTRAINS
    for idx, (node_id, text) in enumerate(zip(ids, texts, strict=True)):
        edges: list[tuple[str, str]] = []
        if idx < len(ids) - 1:
            rel = "constrains" if idx == len(ids) - 2 else "derived_from"
            edges = [(ids[idx + 1], rel)]
        node_type = "entry" if idx == 0 else ("governing" if idx == len(ids) - 1 else "policy")
        items.append(MemoryItem(node_id, text, metadata={"node_type": node_type, "edges": edges}))
    return items, ids[-1]


def load_synthetic_tasks(
    depths: tuple[int, ...] = (1, 2, 3),
    per_depth: int = 15,
    seed: int = 0,
    distractor_areas: int = 6,
) -> list[Task]:
    """Generate depth-controlled recovery tasks.

    For each depth and task, one area's chain carries the governing rule ``depth``
    hops from the query-similar entry node; several other areas' chains and filler
    nodes form the distractor haystack. Every task at a given depth shares the same
    corpus structure but rotates which area is the target, so no single area drives
    the result.
    """
    rng = random.Random(seed)
    tasks: list[Task] = []
    for depth in depths:
        if depth < 1:
            raise ValueError("depth must be >= 1")
        for i in range(per_depth):
            target = AREAS[(i + depth) % len(AREAS)]
            corpus: list[MemoryItem] = []
            target_items, governing_id = _chain(target, depth)
            corpus.extend(target_items)
            # distractor chains from other areas (same depth), query-irrelevant here
            others = [a for a in AREAS if a.slug != target.slug]
            rng.shuffle(others)
            for area in others[:distractor_areas]:
                corpus.extend(_chain(area, depth)[0])
            # plain filler nodes
            for j, filler in enumerate(_FILLER):
                corpus.append(MemoryItem(f"filler-{i}-{j}", filler))
            rng.shuffle(corpus)
            query = (
                f"Add a {target.feature} to the {target.name} module so users can {target.action}."
            )
            tasks.append(
                Task(
                    task_id=f"synthetic-d{depth}-{i:02d}",
                    dataset="synthetic",
                    query=query,
                    repo_ref="synthetic://depth-bench",
                    memory_corpus=tuple(corpus),
                    governing_decisions=(governing_id,),
                    depth=depth,
                    spec_variant=SpecVariant.STRIPPED,
                    scorer=Scorer.COMPLIANCE,
                )
            )
    return tasks
