"""Usage-aware confidence decay, mirroring Brief's per-type edge decay.

Brief keeps its graph healthy with a periodic decay pass (``runPerTypeDecay``):
edges lose confidence over time at a per-relationship-type rate, but links that
get *traversed* are reinforced (a frequently used edge is evidently real), and
edges that are stale, never traversed, and low confidence are pruned. This keeps
the graph's high-confidence edges the ones the agent actually relies on — a
property a static similarity index does not have.

This module reproduces that logic deterministically over a logical clock (``tick``)
rather than wall-clock time, so a benchmark run is reproducible.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from membench.retrieval.graph.store import GraphEdge, RelationshipType, TypedGraph

__all__ = ["DecayPolicy", "DecayReport", "apply_decay"]


@dataclass(frozen=True, slots=True)
class DecayPolicy:
    """Parameters of the usage-aware decay pass.

    Attributes
    ----------
    default_decay_rate
        Fractional confidence lost per pass for a stale edge of a type without an
        explicit rate.
    decay_rates
        Per-relationship-type decay-rate overrides.
    stale_after_ticks
        An edge is "stale" if it has not been traversed within this many ticks.
    restore_threshold
        Traversal count at or above which an edge is reinforced instead of decayed.
    restore_factor
        Multiplier on the type's decay rate applied as a confidence gain when an
        edge is reinforced.
    prune_below
        Edges below this confidence that have never been traversed are pruned.
    """

    default_decay_rate: float = 0.1
    decay_rates: Mapping[RelationshipType, float] = field(default_factory=dict)
    stale_after_ticks: int = 1
    restore_threshold: int = 10
    restore_factor: float = 0.75
    prune_below: float = 0.1

    def rate_for(self, relationship_type: RelationshipType) -> float:
        """Return the decay rate for a relationship type (override or default)."""
        return self.decay_rates.get(relationship_type, self.default_decay_rate)


@dataclass(frozen=True, slots=True)
class DecayReport:
    """Counts of what the decay pass did."""

    decayed: int
    reinforced: int
    pruned: int


def _is_stale(edge: GraphEdge, current_tick: int, stale_after_ticks: int) -> bool:
    if edge.last_traversed_tick is None:
        return True
    return (current_tick - edge.last_traversed_tick) >= stale_after_ticks


def apply_decay(graph: TypedGraph, current_tick: int, policy: DecayPolicy) -> DecayReport:
    """Run one usage-aware decay pass over ``graph`` in place.

    For each edge: reinforce it if it has been traversed at least
    ``restore_threshold`` times; otherwise decay it if it is stale. Afterwards,
    prune edges that are below ``prune_below`` and have never been traversed.
    """
    decayed = reinforced = 0
    for edge in graph.edges():
        rate = policy.rate_for(edge.relationship_type)
        if edge.traversal_count >= policy.restore_threshold:
            edge.confidence = min(1.0, edge.confidence + policy.restore_factor * rate)
            reinforced += 1
        elif _is_stale(edge, current_tick, policy.stale_after_ticks):
            edge.confidence = edge.confidence * (1.0 - rate)
            decayed += 1

    pruned = graph.prune_edges(
        lambda e: not (e.confidence < policy.prune_below and e.traversal_count == 0)
    )
    return DecayReport(decayed=decayed, reinforced=reinforced, pruned=pruned)
