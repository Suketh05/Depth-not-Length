"""Bounded multi-hop BFS traversal, reproducing Brief's ``traverseEntityGraph``.

Brief's production traversal is an in-application breadth-first walk (not a
recursive SQL CTE): it starts from one or more seed entities, expands a default of
two hops (clamped to one-to-three), detects cycles via a visited set, batches edge
fetches per depth level, and stops at a node budget. This module reproduces those
semantics over :class:`~membench.retrieval.graph.store.TypedGraph`.

The walk optionally records traversals on the edges it follows
(``traversal_count`` / ``last_traversed_tick``), which is the usage signal Brief's
per-type confidence decay later rewards — traversed links are reinforced, stale
links decay and are pruned.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from membench.retrieval.graph.store import GraphEdge, GraphNode, RelationshipType, TypedGraph

__all__ = ["DEFAULT_MAX_HOPS", "MAX_HOPS_LIMIT", "TraversalResult", "traverse"]

DEFAULT_MAX_HOPS = 2
MAX_HOPS_LIMIT = 3
_DEFAULT_MAX_NODES = 100


@dataclass(frozen=True, slots=True)
class TraversalResult:
    """The outcome of a bounded BFS walk.

    Attributes
    ----------
    nodes
        ``(node, depth)`` pairs in BFS order, including the seed nodes at depth 0.
    edges
        The edges followed, in the order traversed (may repeat a node's edge if it
        is reached again; useful for usage accounting).
    max_depth_reached
        The greatest depth any node was discovered at.
    truncated
        Whether the walk stopped because it hit the node budget.
    """

    nodes: tuple[tuple[GraphNode, int], ...]
    edges: tuple[GraphEdge, ...]
    max_depth_reached: int
    truncated: bool

    @property
    def node_ids(self) -> list[str]:
        """The discovered node ids in BFS order (seeds first)."""
        return [node.node_id for node, _ in self.nodes]


def traverse(
    graph: TypedGraph,
    start_ids: list[str],
    *,
    max_hops: int = DEFAULT_MAX_HOPS,
    max_nodes: int = _DEFAULT_MAX_NODES,
    direction: str = "both",
    relationship_types: frozenset[RelationshipType] | None = None,
    min_confidence: float = 0.0,
    track_traversal: bool = False,
    tick: int | None = None,
) -> TraversalResult:
    """Walk the graph breadth-first from ``start_ids`` within the hop/node bounds.

    Parameters
    ----------
    graph
        The typed graph to walk.
    start_ids
        Seed node ids (those absent from the graph are skipped).
    max_hops
        Maximum hops from a seed; clamped to ``[1, MAX_HOPS_LIMIT]`` exactly as
        Brief clamps it.
    max_nodes
        Node budget; the walk stops once this many distinct nodes are discovered.
    direction
        Edge direction to follow (``out`` / ``in`` / ``both``).
    relationship_types
        If given, only follow edges of these types.
    min_confidence
        Only follow edges at or above this confidence.
    track_traversal
        If true, increment ``traversal_count`` and set ``last_traversed_tick`` on
        every followed edge (the usage signal the decay model reads).
    tick
        Logical clock value recorded on followed edges when tracking.
    """
    hops = max(1, min(max_hops, MAX_HOPS_LIMIT))
    if max_nodes < 1:
        raise ValueError(f"max_nodes must be >= 1, got {max_nodes}")

    depth_by_id: dict[str, int] = {}
    ordered: list[tuple[GraphNode, int]] = []
    followed: list[GraphEdge] = []
    queue: deque[tuple[str, int]] = deque()

    for sid in start_ids:
        if graph.has_node(sid) and sid not in depth_by_id:
            depth_by_id[sid] = 0
            ordered.append((graph.get_node(sid), 0))
            queue.append((sid, 0))

    truncated = False
    while queue:
        node_id, depth = queue.popleft()
        if depth >= hops:
            continue
        for edge, neighbor in graph.neighbors(
            node_id,
            direction=direction,
            relationship_types=relationship_types,
            min_confidence=min_confidence,
        ):
            if track_traversal:
                edge.traversal_count += 1
                edge.last_traversed_tick = tick
            followed.append(edge)
            if neighbor.node_id in depth_by_id:
                continue
            if len(depth_by_id) >= max_nodes:
                truncated = True
                continue
            depth_by_id[neighbor.node_id] = depth + 1
            ordered.append((neighbor, depth + 1))
            queue.append((neighbor.node_id, depth + 1))

    max_depth = max((d for _, d in ordered), default=0)
    return TraversalResult(
        nodes=tuple(ordered),
        edges=tuple(followed),
        max_depth_reached=max_depth,
        truncated=truncated,
    )
