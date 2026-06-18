"""A typed-edge property graph mirroring Brief's context graph.

The data model is a direct, minimal reproduction of Brief's production graph:

* **Nodes** (``GraphNode``) correspond to ``entity_objects`` — a typed record with
  an id, text, a ``node_type`` (decision / constraint / justification / feature /
  ...), and free-form metadata.
* **Edges** (``GraphEdge``) correspond to ``entity_link`` — a *typed*, directed
  link with a ``relationship_type`` drawn from a fixed catalogue, a confidence in
  ``[0, 1]``, and usage tracking (``traversal_count``, ``last_traversed_tick``)
  that the decay model later rewards.

Endpoints are FK-enforced (an edge may only connect nodes that exist), exactly as
Postgres enforces in Brief, so the graph can never contain a dangling link. The
store maintains forward/backward adjacency indices for O(degree) neighbour
queries, which is what makes bounded BFS traversal cheap.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

__all__ = ["GraphEdge", "GraphNode", "RelationshipType", "TypedGraph"]


class RelationshipType(str, Enum):
    """A curated subset of Brief's 33-value relationship catalogue.

    These are the edge semantics the benchmark exercises. ``CONSTRAINS`` is the
    load-bearing one: a justification node *constrains* the decision it justifies,
    and following that link is the hop a flat retriever cannot make.
    """

    RELATED_TO = "related_to"
    CONSTRAINS = "constrains"
    CONSTRAINED_BY = "constrained_by"
    IMPLEMENTS = "implements"
    VALIDATES = "validates"
    CONTRADICTS = "contradicts"
    INFORMS = "informs"
    SUPERSEDES = "supersedes"
    BLOCKS = "blocks"
    DERIVED_FROM = "derived_from"
    SUPPORTS_GOAL = "supports_goal"
    MOTIVATES = "motivates"
    ENABLES = "enables"
    IMPACTS = "impacts"
    REFERENCES = "references"
    DUPLICATE_OF = "duplicate_of"


@dataclass(frozen=True, slots=True)
class GraphNode:
    """A typed record in the graph (mirrors a row in ``entity_objects``)."""

    node_id: str
    text: str
    node_type: str = "node"
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("GraphNode.node_id must be non-empty")
        if not isinstance(self.metadata, MappingProxyType):
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(slots=True)
class GraphEdge:
    """A typed, directed, confidence-weighted link (mirrors ``entity_link``).

    Mutable on purpose: ``traversal_count`` and ``last_traversed_tick`` are updated
    as the graph is walked, which the usage-aware decay model reads.
    """

    source_id: str
    target_id: str
    relationship_type: RelationshipType
    confidence: float = 1.0
    traversal_count: int = 0
    last_traversed_tick: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        # Accept the string form for ergonomics (loaders may pass raw strings);
        # idempotent on an existing member, so no isinstance branch is needed.
        self.relationship_type = RelationshipType(self.relationship_type)


class TypedGraph:
    """An in-memory typed property graph with FK-enforced edges and adjacency indices."""

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []
        self._out: dict[str, list[int]] = {}  # node_id -> edge indices (outgoing)
        self._in: dict[str, list[int]] = {}  # node_id -> edge indices (incoming)

    # -- construction -----------------------------------------------------------

    def add_node(self, node: GraphNode) -> None:
        """Add a node; raises if a node with the same id already exists."""
        if node.node_id in self._nodes:
            raise ValueError(f"duplicate node id {node.node_id!r}")
        self._nodes[node.node_id] = node
        self._out.setdefault(node.node_id, [])
        self._in.setdefault(node.node_id, [])

    def add_edge(self, edge: GraphEdge) -> None:
        """Add a typed edge; raises if either endpoint node is missing (FK)."""
        if edge.source_id not in self._nodes:
            raise ValueError(f"edge source {edge.source_id!r} is not a node")
        if edge.target_id not in self._nodes:
            raise ValueError(f"edge target {edge.target_id!r} is not a node")
        index = len(self._edges)
        self._edges.append(edge)
        self._out[edge.source_id].append(index)
        self._in[edge.target_id].append(index)

    # -- access -----------------------------------------------------------------

    def get_node(self, node_id: str) -> GraphNode:
        """Return the node with ``node_id`` or raise ``KeyError``."""
        return self._nodes[node_id]

    def has_node(self, node_id: str) -> bool:
        """Return whether ``node_id`` is in the graph."""
        return node_id in self._nodes

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        """Number of edges in the graph."""
        return len(self._edges)

    def nodes(self) -> Iterator[GraphNode]:
        """Iterate over the graph's nodes."""
        return iter(self._nodes.values())

    def edges(self) -> Iterator[GraphEdge]:
        """Iterate over the graph's edges."""
        return iter(self._edges)

    def neighbors(
        self,
        node_id: str,
        *,
        direction: str = "out",
        relationship_types: frozenset[RelationshipType] | None = None,
        min_confidence: float = 0.0,
    ) -> list[tuple[GraphEdge, GraphNode]]:
        """Return ``(edge, neighbour)`` pairs adjacent to ``node_id``.

        Parameters
        ----------
        node_id
            The node whose neighbours to return.
        direction
            ``"out"`` (outgoing edges), ``"in"`` (incoming), or ``"both"``.
        relationship_types
            If given, keep only edges of these types.
        min_confidence
            Drop edges below this confidence.
        """
        if node_id not in self._nodes:
            raise KeyError(node_id)
        if direction not in ("out", "in", "both"):
            raise ValueError(f"direction must be out/in/both, got {direction!r}")

        indices: list[int] = []
        if direction in ("out", "both"):
            indices += self._out[node_id]
        if direction in ("in", "both"):
            indices += self._in[node_id]

        results: list[tuple[GraphEdge, GraphNode]] = []
        for idx in indices:
            edge = self._edges[idx]
            if relationship_types is not None and edge.relationship_type not in relationship_types:
                continue
            if edge.confidence < min_confidence:
                continue
            other_id = edge.target_id if edge.source_id == node_id else edge.source_id
            results.append((edge, self._nodes[other_id]))
        return results
