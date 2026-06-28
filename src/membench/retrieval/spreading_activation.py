r"""Spreading-activation retrieval over the typed context graph.

This arm implements the classic *spreading activation* model of associative
memory (Collins & Loftus 1975; Anderson 1983, ACT-R), in the discrete,
weighted-network form that the information-retrieval literature standardised
(Crestani 1997, *Application of Spreading Activation Techniques in Information
Retrieval*).

Algorithm
---------
A small set of query-similar **seed** nodes are given an initial activation. On
each of ``hops`` discrete pulses, every node whose activation is at or above a
**firing threshold** ``theta`` propagates a ``decay``-attenuated, edge-weighted
share of its activation to its neighbours; the per-node activation received on a
pulse is accumulated into a running total, and nodes are finally ranked by that
total. Writing :math:`a^{(t)}_i` for the activation of node :math:`i` *arriving*
on pulse :math:`t` and :math:`w_{ji}` for the weight (edge confidence) of the
link :math:`j \to i`, the update is

.. math::

    a^{(0)}_i &= s_i \quad\text{(seed activation, 0 for non-seeds)} \\
    a^{(t)}_i &= \gamma \sum_{j \,:\, a^{(t-1)}_j \ge \theta} a^{(t-1)}_j \, w_{ji}
                 \qquad (t = 1, \dots, H) \\
    A_i       &= \sum_{t=0}^{H} a^{(t)}_i ,

where :math:`\gamma \in [0, 1]` is the per-hop decay factor, :math:`\theta` the
firing threshold, :math:`H` the hop count, and :math:`A_i` the final accumulated
activation used for ranking. The decay factor is the path-length attenuation of
Collins & Loftus (activation weakens with distance from the source); the firing
threshold and edge weights are the standard network-model refinements.

Two limit cases make the model easy to verify:

* On a unit-weight chain ``seed -> n1 -> n2 -> ...`` with no thresholding, the
  activation reaching the node :math:`d` hops from a single seed of strength
  :math:`s` is exactly :math:`s\,\gamma^{d}` — a clean geometric decay.
* With :math:`\gamma = 0` no activation ever leaves the seeds, so the result is
  exactly the seed activations (every other node stays at zero). This is the
  degenerate "no spreading" baseline.

Relation to the other arms
---------------------------
Like :mod:`membench.retrieval.graph.arm`, this arm seeds by embedding similarity
and then exploits the *structure* of the graph, reusing the same
:class:`~membench.retrieval.graph.store.TypedGraph` store. It differs from the
bounded-BFS graph arm in *how* it scores reached nodes: BFS treats every node
within the hop bound equally, whereas spreading activation gives each node a
graded score that falls off with distance and with the strength of the links on
the path — so a governing decision two confident hops behind a query-similar
justification still receives, and is ranked by, a well-defined activation.

References
----------
Collins, A. M., & Loftus, E. F. (1975). A spreading-activation theory of
semantic processing. *Psychological Review*, 82(6), 407-428.

Anderson, J. R. (1983). A spreading activation theory of memory. *Journal of
Verbal Learning and Verbal Behavior*, 22(3), 261-295.

Crestani, F. (1997). Application of spreading activation techniques in
information retrieval. *Artificial Intelligence Review*, 11(6), 453-482.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from membench.retrieval._native import cosine_topk_reference
from membench.retrieval.base import ArmFamily, MemorySystem, pack_to_budget, register_arm
from membench.retrieval.embeddings import EmbeddingProvider, HashingEmbeddingProvider
from membench.retrieval.graph.store import GraphEdge, GraphNode, RelationshipType, TypedGraph
from membench.types import MemoryItem, RetrievedContext

__all__ = ["DEFAULT_DECAY", "DEFAULT_HOPS", "SpreadingActivationMemory", "spread_activation"]

DEFAULT_HOPS = 3
"""Default number of activation pulses propagated from the seeds."""

DEFAULT_DECAY = 0.5
"""Default per-hop activation decay factor (path-length attenuation)."""


def spread_activation(
    graph: TypedGraph,
    seeds: Mapping[str, float],
    *,
    decay: float = DEFAULT_DECAY,
    hops: int = DEFAULT_HOPS,
    threshold: float = 0.0,
    direction: str = "both",
    min_confidence: float = 0.0,
) -> dict[str, float]:
    r"""Spread activation from ``seeds`` over ``graph`` and return accumulated totals.

    Implements the discrete pulse update :math:`a^{(t)}_i = \gamma \sum_j a^{(t-1)}_j
    w_{ji}` over firing nodes (those at or above ``threshold``), accumulating the
    activation arriving on every pulse, as described in the module docstring.

    Parameters
    ----------
    graph
        The typed graph to spread over; only edge confidence (the link weight) and
        adjacency are used.
    seeds
        Mapping of seed ``node_id`` to its initial activation. Seeds absent from the
        graph are ignored.
    decay
        Per-hop decay factor :math:`\gamma \in [0, 1]`. ``0`` disables spreading
        (only the seeds remain activated); ``1`` propagates without attenuation.
    hops
        Number of activation pulses :math:`H \ge 0` to propagate.
    threshold
        Firing threshold :math:`\theta`: a node propagates on a pulse only if the
        activation it carries is ``>= threshold``.
    direction
        Edge direction followed when spreading (``"out"`` / ``"in"`` / ``"both"``),
        passed through to :meth:`TypedGraph.neighbors`.
    min_confidence
        Ignore links below this confidence when spreading.

    Returns
    -------
    dict[str, float]
        Mapping of ``node_id`` to total accumulated activation, containing exactly
        the nodes that received non-zero activation (always including the seeds).
    """
    if not 0.0 <= decay <= 1.0:
        raise ValueError(f"decay must be in [0, 1], got {decay}")
    if hops < 0:
        raise ValueError(f"hops must be >= 0, got {hops}")

    current: dict[str, float] = {
        node_id: value for node_id, value in seeds.items() if graph.has_node(node_id)
    }
    total: dict[str, float] = dict(current)

    for _ in range(hops):
        nxt: dict[str, float] = {}
        for node_id, activation in current.items():
            if activation < threshold:
                continue
            for edge, neighbor in graph.neighbors(
                node_id, direction=direction, min_confidence=min_confidence
            ):
                contribution = decay * activation * edge.confidence
                if contribution == 0.0:
                    continue
                nxt[neighbor.node_id] = nxt.get(neighbor.node_id, 0.0) + contribution
        if not nxt:
            break
        for node_id, activation in nxt.items():
            total[node_id] = total.get(node_id, 0.0) + activation
        current = nxt

    return total


@register_arm(
    "spreading_activation",
    family=ArmFamily.GRAPH,
    follows_links=True,
    uses_embeddings=True,
)
class SpreadingActivationMemory(MemorySystem):
    """Seed by similarity, then rank by spreading activation over the typed graph.

    The corpus is turned into the same :class:`TypedGraph` the structured-graph arm
    builds (one node per item; a ``constrains`` metadata key or an explicit
    ``edges`` list becomes a typed edge). Retrieval embeds the query, takes the most
    similar nodes as activation seeds, spreads activation over the typed links, and
    ranks every reached node by its accumulated activation before packing to budget.

    Parameters
    ----------
    max_hops
        Number of activation pulses propagated from the seeds.
    seeds
        Number of similarity-selected seed nodes.
    decay
        Per-hop activation decay factor in ``[0, 1]``.
    threshold
        Firing threshold; a node propagates only while its activation is at or above
        it.
    direction
        Edge direction followed when spreading (``"out"`` / ``"in"`` / ``"both"``).
    min_confidence
        Ignore links below this confidence when spreading.
    provider
        Embedding backend for seed selection (defaults to the offline hashing
        provider).
    """

    def __init__(
        self,
        max_hops: int = DEFAULT_HOPS,
        seeds: int = 5,
        decay: float = DEFAULT_DECAY,
        threshold: float = 0.0,
        direction: str = "both",
        min_confidence: float = 0.0,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        if seeds < 1:
            raise ValueError(f"seeds must be >= 1, got {seeds}")
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        if max_hops < 0:
            raise ValueError(f"max_hops must be >= 0, got {max_hops}")
        self._max_hops = max_hops
        self._seeds = seeds
        self._decay = decay
        self._threshold = threshold
        self._direction = direction
        self._min_confidence = min_confidence
        self._provider = provider or HashingEmbeddingProvider()
        self._graph = TypedGraph()
        self._items: list[MemoryItem] = []

    def write(self, items: Iterable[MemoryItem]) -> None:
        """Build the typed graph from the corpus (nodes + constrains/edges metadata)."""
        self._items.extend(items)
        for item in self._items:
            self._graph.add_node(
                GraphNode(item.item_id, item.text, node_type=item.node_type or "node")
            )
        for item in self._items:
            if item.constrains and self._graph.has_node(item.constrains):
                self._graph.add_edge(
                    GraphEdge(item.item_id, item.constrains, RelationshipType.CONSTRAINS)
                )
            for spec in item.metadata.get("edges", ()):  # explicit edges: (target, rel_type)
                target, rel = spec
                if self._graph.has_node(target):
                    self._graph.add_edge(GraphEdge(item.item_id, target, RelationshipType(rel)))

    def _seed_ids(self, query: str) -> list[str]:
        matrix = self._provider.embed([item.text for item in self._items])
        query_vec = self._provider.embed_one(query)
        idx, _ = cosine_topk_reference(matrix, query_vec, min(self._seeds, len(self._items)))
        return [self._items[i].item_id for i in idx]

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Spread activation from the query-similar seeds and rank reached nodes.

        Seeds each receive unit activation; activation is then spread over the typed
        links with the configured decay and firing threshold, and every node that
        accumulates positive activation is ranked by that total (ties broken by
        corpus order) before being packed into the budget.
        """
        if not self._items:
            return RetrievedContext.empty()
        seeds = dict.fromkeys(self._seed_ids(query), 1.0)
        activation = spread_activation(
            self._graph,
            seeds,
            decay=self._decay,
            hops=self._max_hops,
            threshold=self._threshold,
            direction=self._direction,
            min_confidence=self._min_confidence,
        )
        index_by_id = {item.item_id: position for position, item in enumerate(self._items)}
        activated = [(node_id, score) for node_id, score in activation.items() if score > 0.0]
        activated.sort(key=lambda pair: (-pair[1], index_by_id.get(pair[0], 0)))
        ranked = [(node_id, self._graph.get_node(node_id).text) for node_id, _ in activated]
        return pack_to_budget(ranked, budget_tokens)
