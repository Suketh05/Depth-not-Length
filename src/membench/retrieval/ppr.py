r"""Personalized PageRank retrieval over the typed context graph.

This arm ranks graph nodes by the stationary mass of a *personalized* (a.k.a.
topic-sensitive) PageRank random walk, then packs the highest-mass nodes into the
shared token budget. It is a global, link-structure-aware alternative to the
bounded BFS of :mod:`membench.retrieval.graph.arm`: where BFS only keeps nodes
within ``max_hops`` of a seed, PageRank propagates relevance along *every* typed
edge until it converges, so a governing decision many hops behind a query-similar
justification still accumulates mass.

Algorithm
---------
Let :math:`P` be the row-normalized adjacency of the typed-edge graph (so
:math:`P_{ij}` is the probability of stepping from node :math:`i` to node
:math:`j`) and let :math:`s` be the *restart* (personalization) distribution that
seeds the walk by each node's similarity to the query. With teleport probability
:math:`\alpha \in (0, 1)` the personalized PageRank vector :math:`r` is the fixed
point of the power iteration

.. math::

    r \leftarrow (1 - \alpha)\, P^{\top} r + \alpha\, s,

i.e. with probability :math:`1 - \alpha` the surfer follows a random out-edge and
with probability :math:`\alpha` it teleports back to the personalization
distribution :math:`s`. ``r`` is a probability distribution: it is initialised to
:math:`s` and every update preserves :math:`\sum_i r_i = 1`. The unique stationary
solution exists and is reached geometrically (at rate :math:`1 - \alpha`) for any
:math:`\alpha > 0` because the teleport term makes the chain irreducible and
aperiodic.

Dangling nodes (no out-edges, i.e. an all-zero row of :math:`P`) would otherwise
leak mass; their mass is reinjected through the personalization vector each
iteration, the standard correction that keeps :math:`r` a distribution. When the
graph has no edges at all every node is dangling and the iteration collapses to
:math:`r = s`, so the arm degrades gracefully to the pure query-similarity ranking.

References
----------
.. [1] T. H. Haveliwala, "Topic-Sensitive PageRank," in Proceedings of the 11th
   International World Wide Web Conference (WWW), 2002.
.. [2] L. Page, S. Brin, R. Motwani, and T. Winograd, "The PageRank Citation
   Ranking: Bringing Order to the Web," Stanford InfoLab Technical Report, 1999.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from membench.retrieval.base import ArmFamily, MemorySystem, pack_to_budget, register_arm
from membench.retrieval.embeddings import (
    EmbeddingProvider,
    HashingEmbeddingProvider,
    cosine_similarity_matrix,
)
from membench.retrieval.graph.store import GraphEdge, GraphNode, RelationshipType, TypedGraph
from membench.types import MemoryItem, RetrievedContext

__all__ = ["DEFAULT_ALPHA", "PageRankResult", "PersonalizedPageRankMemory", "personalized_pagerank"]

DEFAULT_ALPHA = 0.15
"""Teleport (restart) probability; ``1 - 0.15 = 0.85`` is the classic damping factor."""

DEFAULT_MAX_ITER = 200
"""Maximum power-iteration steps before giving up on the convergence tolerance."""

DEFAULT_TOL = 1e-12
"""L1 change between successive iterates below which the walk is declared converged."""


@dataclass(frozen=True, slots=True)
class PageRankResult:
    """The outcome of a personalized PageRank power iteration.

    Attributes
    ----------
    mass
        The stationary distribution ``r`` over nodes; non-negative and sums to 1.
    iterations
        Number of power-iteration steps performed.
    converged
        Whether the L1 change fell below the tolerance before ``max_iter``.
    """

    mass: NDArray[np.float64]
    iterations: int
    converged: bool


def personalized_pagerank(
    transition: NDArray[np.float64],
    restart: NDArray[np.float64],
    *,
    alpha: float = DEFAULT_ALPHA,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
) -> PageRankResult:
    r"""Solve ``r = (1 - alpha) P^T r + alpha s`` by power iteration.

    Parameters
    ----------
    transition
        Square ``(n, n)`` row-stochastic transition matrix ``P``: each row sums to
        1, except dangling rows (no out-edges) which sum to 0 and have their mass
        reinjected through ``restart`` each step.
    restart
        Length-``n`` restart/personalization distribution ``s``. Must be
        non-negative; it is L1-normalised internally (an all-zero vector falls back
        to the uniform distribution).
    alpha
        Teleport probability in ``[0, 1]``. Convergence to the unique stationary
        vector is guaranteed for ``alpha > 0``.
    max_iter
        Maximum number of iterations.
    tol
        Convergence threshold on the L1 distance between successive iterates.

    Returns
    -------
    PageRankResult
        The stationary mass vector, the iteration count, and whether it converged.

    Notes
    -----
    The update conserves total mass, so the returned ``mass`` always sums to 1
    (up to floating-point error). See the module docstring for the derivation and
    the Haveliwala (2002) reference.
    """
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError(f"transition must be a square matrix, got shape {transition.shape}")
    n = transition.shape[0]
    if restart.shape != (n,):
        raise ValueError(f"restart must have shape ({n},), got {restart.shape}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")

    seed = np.asarray(restart, dtype=np.float64)
    if (seed < 0.0).any():
        raise ValueError("restart distribution must be non-negative")
    total = float(seed.sum())
    seed = np.full(n, 1.0 / n) if total == 0.0 else seed / total

    if n == 0:
        return PageRankResult(mass=np.empty(0, dtype=np.float64), iterations=0, converged=True)

    matrix = np.asarray(transition, dtype=np.float64)
    row_sums = matrix.sum(axis=1)
    if not np.all(np.isclose(row_sums, 0.0) | np.isclose(row_sums, 1.0)):
        raise ValueError("transition rows must each sum to 0 (dangling) or 1 (stochastic)")
    dangling = np.isclose(row_sums, 0.0)
    transposed = matrix.T

    mass = seed.copy()
    iterations = 0
    converged = False
    while iterations < max_iter:
        iterations += 1
        dangling_mass = float(mass[dangling].sum())
        nxt = (1.0 - alpha) * (transposed @ mass + dangling_mass * seed) + alpha * seed
        delta = float(np.abs(nxt - mass).sum())
        mass = nxt
        if delta < tol:
            converged = True
            break

    return PageRankResult(mass=mass, iterations=iterations, converged=converged)


@register_arm("ppr", family=ArmFamily.GRAPH, follows_links=True, uses_embeddings=True)
class PersonalizedPageRankMemory(MemorySystem):
    """Rank typed-graph nodes by personalized PageRank mass, then pack to budget.

    The query seeds the restart distribution (each node weighted by its embedding
    cosine similarity to the query) and a random walk over the row-normalized typed
    edges propagates that relevance along the link structure; nodes are ranked by
    stationary mass. See the module docstring for the algorithm and citation.

    Parameters
    ----------
    alpha
        Teleport probability of the walk (defaults to :data:`DEFAULT_ALPHA`).
    max_iter
        Maximum power-iteration steps (defaults to :data:`DEFAULT_MAX_ITER`).
    tol
        Convergence tolerance on the L1 iterate change (defaults to
        :data:`DEFAULT_TOL`).
    provider
        Embedding backend used to seed the restart distribution (defaults to the
        deterministic offline hashing provider).
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        max_iter: int = DEFAULT_MAX_ITER,
        tol: float = DEFAULT_TOL,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        self._alpha = alpha
        self._max_iter = max_iter
        self._tol = tol
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

    def _transition_matrix(self) -> NDArray[np.float64]:
        """Return the row-normalized typed-edge adjacency (confidence-weighted)."""
        n = len(self._items)
        index = {item.item_id: i for i, item in enumerate(self._items)}
        adjacency = np.zeros((n, n), dtype=np.float64)
        for edge in self._graph.edges():
            adjacency[index[edge.source_id], index[edge.target_id]] += edge.confidence
        row_sums = adjacency.sum(axis=1, keepdims=True)
        safe = np.where(row_sums == 0.0, 1.0, row_sums)  # leave dangling rows all-zero
        return np.asarray(adjacency / safe, dtype=np.float64)

    def _restart_distribution(self, query: str) -> NDArray[np.float64]:
        """Seed the walk by each node's (clamped, normalized) query similarity."""
        matrix = self._provider.embed([item.text for item in self._items])
        query_vec = self._provider.embed_one(query)
        sims = cosine_similarity_matrix(query_vec, matrix)[0]
        weights = np.clip(sims, 0.0, None)
        total = float(weights.sum())
        if total == 0.0:
            return np.full(len(self._items), 1.0 / len(self._items))
        return np.asarray(weights / total, dtype=np.float64)

    def retrieve(self, query: str, budget_tokens: int) -> RetrievedContext:
        """Rank nodes by personalized PageRank stationary mass, then pack to budget."""
        if not self._items:
            return RetrievedContext.empty()
        transition = self._transition_matrix()
        restart = self._restart_distribution(query)
        result = personalized_pagerank(
            transition, restart, alpha=self._alpha, max_iter=self._max_iter, tol=self._tol
        )
        order = np.argsort(-result.mass, kind="stable")
        ranked = [(self._items[i].item_id, self._items[i].text) for i in order]
        return pack_to_budget(ranked, budget_tokens)
