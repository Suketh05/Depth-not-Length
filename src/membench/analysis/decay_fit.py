"""Empirical validation of the similarity-decay law.

Definition 1 of the theory assumes resemblance to the query decays geometrically
with causal distance (``s_i = s_0 rho^i``). This module *measures* that on real
embeddings: it embeds the query and each node along a dependency chain, records the
cosine similarity at each hop, fits the geometric law, and renders the
measured-vs-fitted figure with the recovered ``rho`` and the R^2 -- the figure that
shows the model's central assumption actually holds on the data.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from membench.retrieval.embeddings import EmbeddingProvider, cosine_similarity
from membench.theory.decay import DecayFit, fit_similarity_decay

__all__ = ["decay_fit_figure", "measure_decay"]

_MIN_SIM = 1e-3  # floor so the log-space fit is defined for non-positive cosines


def measure_decay(
    provider: EmbeddingProvider, query: str, chain_texts: Sequence[str]
) -> tuple[list[int], list[float]]:
    """Return (hop, cosine-similarity-to-query) along a dependency chain.

    Hop 0 is the entry node (closest to the query); the last hop is the governing
    node. Similarities are floored at a small positive value so the downstream
    log-space decay fit is well defined.
    """
    if not chain_texts:
        raise ValueError("chain_texts must be non-empty")
    query_vec = provider.embed_one(query)
    matrix = provider.embed(list(chain_texts))
    hops = list(range(len(chain_texts)))
    sims = [max(cosine_similarity(query_vec, matrix[i]), _MIN_SIM) for i in hops]
    return hops, sims


def decay_fit_figure(
    distances: Sequence[float],
    similarities: Sequence[float],
    *,
    title: str = "Similarity decay along the chain",
) -> tuple[Figure, DecayFit]:
    """Fit the geometric decay law to (distance, similarity) and plot measured vs. fit."""
    fit = fit_similarity_decay(distances, similarities)
    xs = np.asarray(distances, dtype=np.float64)
    grid = np.linspace(float(xs.min()), float(xs.max()), 100)
    curve = [fit.model.retrievability(float(g)) for g in grid]
    label = f"fit rho={fit.model.rho:.2f}, R2={fit.r_squared:.2f}"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xs, similarities, label="measured", zorder=3)
    ax.plot(grid, curve, color="C1", label=label)
    ax.set_xlabel("causal distance $i$ (hops)")
    ax.set_ylabel("cosine similarity to query $s_i$")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, fit
