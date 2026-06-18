"""Tests for the empirical similarity-decay measurement and figure."""

from __future__ import annotations

from pathlib import Path

import pytest

from membench.analysis.decay_fit import decay_fit_figure, measure_decay
from membench.analysis.plots import save_figure
from membench.retrieval.embeddings import HashingEmbeddingProvider


def _chain() -> tuple[str, list[str]]:
    query = "alpha beta gamma delta export audit"
    # decreasing overlap with the query along the chain
    chain = [
        "alpha beta gamma delta export audit surface",
        "alpha beta gamma export policy layer",
        "alpha beta retention policy",
        "alpha governance rule",
        "unrelated terminal rule clause",
    ]
    return query, chain


class TestMeasureDecay:
    def test_returns_hops_and_decreasing_similarity(self) -> None:
        provider = HashingEmbeddingProvider(dim=256)
        query, chain = _chain()
        hops, sims = measure_decay(provider, query, chain)
        assert hops == [0, 1, 2, 3, 4]
        assert sims[0] > sims[-1]  # entry resembles the query more than the terminal
        assert all(s > 0 for s in sims)

    def test_rejects_empty_chain(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            measure_decay(HashingEmbeddingProvider(), "q", [])


class TestDecayFitFigure:
    def test_fits_and_renders(self, tmp_path: Path) -> None:
        provider = HashingEmbeddingProvider(dim=256)
        query, chain = _chain()
        hops, sims = measure_decay(provider, query, chain)
        fig, fit = decay_fit_figure(hops, sims)
        assert 0.0 < fit.model.rho < 1.0  # genuine decay
        out = save_figure(fig, tmp_path / "decay.png")
        assert out.exists() and out.stat().st_size > 0
