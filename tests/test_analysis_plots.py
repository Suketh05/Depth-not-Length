"""Tests that the publication figures render and save without error."""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

from membench.analysis.plots import (
    plot_cost_frontier,
    plot_critical_difference,
    plot_depth_crossover,
    save_figure,
)


def test_depth_crossover_renders_and_saves(tmp_path: Path) -> None:
    series = {
        "brief_graph": {1: 1.0, 2: 1.0, 3: 1.0},
        "dense": {1: 0.95, 2: 0.6, 3: 0.3},
    }
    fig = plot_depth_crossover(series)
    assert isinstance(fig, Figure)
    out = save_figure(fig, tmp_path / "crossover.png")
    assert out.exists() and out.stat().st_size > 0


def test_cost_frontier_renders(tmp_path: Path) -> None:
    points = {"brief_graph": (1.0, 130.0), "dense": (0.6, 130.0), "full_context": (0.6, 400.0)}
    fig = plot_cost_frontier(points, frontier=["brief_graph"])
    out = save_figure(fig, tmp_path / "frontier.png")
    assert out.exists() and out.stat().st_size > 0


def test_critical_difference_renders(tmp_path: Path) -> None:
    ranks = {"brief_graph": 1.2, "hybrid_rrf": 2.5, "dense": 3.1, "none": 5.0}
    fig = plot_critical_difference(ranks, critical_difference=1.3)
    out = save_figure(fig, tmp_path / "cd.png")
    assert out.exists() and out.stat().st_size > 0
