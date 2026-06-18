"""Publication figures for the benchmark.

Renders the three figures that carry the paper: the depth-crossover curve
(similarity decays with depth, the structured arm stays flat), the accuracy/cost
Pareto frontier (the joint win), and the Demšar critical-difference diagram
(ranking all arms with significance). Uses the non-interactive Agg backend so it
runs headless in CI, and returns ``matplotlib`` Figure objects the caller saves.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

__all__ = [
    "plot_cost_frontier",
    "plot_critical_difference",
    "plot_depth_crossover",
    "save_figure",
]


def plot_depth_crossover(
    series_by_arm: Mapping[str, Mapping[int, float]],
    *,
    ylabel: str = "full-chain recovery",
    title: str = "Depth crossover",
) -> Figure:
    """Plot per-arm recovery vs. causal depth (one line per arm)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm, series in series_by_arm.items():
        depths = sorted(series)
        ax.plot(depths, [series[d] for d in depths], marker="o", label=arm)
    ax.set_xlabel("causal depth $d$")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def plot_cost_frontier(
    points_by_arm: Mapping[str, tuple[float, float]],
    frontier: Sequence[str] = (),
    *,
    title: str = "Accuracy vs. token cost",
) -> Figure:
    """Scatter (cost, accuracy) per arm, highlighting the Pareto frontier."""
    fig, ax = plt.subplots(figsize=(6, 4))
    frontier_set = set(frontier)
    for arm, (accuracy, cost) in points_by_arm.items():
        on_front = arm in frontier_set
        ax.scatter(
            cost,
            accuracy,
            s=70 if on_front else 35,
            marker="*" if on_front else "o",
            zorder=3 if on_front else 2,
        )
        ax.annotate(arm, (cost, accuracy), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("tokens to correct (lower better)")
    ax.set_ylabel("accuracy (higher better)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_critical_difference(
    average_ranks: Mapping[str, float],
    critical_difference: float,
    *,
    title: str = "Critical-difference diagram",
) -> Figure:
    """Plot mean ranks on a line with a critical-difference bar (lower rank = better)."""
    fig, ax = plt.subplots(figsize=(7, 0.5 + 0.32 * len(average_ranks)))
    ordered = sorted(average_ranks.items(), key=lambda kv: kv[1])
    names = [n for n, _ in ordered]
    ranks = [r for _, r in ordered]
    y = list(range(len(names)))
    ax.scatter(ranks, y, zorder=3)
    for yi, (name, rank) in zip(y, ordered, strict=True):
        ax.annotate(
            f"{name} ({rank:.2f})",
            (rank, yi),
            fontsize=8,
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
        )
    best = ranks[0]
    ax.plot([best, best + critical_difference], [-0.6, -0.6], color="black", lw=2)
    ax.annotate(f"CD = {critical_difference:.2f}", (best, -0.9), fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("mean rank (1 = best)")
    ax.set_title(title)
    ax.invert_xaxis()
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: str | Path, *, dpi: int = 150) -> Path:
    """Save a figure to ``path`` (creating parent dirs) and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out
