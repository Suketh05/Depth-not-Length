"""The ``membench`` command-line interface.

Five commands cover the workflow: ``theory`` (predict the crossover depth from
parameters), ``run`` (execute a dataset x arms sweep and persist scored rows),
``tables`` (print the headline tables), ``report`` (write the Markdown report), and
``figures`` (render the publication figures). Everything runs offline by default.
"""

from __future__ import annotations

from pathlib import Path

import typer

from membench.analysis.report import generate_report
from membench.analysis.tables import depth_crossover_table, headline_table
from membench.competitors import load_vendor_claims
from membench.harness import DEFAULT_ARMS, load_rows, run_benchmark, save_rows
from membench.theory.crossover import find_crossover_depth, overhead_ratio
from membench.theory.decay import SimilarityDecayModel
from membench.theory.recovery import RecoveryModel

app = typer.Typer(add_completion=False, help="Depth-stratified agent-memory benchmark.")

_DEFAULT_RESULTS = Path("results/rows.jsonl")


@app.command()
def theory(
    rho: float = 0.5,
    q: float = 0.97,
    s0: float = 0.9,
    maintenance: float = 0.2,
    value: float = 1.0,
) -> None:
    """Predict the crossover depth d* from decay/structure parameters."""
    model = RecoveryModel(decay=SimilarityDecayModel(s0=s0, rho=rho), q=q)
    tau = overhead_ratio(maintenance, value)
    result = find_crossover_depth(model, tau)
    if result.exists:
        typer.echo(
            f"predicted d* = {result.d_star} (continuous {result.d_star_continuous:.2f}); "
            f"win region {result.win_region}; tau={tau:.3f}"
        )
    else:
        typer.echo(f"no crossover at tau={tau:.3f} (structured overhead never clears)")


@app.command()
def run(
    dataset: str = "synthetic",
    budget: int = 150,
    per_depth: int = 10,
    seed: int = 0,
    out: Path = _DEFAULT_RESULTS,
) -> None:
    """Run a dataset x arms sweep (offline) and write scored rows to JSONL."""
    rows = run_benchmark(dataset, DEFAULT_ARMS, budget, per_depth=per_depth, seed=seed)
    save_rows(rows, out)
    typer.echo(f"ran {len(rows)} scored rows over {dataset} -> {out}")
    crossover = depth_crossover_table(rows)
    arms = sorted({a for a, _ in crossover})
    depths = sorted({d for _, d in crossover})
    typer.echo("depth crossover (full-chain recovery):")
    for arm in arms:
        cells = " ".join(f"d{d}={crossover.get((arm, d), float('nan')):.2f}" for d in depths)
        typer.echo(f"  {arm:<18} {cells}")


@app.command()
def tables(src: Path = _DEFAULT_RESULTS) -> None:
    """Print the headline tables from a results file."""
    rows = load_rows(src)
    typer.echo("headline compliance (dataset, arm):")
    for (dataset, arm), value in sorted(headline_table(rows).items(), key=lambda kv: str(kv[0])):
        typer.echo(f"  {dataset:<12} {arm:<18} {value:.2f}")


@app.command()
def report(src: Path = _DEFAULT_RESULTS, out: Path = Path("results/report.md")) -> None:
    """Write the Markdown report from a results file."""
    rows = load_rows(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(generate_report(rows, vendor_claims=load_vendor_claims()))
    typer.echo(f"wrote report -> {out}")


@app.command()
def figures(src: Path = _DEFAULT_RESULTS, out_dir: Path = Path("figures")) -> None:
    """Render the publication figures from a results file."""
    from membench.analysis.plots import plot_depth_crossover, save_figure

    rows = load_rows(src)
    crossover = depth_crossover_table(rows)
    series: dict[str, dict[int, float]] = {}
    for (arm, depth), value in crossover.items():
        series.setdefault(arm, {})[depth] = value
    path = save_figure(plot_depth_crossover(series), out_dir / "depth_crossover.png")
    typer.echo(f"wrote figure -> {path}")


if __name__ == "__main__":  # pragma: no cover
    app()
