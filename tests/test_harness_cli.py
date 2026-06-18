"""Tests for the end-to-end harness and CLI."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from membench.analysis.tables import depth_crossover_table
from membench.cli import app
from membench.harness import load_rows, run_benchmark, save_rows

runner = CliRunner()


class TestHarness:
    def test_run_benchmark_shows_depth_advantage(self) -> None:
        rows = run_benchmark("synthetic", per_depth=8, budget=150, seed=0)
        table = depth_crossover_table(rows)
        # at the deepest depth, the graph arm recovers far more than dense
        assert table[("brief_graph_3hop", 3)] > table[("dense", 3)]
        assert table[("brief_graph_3hop", 3)] >= 0.9

    def test_round_trip_rows(self, tmp_path: Path) -> None:
        rows = run_benchmark("synthetic", per_depth=4, budget=150, seed=1)
        path = save_rows(rows, tmp_path / "rows.jsonl")
        reloaded = load_rows(path)
        assert len(reloaded) == len(rows)
        assert reloaded[0] == rows[0]


class TestCLI:
    def test_theory_command(self) -> None:
        result = runner.invoke(app, ["theory", "--rho", "0.5", "--q", "0.97"])
        assert result.exit_code == 0
        assert "predicted d*" in result.stdout or "no crossover" in result.stdout

    def test_run_then_tables_report_figures(self, tmp_path: Path) -> None:
        rows_path = tmp_path / "rows.jsonl"
        run_res = runner.invoke(
            app, ["run", "--dataset", "synthetic", "--per-depth", "4", "--out", str(rows_path)]
        )
        assert run_res.exit_code == 0 and rows_path.exists()
        assert "depth crossover" in run_res.stdout

        tables_res = runner.invoke(app, ["tables", "--src", str(rows_path)])
        assert tables_res.exit_code == 0 and "headline" in tables_res.stdout

        report_path = tmp_path / "report.md"
        rep_res = runner.invoke(app, ["report", "--src", str(rows_path), "--out", str(report_path)])
        assert rep_res.exit_code == 0 and report_path.exists()
        assert "Executive summary" in report_path.read_text()

        fig_res = runner.invoke(
            app, ["figures", "--src", str(rows_path), "--out-dir", str(tmp_path / "figs")]
        )
        assert fig_res.exit_code == 0
        assert (tmp_path / "figs" / "depth_crossover.png").exists()
