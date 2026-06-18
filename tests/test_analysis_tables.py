"""Tests for scored-result aggregation and the output tables."""

from __future__ import annotations

from membench.agents.runner import AttemptRecord
from membench.analysis.tables import (
    ScoredRow,
    ablation_table,
    competitor_matrix,
    depth_crossover_table,
    headline_table,
    score_records,
)
from membench.competitors import ProvenanceTier, load_vendor_claims
from membench.metrics.compliance import ComplianceResult  # noqa: F401  (kept for parity)
from membench.types import MemoryItem, Scorer, SpecVariant, Task


def _task(task_id: str, depth: int) -> Task:
    corpus = (MemoryItem("D-1", "must call withAuditLog()"), MemoryItem("D-2", "use Button"))
    return Task(
        task_id=task_id,
        dataset="synthetic",
        query="q",
        repo_ref="r",
        memory_corpus=corpus,
        governing_decisions=("D-1",),
        depth=depth,
        spec_variant=SpecVariant.STRIPPED,
        scorer=Scorer.COMPLIANCE,
    )


def _rec(task_id: str, arm: str, depth: int, *, complied: bool, recovered: bool) -> AttemptRecord:
    return AttemptRecord(
        task_id=task_id,
        dataset="synthetic",
        arm=arm,
        model="claude-sonnet-4-6",
        attempt=1,
        depth=depth,
        spec_variant="stripped",
        scorer="compliance",
        retrieved_ids=("D-1",) if recovered else (),
        governing_decisions=("D-1",),
        response_text="call withAuditLog()" if complied else "nothing relevant",
        input_tokens=100,
        output_tokens=20,
        dollars=0.001,
    )


def _dataset() -> tuple[list[AttemptRecord], dict[str, Task]]:
    tasks = {f"t{d}": _task(f"t{d}", d) for d in (1, 2, 3)}
    records = []
    for d in (1, 2, 3):
        # graph arm always recovers + complies; none arm never does
        records.append(_rec(f"t{d}", "brief_graph", d, complied=True, recovered=True))
        records.append(_rec(f"t{d}", "none", d, complied=False, recovered=False))
    return records, tasks


class TestScoreRecords:
    def test_joins_metrics(self) -> None:
        records, tasks = _dataset()
        rows = score_records(records, tasks)
        assert len(rows) == 6
        graph = [r for r in rows if r.arm == "brief_graph"]
        assert all(r.compliance_rate == 1.0 and r.chain_recovered for r in graph)
        none = [r for r in rows if r.arm == "none"]
        assert all(r.compliance_rate == 0.0 and not r.chain_recovered for r in none)


class TestTables:
    def test_headline(self) -> None:
        rows = score_records(*_dataset())
        table = headline_table(rows)
        assert table[("synthetic", "brief_graph")] == 1.0
        assert table[("synthetic", "none")] == 0.0

    def test_depth_crossover(self) -> None:
        rows = score_records(*_dataset())
        table = depth_crossover_table(rows)
        for d in (1, 2, 3):
            assert table[("brief_graph", d)] == 1.0
            assert table[("none", d)] == 0.0

    def test_ablation_cells(self) -> None:
        # The ecological ablation lives on dcbench/swebench with the 3-hop graph arm.
        def row(dataset: str, arm: str, depth: int, rate: float) -> ScoredRow:
            variant = "full" if depth == 1 else "stripped"
            return ScoredRow(
                dataset,
                arm,
                "claude-sonnet-4-6",
                depth,
                variant,
                rate,
                rate == 1.0,
                rate,
                rate,
                rate == 1.0,
                120,
                0.0,
            )

        rows = [
            row("dcbench", "none", 1, 1.0),
            row("dcbench", "none", 3, 0.0),
            row("dcbench", "brief_graph_3hop", 3, 0.6),
            row("dcbench", "random_context", 3, 0.1),
            # A synthetic row at the same (arm, depth) must NOT leak into the ablation.
            row("synthetic", "brief_graph_3hop", 3, 1.0),
        ]
        table = ablation_table(rows)
        expected = {"full_spec/none", "stripped/none", "stripped/brief", "stripped/random"}
        assert set(table) == expected
        assert table["full_spec/none"] == 1.0
        assert table["stripped/none"] == 0.0
        # The dcbench value, not the synthetic 1.0 -- synthetic is excluded by design.
        assert table["stripped/brief"] == 0.6
        assert table["stripped/random"] == 0.1


class TestCompetitorMatrix:
    def test_two_tiers_kept_separate(self) -> None:
        rows = score_records(*_dataset())
        matrix = competitor_matrix(rows, load_vendor_claims())
        assert matrix.measured_tier is ProvenanceTier.MEASURED
        assert matrix.vendor_tier is ProvenanceTier.VENDOR_REPORTED
        assert matrix.measured["brief_graph"] == 1.0
        assert all(c.tier is ProvenanceTier.VENDOR_REPORTED for c in matrix.vendor_reported)
