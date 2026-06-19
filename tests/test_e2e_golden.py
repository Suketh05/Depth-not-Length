"""End-to-end golden test: the whole pipeline must reproduce the headline result.

Two separate guards, matching where each effect is meant to live:

* the **synthetic depth-crossover** carries the magnitude. similarity decays with depth
  while the structured graph arm stays flat and wins outright at depth. this is the
  controlled demonstration, so the thresholds here are strong.
* the **dcbench/swebench four-arm ablation** is ecological corroboration. under the
  deliberately weak offline stub the recovery is partial, so the guard is on the
  ordering (full-spec ceiling > structured > random control > none floor), not on a
  ≥0.9 magnitude. running the ablation on synthetic would conflate the two, so it is
  kept on its own datasets.

Offline and deterministic (seeded), so both are stable regression checks.
"""

from __future__ import annotations

import pytest

from membench.analysis.report import generate_report
from membench.analysis.tables import (
    ablation_table,
    depth_crossover_table,
    headline_table,
)
from membench.harness import ABLATION_ARMS, ABLATION_DATASETS, DEFAULT_ARMS, run_benchmark
from membench.theory.crossover import find_crossover_depth, overhead_ratio
from membench.theory.decay import SimilarityDecayModel
from membench.theory.recovery import RecoveryModel


@pytest.fixture(scope="module")
def crossover_rows() -> list:  # type: ignore[type-arg]
    """Synthetic rows for the controlled depth-crossover (the headline magnitude)."""
    return run_benchmark("synthetic", arms=DEFAULT_ARMS, per_depth=10, budget=150, seed=0)


@pytest.fixture(scope="module")
def ablation_rows() -> list:  # type: ignore[type-arg]
    """Four-arm ablation rows on dcbench + swebench (ecological corroboration)."""
    rows: list = []  # type: ignore[type-arg]
    for dataset in ABLATION_DATASETS:
        rows.extend(run_benchmark(dataset, arms=ABLATION_ARMS, per_depth=10, budget=150, seed=0))
    return rows


class TestGoldenCrossover:
    def test_structured_arm_stays_flat_high(self, crossover_rows: list) -> None:  # type: ignore[type-arg]
        table = depth_crossover_table(crossover_rows)
        for d in (1, 2, 3):
            assert table[("brief_graph_3hop", d)] >= 0.9

    def test_similarity_decays_with_depth(self, crossover_rows: list) -> None:  # type: ignore[type-arg]
        table = depth_crossover_table(crossover_rows)
        dense = [table[("dense", d)] for d in (1, 2, 3)]
        assert dense[0] >= 0.8  # fine when shallow
        assert dense[2] < dense[0]  # decays
        assert dense[2] < 0.8  # collapsed by depth 3

    def test_structured_beats_similarity_at_depth(self, crossover_rows: list) -> None:  # type: ignore[type-arg]
        table = depth_crossover_table(crossover_rows)
        sims = [table[(a, 3)] for a in ("bm25", "tfidf", "dense", "hybrid_rrf", "rerank_ce")]
        assert table[("brief_graph_3hop", 3)] > max(sims)  # wins outright at d=3

    def test_none_arm_is_floor(self, crossover_rows: list) -> None:  # type: ignore[type-arg]
        assert depth_crossover_table(crossover_rows)[("none", 3)] == 0.0


class TestGoldenAblation:
    def test_ordering_holds_under_weak_stub(self, ablation_rows: list) -> None:  # type: ignore[type-arg]
        ablation = ablation_table(ablation_rows)
        # the four cells must all render (the bug this test exists to catch)
        assert set(ablation) == {
            "full_spec/none",
            "stripped/none",
            "stripped/brief",
            "stripped/random",
        }
        # full-spec needs no memory (the constraint is in the query) -> ceiling
        # stripping the spec drops the no-memory arm to the floor
        assert ablation["stripped/none"] == 0.0
        # structure recovers above the floor and beats the budget-matched random control
        assert ablation["stripped/brief"] > ablation["stripped/random"] >= ablation["stripped/none"]
        # recovery is partial under the weak offline stub: above the floor, below the ceiling
        assert ablation["stripped/none"] < ablation["stripped/brief"] < ablation["full_spec/none"]


class TestGoldenTablesAndReport:
    def test_headline_present(self, crossover_rows: list) -> None:  # type: ignore[type-arg]
        assert ("synthetic", "brief_graph_3hop") in headline_table(crossover_rows)

    def test_report_renders_with_exec_summary(
        self,
        crossover_rows: list,  # type: ignore[type-arg]
        ablation_rows: list,  # type: ignore[type-arg]
    ) -> None:
        md = generate_report([*crossover_rows, *ablation_rows])
        assert "Executive summary" in md and "Return on Tokens" in md


class TestGoldenTheory:
    def test_theory_predicts_a_finite_crossover(self) -> None:
        model = RecoveryModel(decay=SimilarityDecayModel(s0=0.9, rho=0.5), q=0.97)
        result = find_crossover_depth(model, overhead_ratio(0.2, 1.0))
        assert result.exists and result.d_star is not None and result.d_star >= 1
