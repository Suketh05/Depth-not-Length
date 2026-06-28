"""Tests for the embarrassingly-parallel sweep runner.

The parallel runner must be a pure speed-up: distributing the independent
``(dataset, arm)`` cells across processes and recombining must reproduce the serial
harness output exactly (same rows, same order), and must be invariant to the worker
count. These tests pin that against the serial ``run_benchmark`` reference plus a
hand-computed structural golden, all offline and tiny.
"""

from __future__ import annotations

from dataclasses import astuple

from membench.analysis.sweep_parallel import SweepJob, run_sweep_parallel
from membench.analysis.tables import ScoredRow
from membench.harness import run_benchmark

# A deliberately tiny, fully-offline config. load_synthetic_tasks uses the fixed depth
# triple (1, 2, 3), so per_depth=1 yields exactly 3 tasks; single_shot => one
# AttemptRecord per task => one ScoredRow per task. With two arms the whole sweep is:
#   |arms| * |depths| * per_depth = 2 * 3 * 1 = 6 rows.   (hand-computed golden)
_DATASETS = ["synthetic"]
_ARMS = ["none", "bm25"]
_PER_DEPTH = 1
_BUDGET = 150
_SEED = 0
_EXPECTED_ROWS = 6  # = 2 arms x 3 synthetic depths x per_depth(1)


def _serial_rows() -> list[ScoredRow]:
    """The serial reference the parallel runner must reproduce."""
    rows: list[ScoredRow] = []
    for dataset in _DATASETS:
        rows.extend(
            run_benchmark(
                dataset,
                _ARMS,
                _BUDGET,
                per_depth=_PER_DEPTH,
                seed=_SEED,
                offline=True,
            )
        )
    return rows


def _parallel_rows(max_workers: int) -> list[ScoredRow]:
    return run_sweep_parallel(
        _DATASETS,
        _ARMS,
        _BUDGET,
        per_depth=_PER_DEPTH,
        seed=_SEED,
        offline=True,
        max_workers=max_workers,
    )


def _as_multiset(rows: list[ScoredRow]) -> list[tuple[object, ...]]:
    """Order-independent comparison key: the rows as a sorted list of field tuples."""
    return sorted((astuple(r) for r in rows), key=repr)


class TestRowCount:
    def test_hand_computed_total(self) -> None:
        # Golden: 2 arms x 3 depths x per_depth(1) = 6 rows, independent of worker count.
        assert len(_serial_rows()) == _EXPECTED_ROWS
        assert len(_parallel_rows(2)) == _EXPECTED_ROWS


class TestMatchesSerial:
    def test_parallel_set_equals_serial(self) -> None:
        # Primary invariant: same rows as the serial harness (compared order-independently).
        serial = _serial_rows()
        parallel = _parallel_rows(2)
        assert _as_multiset(parallel) == _as_multiset(serial)

    def test_parallel_order_matches_serial_exactly(self) -> None:
        # Stronger than set-equality: map() preserves submission order, and the runner
        # reassembles dataset-major/arm-major, so the ordered list is identical.
        assert _parallel_rows(2) == _serial_rows()


class TestWorkerInvariance:
    def test_one_worker_equals_many(self) -> None:
        # max_workers=1 and max_workers>1 must agree exactly (deterministic reduction).
        assert _parallel_rows(1) == _parallel_rows(3)


class TestEmptyRetrievalGolden:
    def test_none_arm_has_zero_precision_recall(self) -> None:
        # The `none` control retrieves nothing, so retrieved_ids is empty. Precision@k
        # and recall@k of an empty retrieved set are 0/|gold| = 0.0 by definition, and
        # an empty set cannot recover the chain -- a textbook golden that holds for every
        # `none` row regardless of the (offline) model output.
        none_rows = [r for r in _parallel_rows(2) if r.arm == "none"]
        assert none_rows  # the control did run
        assert all(r.recall == 0.0 for r in none_rows)
        assert all(r.precision == 0.0 for r in none_rows)
        assert all(not r.chain_recovered for r in none_rows)


class TestEmptySweep:
    def test_no_arms_returns_no_rows(self) -> None:
        assert run_sweep_parallel(_DATASETS, [], _BUDGET, per_depth=_PER_DEPTH) == []


class TestSweepJob:
    def test_job_is_frozen_and_picklable(self) -> None:
        import pickle

        job = SweepJob(
            dataset="synthetic",
            arm="none",
            budget=_BUDGET,
            per_depth=_PER_DEPTH,
            seed=_SEED,
            offline=True,
            model="claude",
            retry_policy="single_shot",
        )
        # Picklability is required to ship the job to a worker process.
        assert pickle.loads(pickle.dumps(job)) == job
