"""Embarrassingly-parallel sweep runner that reproduces the serial harness exactly.

The benchmark sweep is *embarrassingly parallel*: every ``(dataset, arm)`` cell is an
independent job -- it loads its own tasks, builds a fresh arm per task, runs the offline
model, and scores the attempts -- with no shared mutable state between cells. Such a
workload is the textbook target of the **master/worker** (a.k.a. task-farming) pattern:
a coordinator decomposes the work into independent tasks, a pool of workers executes
them concurrently, and the coordinator gathers the partial results [1]_ [2]_.

Determinism (the property that makes parallel == serial) comes from *per-job seeding*:
each job is given the same ``seed`` the serial harness uses, and
:func:`membench.harness.load_tasks` is a pure function of ``(dataset, per_depth, seed)``.
Because the offline arms and offline model are deterministic, the rows a cell emits do
not depend on *when* or *in which process* it runs. The coordinator therefore reassembles
the gathered rows in the canonical serial order -- dataset-major, then arm, then task --
so the output is identical (not merely set-equal) to::

    [row
     for dataset in datasets
     for row in run_benchmark(dataset, arms, budget, per_depth=..., seed=..., ...)]

This is the *deterministic parallel reduction* invariant: distributing a pure,
seed-pinned map over a process pool and recombining in a fixed order is invariant to the
scheduler, so ``max_workers=1`` and ``max_workers>1`` yield byte-identical results [1]_.

References
----------
.. [1] I. Foster, *Designing and Building Parallel Programs*, Addison-Wesley, 1995,
   ch. 2 ("Designing Parallel Algorithms") and ch. 1 (the master/worker task-farming
   structure for embarrassingly-parallel problems). https://www.mcs.anl.gov/~itf/dbpp/
.. [2] Python Software Foundation, "concurrent.futures -- Launching parallel tasks",
   Python documentation (``ProcessPoolExecutor``). https://docs.python.org/3/library/concurrent.futures.html
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

from membench.harness import DEFAULT_ARMS, run_benchmark

if TYPE_CHECKING:
    from collections.abc import Sequence

    from membench.analysis.tables import ScoredRow

__all__ = ["SweepJob", "run_sweep_parallel"]


@dataclass(frozen=True, slots=True)
class SweepJob:
    """One independent ``(dataset, arm)`` unit of work for a parallel sweep.

    The job is a self-contained, picklable description of a single serial
    :func:`membench.harness.run_benchmark` call restricted to one arm, so a worker
    process can execute it in isolation. The fields mirror ``run_benchmark``'s
    parameters one-to-one (the fairness lock -- identical ``budget``/``model`` across
    arms -- is preserved because every job carries the same values).

    Attributes
    ----------
    dataset
        Dataset name (resolved by :func:`membench.harness.load_tasks`).
    arm
        Registry name of the single arm this job runs.
    budget
        Per-arm token budget (the shared fairness budget).
    per_depth
        Number of tasks per depth to load.
    seed
        RNG seed; the source of cross-process determinism.
    offline
        Whether to use the offline (deterministic) model/arm stubs.
    model
        Model name to run against.
    retry_policy
        Runner retry policy (``"single_shot"`` by default).
    """

    dataset: str
    arm: str
    budget: int
    per_depth: int
    seed: int
    offline: bool
    model: str
    retry_policy: str


def _run_job(job: SweepJob) -> list[ScoredRow]:
    """Execute one :class:`SweepJob` in a worker and return its scored rows.

    A module-level function (not a closure/lambda) so it is picklable for
    :class:`~concurrent.futures.ProcessPoolExecutor`. It delegates verbatim to the
    serial :func:`membench.harness.run_benchmark` -- the single execution path -- with
    the arm list pinned to ``[job.arm]``, so a cell run here is identical to that cell
    in a serial sweep.

    Parameters
    ----------
    job
        The unit of work to execute.

    Returns
    -------
    list of ScoredRow
        The scored rows for this ``(dataset, arm)`` cell.
    """
    return run_benchmark(
        job.dataset,
        [job.arm],
        job.budget,
        per_depth=job.per_depth,
        seed=job.seed,
        offline=job.offline,
        model=job.model,
        retry_policy=job.retry_policy,
    )


def run_sweep_parallel(
    datasets: Sequence[str],
    arms: Sequence[str] = DEFAULT_ARMS,
    budget: int = 150,
    *,
    per_depth: int = 10,
    seed: int = 0,
    offline: bool = True,
    model: str = "claude",
    retry_policy: str = "single_shot",
    max_workers: int | None = None,
) -> list[ScoredRow]:
    """Run the sweep across processes, returning rows identical to the serial harness.

    Distributes each ``(dataset, arm)`` cell to a
    :class:`~concurrent.futures.ProcessPoolExecutor` worker via the master/worker
    pattern, then reassembles the gathered rows in the canonical serial order
    (dataset-major, then arm, then task). Because each cell is a pure, ``seed``-pinned
    call to :func:`membench.harness.run_benchmark`, the result is identical -- same rows
    in the same order -- to running the cells serially, and is invariant to
    ``max_workers`` (``1`` and ``>1`` agree).

    Parameters
    ----------
    datasets
        Dataset names to sweep (each resolved by
        :func:`membench.harness.load_tasks`).
    arms
        Registry names of the arms to run on every dataset. Defaults to
        :data:`membench.harness.DEFAULT_ARMS`.
    budget
        Per-arm token budget shared by all arms (the fairness lock).
    per_depth
        Number of tasks per depth to load.
    seed
        RNG seed propagated unchanged to every job (per-job seeding); the source of
        cross-process determinism.
    offline
        Run the offline (deterministic) model/arm stubs.
    model
        Model name to run against.
    retry_policy
        Runner retry policy.
    max_workers
        Maximum worker processes. ``None`` lets the executor choose
        (``os.cpu_count()``). ``1`` runs every job in a single worker; the result is
        identical to any larger value.

    Returns
    -------
    list of ScoredRow
        The scored rows for the whole sweep, ordered identically to::

            [row
             for dataset in datasets
             for row in run_benchmark(dataset, arms, budget, per_depth=per_depth,
                                      seed=seed, offline=offline, model=model,
                                      retry_policy=retry_policy)]

    Notes
    -----
    The set of rows is independent of scheduling order; the explicit reassembly only
    fixes their *order* so the output matches the serial harness exactly rather than
    merely as a set.
    """
    jobs = [
        SweepJob(
            dataset=dataset,
            arm=arm,
            budget=budget,
            per_depth=per_depth,
            seed=seed,
            offline=offline,
            model=model,
            retry_policy=retry_policy,
        )
        for dataset in datasets
        for arm in arms
    ]
    if not jobs:
        return []

    # ProcessPoolExecutor.map preserves submission order, so flattening its output in
    # place reproduces the serial dataset-major/arm-major/task ordering exactly -- no
    # re-sort needed, and duplicate cells (a repeated arm) are run and kept as in serial.
    ordered: list[ScoredRow] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for rows in executor.map(_run_job, jobs):
            ordered.extend(rows)
    return ordered
