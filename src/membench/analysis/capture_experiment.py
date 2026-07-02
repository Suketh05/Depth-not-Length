"""Driver for the capture experiment: three storage conditions, recall by depth.

Runs the experiment of paper Section ``sec:capture`` ("The Capture Experiment:
Links, Not Discreteness") and produces the rows behind Table ``tab:capexp``:
recall by depth (d1/d2/d3) of the governing decision under the three storage
conditions of Subsection ``sec:capconds``, per retriever, "model-free, over the
identical queries and budget". The experiment is retrieval-side only -- no
language model is in the loop ("the organization sweep ... and the capture
experiment ... are retrieval-side and model-free: run in the offline harness
without a language model, on the synthetic corpus where depth and scatter are
controlled by construction", Section ``sec:disc``) -- so the driver calls each
arm's ``write``/``retrieve`` contract directly and scores the returned ids with
the standard retrieval metrics.

Design, mirroring the paper's protocol:

* **Paired conditions.** The same synthetic tasks (identical ``task_id``,
  query, depth, budget) are transformed into CAPTURED / Discrete-no-links /
  Raw-scattered via :func:`membench.datasets.capture_conditions.apply_condition`
  -- "n=40 tasks per depth, paired across conditions" (Table ``tab:capexp``
  caption); :data:`DEFAULT_PER_DEPTH` matches that n.
* **The four retrievers of the table.** :data:`CAPTURE_RETRIEVERS` is exactly
  the retriever column of Table ``tab:capexp``: Brief (the typed-graph arm
  ``brief_graph_3hop``), ``bm25``, ``tfidf``, ``dense``.
* **Fairness lock.** Every arm is built fresh per task (no state leakage) and
  retrieves under the same token budget (:data:`DEFAULT_BUDGET`, the harness
  default), so the storage condition is the only variable within a retriever
  and the retriever the only variable within a condition.
* **Recall semantics.** Recall is scored against the task's gold ids by
  :func:`membench.metrics.retrieval.score_retrieval`. Under Raw-scattered the
  gold ids are the governing decision's fragments, so recall is the fraction of
  the decision *assembled* -- the scatter-assembly event of Theorem
  ``thm:scatter``.

What this driver deliberately does not do: assert or reproduce the published
cell values. The paper's headline ladder (Brief d3: 1.00 -> 0.42 -> ~0.33 in
prose; 0.96 -> 0.76 -> 0.47 in the table cells -- the two differ, see
:mod:`membench.datasets.capture_conditions`) belongs to the measured run;
this module produces well-formed, paired rows and their tab:capexp-shaped
aggregation from whatever corpus and parameters it is given.

Examples
--------
Reproduce the tab:capexp protocol at full size (offline, deterministic)::

    from membench.analysis.capture_experiment import (
        render_capture_table,
        run_capture_experiment,
    )

    rows = run_capture_experiment()          # 3 conditions x 4 retrievers x 120 tasks
    print(render_capture_table(rows))        # retriever x condition x d1/d2/d3

The qualitative shape to expect (paper conclusions (i)/(ii), Subsection
``sec:capconc``): the graph arm is depth-flat under ``captured``, drops under
``discrete_no_links`` while the lexical arms' rows are unchanged, and every
arm falls into a common low band under ``raw_scattered``.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from membench.datasets.capture_conditions import (
    DEFAULT_SIGMA,
    CaptureCondition,
    apply_condition,
)
from membench.datasets.synthetic import load_synthetic_tasks
from membench.metrics.retrieval import score_retrieval
from membench.retrieval import bm25, dense, tfidf  # noqa: F401  (register the lexical/dense arms)
from membench.retrieval.base import build_arm
from membench.retrieval.graph import arm as _graph_arm  # noqa: F401  (register the graph arm)
from membench.types import Task

__all__ = [
    "CAPTURE_RETRIEVERS",
    "DEFAULT_BUDGET",
    "DEFAULT_DEPTHS",
    "DEFAULT_PER_DEPTH",
    "CaptureRow",
    "capture_table",
    "render_capture_table",
    "run_capture_experiment",
]

CAPTURE_RETRIEVERS: tuple[str, ...] = ("brief_graph_3hop", "bm25", "tfidf", "dense")
"""The retriever column of Table ``tab:capexp``, in the table's row order.

``brief_graph_3hop`` is the paper's "Brief" arm (the typed-graph store; the
3-hop variant is the one every synthetic experiment uses, cf.
``membench.harness.DEFAULT_ARMS``); the other three are the similarity
baselines the table reports.
"""

DEFAULT_DEPTHS: tuple[int, ...] = (1, 2, 3)
"""Depth columns d1/d2/d3 of Table ``tab:capexp``."""

DEFAULT_PER_DEPTH = 40
"""Tasks per depth: "Recall is over n=40 tasks per depth, paired across
conditions" (Table ``tab:capexp`` caption)."""

DEFAULT_BUDGET = 150
"""Retrieval token budget, identical across arms and conditions (the harness
default budget, cf. ``membench.harness.run_benchmark``)."""


@dataclass(frozen=True, slots=True)
class CaptureRow:
    """One (retriever, condition, task) retrieval outcome -- the experiment's row.

    ``recall`` and ``chain_recovered`` follow
    :class:`membench.metrics.retrieval.RetrievalScore`; ``n_gold`` makes the
    Raw-scattered assembly burden visible (it equals the fragment count of the
    governing decision there, 1 elsewhere).
    """

    retriever: str
    condition: str
    depth: int
    task_id: str
    recall: float
    chain_recovered: bool
    n_gold: int
    n_retrieved: int


def run_capture_experiment(
    retrievers: Sequence[str] = CAPTURE_RETRIEVERS,
    *,
    depths: tuple[int, ...] = DEFAULT_DEPTHS,
    per_depth: int = DEFAULT_PER_DEPTH,
    budget: int = DEFAULT_BUDGET,
    sigma: int = DEFAULT_SIGMA,
    seed: int = 0,
    tasks: Sequence[Task] | None = None,
) -> list[CaptureRow]:
    """Run all three storage conditions through the offline retrieval harness.

    For every condition of Subsection ``sec:capconds``, transforms the same
    task set (pairing), then for every retriever and task builds a *fresh* arm,
    writes the condition's corpus, retrieves under the shared budget, and
    scores the returned ids against the condition's gold ids. Deterministic:
    the synthetic corpus, the shred, and every default arm are seeded or
    seed-free deterministic, so the same arguments always yield identical rows.

    Parameters
    ----------
    retrievers
        Registry names of the arms to run (default: the Table ``tab:capexp``
        retriever column).
    depths, per_depth, seed
        Passed to :func:`membench.datasets.synthetic.load_synthetic_tasks`
        when ``tasks`` is not supplied; ``seed`` also seeds the shred.
    budget
        Token budget shared by every arm, condition, and task.
    sigma
        Fragment count for the Raw-scattered condition.
    tasks
        Optional explicit CAPTURED-condition task list (e.g. a hand-built
        corpus in tests); when given, ``depths``/``per_depth`` are not used to
        generate tasks.

    Returns
    -------
    list of CaptureRow
        One row per (condition, retriever, task), in condition-major order
        (conditions in :class:`CaptureCondition` declaration order, then
        ``retrievers`` order, then task order).
    """
    base_tasks: Sequence[Task] = (
        load_synthetic_tasks(depths=depths, per_depth=per_depth, seed=seed)
        if tasks is None
        else tasks
    )
    rows: list[CaptureRow] = []
    for condition in CaptureCondition:
        transformed = [
            apply_condition(task, condition, sigma=sigma, seed=seed) for task in base_tasks
        ]
        for retriever in retrievers:
            for task in transformed:
                arm = build_arm(retriever)  # fresh per task: no cross-task state leakage
                arm.write(task.memory_corpus)
                retrieved = arm.retrieve(task.query, budget)
                score = score_retrieval(retrieved.ids, task.governing_decisions)
                rows.append(
                    CaptureRow(
                        retriever=retriever,
                        condition=condition.value,
                        depth=task.depth,
                        task_id=task.task_id,
                        recall=score.recall,
                        chain_recovered=score.chain_recovered,
                        n_gold=score.gold,
                        n_retrieved=score.retrieved,
                    )
                )
    return rows


def capture_table(rows: Sequence[CaptureRow]) -> dict[tuple[str, str, int], float]:
    """Aggregate rows into the Table ``tab:capexp`` cells: mean recall per cell.

    Parameters
    ----------
    rows
        Rows from :func:`run_capture_experiment`.

    Returns
    -------
    dict
        ``(retriever, condition, depth) -> mean recall`` over the tasks in that
        cell -- exactly the quantity a Table ``tab:capexp`` cell reports.
    """
    groups: dict[tuple[str, str, int], list[float]] = {}
    for row in rows:
        groups.setdefault((row.retriever, row.condition, row.depth), []).append(row.recall)
    return {key: statistics.mean(values) for key, values in groups.items()}


def render_capture_table(rows: Sequence[CaptureRow]) -> str:
    """Render rows as a plain-text Table ``tab:capexp``: retriever x condition x depth.

    Retrievers appear in first-appearance order, conditions in
    :class:`CaptureCondition` declaration order (the paper's degradation
    order), depths sorted ascending; cells are mean recall to two decimals,
    matching the table's precision. Cells with no rows render as ``--``.

    Parameters
    ----------
    rows
        Rows from :func:`run_capture_experiment`.

    Returns
    -------
    str
        A fixed-width text table, one line per (retriever, condition).
    """
    cells = capture_table(rows)
    retrievers = list(dict.fromkeys(row.retriever for row in rows))
    depths = sorted({row.depth for row in rows})
    conditions = [c.value for c in CaptureCondition]

    name_width = max([len("retriever"), *(len(r) for r in retrievers)], default=len("retriever"))
    cond_width = max(len("condition"), *(len(c) for c in conditions))
    header = f"{'retriever':<{name_width}}  {'condition':<{cond_width}}  " + "  ".join(
        f"{f'd{d}':>5}" for d in depths
    )
    lines = [header.rstrip()]
    for retriever in retrievers:
        for condition in conditions:
            values = []
            for depth in depths:
                cell = cells.get((retriever, condition, depth))
                values.append("   --" if cell is None else f"{cell:>5.2f}")
            line = f"{retriever:<{name_width}}  {condition:<{cond_width}}  " + "  ".join(values)
            lines.append(line.rstrip())
    return "\n".join(lines)
