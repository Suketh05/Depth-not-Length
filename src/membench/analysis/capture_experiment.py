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
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CAPTURE_RETRIEVERS",
    "DEFAULT_BUDGET",
    "DEFAULT_DEPTHS",
    "DEFAULT_PER_DEPTH",
    "CaptureRow",
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
