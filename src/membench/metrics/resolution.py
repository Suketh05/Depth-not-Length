"""SWE-ContextBench resolution scoring (paper ``sec:pubbench``, ``tab:stdbench``).

The paper's ``tab:stdbench`` "SWE-ContextBench resolution (%)" row (Brief
47.3% vs. Unabyss 37.6% for the runner-up; cited for provenance, never
asserted here) reports, per system, the fraction of code tasks *resolved*.
Resolution is the SWE-bench outcome (Jimenez et al.): a task is resolved iff
the produced change goes in as written and the benchmark's own harness accepts
it -- a binary event per task, with the benchmark score the resolved fraction.
The multi-turn token-economics sections (``sec:tokecon``,
``tab:tok_agent_economics``) reuse the same binary outcome per session
("resolved" / "unresolved" ledger rows).

Two scoring paths, one contract:

* **Harness hook** (:class:`ResolutionJudge`): the real predicate -- apply the
  patch, run the benchmark's gold tests -- supplied by a live harness. The
  paper's numbers come from "each benchmark's own harness under one fixed
  configuration" (``sec:pubbench``).
* **Deterministic offline proxy**: this repo's runner is a text harness (no
  repo checkout, no test execution), so the default resolution predicate is
  the merge-ready proxy of :mod:`membench.metrics.correctness` -- non-empty,
  non-refusal, every governing decision honoured. The dependency is
  deliberate and documented there: when a real diff+test harness is wired in
  it replaces the proxy *at this seam* without touching the compliance
  contract.

The paper reports the ``swe3`` compliance slice with n = 12 tasks explicitly
inside the noise band (``sec:pubbench``); :func:`resolution_by_slice` exists
so per-slice n is always carried next to per-slice rate and small slices
cannot be silently pooled away.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from membench.metrics.compliance import ComplianceResult
from membench.metrics.correctness import score_correctness

__all__ = [
    "ResolutionJudge",
    "ResolutionRate",
    "TaskResolution",
    "resolution_by_slice",
    "resolution_rate",
    "score_resolution",
]


@dataclass(frozen=True, slots=True)
class TaskResolution:
    """The binary resolution outcome of one task, with the reason recorded."""

    task_id: str
    resolved: bool
    reason: str


class ResolutionJudge(Protocol):
    """Harness hook: the benchmark's own resolved/not predicate for one task.

    A live SWE-ContextBench harness implements this as "the patch applies and
    the gold fail-to-pass tests pass"; the offline pipeline substitutes the
    deterministic merge-ready proxy instead (see module docstring). The judge
    sees the task id and the raw response text only -- like the answer judge
    of :mod:`membench.metrics.answer_scoring`, it carries no arm identity.
    """

    def __call__(self, task_id: str, response_text: str) -> bool:
        """Return ``True`` iff the benchmark harness accepts the response."""
        ...


def score_resolution(
    task_id: str,
    response_text: str,
    compliance: ComplianceResult,
    judge: ResolutionJudge | None = None,
) -> TaskResolution:
    """Score one task as resolved / not resolved.

    With a ``judge`` (live harness), the verdict is the harness's own
    resolved predicate. Without one, the deterministic proxy applies:
    resolved iff :func:`membench.metrics.correctness.score_correctness`
    deems the response merge-ready (non-empty, non-refusal, every governing
    decision honoured) -- the offline stand-in the module docstring
    motivates, with the correctness reason carried through verbatim.

    Parameters
    ----------
    task_id
        Task identifier (joins the outcome back to slices and ledgers).
    response_text
        The system's produced change / answer for the task.
    compliance
        Compliance result for the same response (proxy path input; the
        harness path ignores it by design -- tests decide, not keywords).
    judge
        Optional live-harness predicate; overrides the proxy entirely.

    Returns
    -------
    TaskResolution
        Binary outcome with its reason ("harness verdict" on the judge path).
    """
    if judge is not None:
        return TaskResolution(
            task_id=task_id,
            resolved=judge(task_id, response_text),
            reason="harness verdict",
        )
    correctness = score_correctness(response_text, compliance)
    return TaskResolution(
        task_id=task_id,
        resolved=correctness.merge_ready,
        reason=correctness.reason,
    )


@dataclass(frozen=True, slots=True)
class ResolutionRate:
    """Resolved count, total, and their ratio -- the ``tab:stdbench`` statistic.

    ``rate = resolved / total`` in ``[0, 1]``; ``tab:stdbench`` reports it in
    percent. The integer counts ride along so small-n slices (the paper's
    ``swe3`` slice has n = 12) are always visibly small.
    """

    resolved: int
    total: int
    rate: float


def resolution_rate(resolutions: Sequence[TaskResolution]) -> ResolutionRate:
    """Aggregate per-task outcomes into the benchmark resolution rate.

    Parameters
    ----------
    resolutions
        Per-task binary outcomes from :func:`score_resolution`.

    Returns
    -------
    ResolutionRate
        ``resolved / total`` with both counts.

    Raises
    ------
    ValueError
        If ``resolutions`` is empty -- zero tasks has no rate, and a silent
        0 would fabricate a benchmark number.
    """
    if not resolutions:
        raise ValueError("resolution_rate needs at least one task outcome")
    resolved = sum(outcome.resolved for outcome in resolutions)
    return ResolutionRate(
        resolved=resolved,
        total=len(resolutions),
        rate=resolved / len(resolutions),
    )


def resolution_by_slice(
    resolutions: Sequence[TaskResolution],
    slice_of: Mapping[str, str],
) -> dict[str, ResolutionRate]:
    """Resolution rate per named slice (e.g. the paper's ``swe3`` slice).

    Every task must be assigned a slice: a task id missing from ``slice_of``
    raises rather than being silently dropped or pooled, so a slice's n is
    exactly the number of tasks the caller placed in it.

    Parameters
    ----------
    resolutions
        Per-task binary outcomes.
    slice_of
        Maps each ``task_id`` to its slice label.

    Returns
    -------
    dict of str to ResolutionRate
        One rate per slice label, keyed by label.

    Raises
    ------
    ValueError
        If a task id has no slice label, or if ``resolutions`` is empty.
    """
    if not resolutions:
        raise ValueError("resolution_by_slice needs at least one task outcome")
    grouped: dict[str, list[TaskResolution]] = {}
    for outcome in resolutions:
        label = slice_of.get(outcome.task_id)
        if label is None:
            raise ValueError(f"task {outcome.task_id!r} has no slice label")
        grouped.setdefault(label, []).append(outcome)
    return {label: resolution_rate(members) for label, members in grouped.items()}
