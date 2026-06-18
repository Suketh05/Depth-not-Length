r"""Depth-labelling reliability: Cohen's kappa and the dual-annotation protocol.

The depth-crossover analysis is only trustworthy if the per-task depth labels are
reliable. The protocol (from CLAUDE.md) is: two annotators independently trace each
task's reasoning chain; tasks whose labels disagree by more than one hop are dropped
or rewritten; and the inter-annotator agreement is reported as Cohen's kappa so the
labelling's reliability is quantified rather than asserted.

This module implements that machinery: weighted/unweighted Cohen's kappa over ordinal
depth labels, and a certification pass that applies the disagreement rule and returns
the agreed labels plus a reliability report. The current vendored labels are a single
automated pass (``dual_annotated: false``); this is the apparatus a real second
annotation pass plugs into, and the gate the depth-crossover table should clear before
its numbers are published.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["AgreementReport", "certify_depths", "cohens_kappa"]


def _confusion(
    a: NDArray[np.int_], b: NDArray[np.int_], categories: list[int]
) -> NDArray[np.float64]:
    index = {c: i for i, c in enumerate(categories)}
    matrix = np.zeros((len(categories), len(categories)), dtype=np.float64)
    for x, y in zip(a, b, strict=True):
        matrix[index[int(x)], index[int(y)]] += 1.0
    return matrix


def cohens_kappa(
    labels_a: list[int],
    labels_b: list[int],
    *,
    weights: str | None = None,
) -> float:
    r"""Cohen's kappa between two annotators' ordinal depth labels.

    Parameters
    ----------
    labels_a, labels_b
        Paired integer labels (same length).
    weights
        ``None`` for unweighted agreement, ``"linear"`` or ``"quadratic"`` for
        ordinal weighting that penalises larger disagreements more (appropriate for
        depth, where off-by-one matters less than off-by-three).

    Returns
    -------
    float
        :math:`\kappa = 1 - \frac{\sum w_{ij} O_{ij}}{\sum w_{ij} E_{ij}}`. Returns
        1.0 when both annotators are constant and identical (no variance to correct).
    """
    a = np.asarray(labels_a, dtype=int)
    b = np.asarray(labels_b, dtype=int)
    if a.shape != b.shape:
        raise ValueError("label lists must have equal length")
    if a.size == 0:
        raise ValueError("need at least one labelled pair")

    categories = sorted(set(a.tolist()) | set(b.tolist()))
    n_cat = len(categories)
    observed = _confusion(a, b, categories)
    total = observed.sum()
    observed /= total
    row = observed.sum(axis=1, keepdims=True)
    col = observed.sum(axis=0, keepdims=True)
    expected = row @ col

    if weights is None:
        w = 1.0 - np.eye(n_cat)
    else:
        idx = np.array(categories, dtype=np.float64)
        diff = np.abs(idx[:, None] - idx[None, :])
        denom = (n_cat - 1) or 1
        w = diff / denom if weights == "linear" else (diff / denom) ** 2
        if weights not in ("linear", "quadratic"):
            raise ValueError("weights must be None, 'linear', or 'quadratic'")

    denom_e = float((w * expected).sum())
    if denom_e == 0.0:
        return 1.0  # no expected disagreement (degenerate, perfect-agreement case)
    return 1.0 - float((w * observed).sum()) / denom_e


@dataclass(frozen=True, slots=True)
class AgreementReport:
    """Inter-annotator agreement and the certified depth labels.

    Attributes
    ----------
    kappa
        Linear-weighted Cohen's kappa over the two annotators' labels.
    exact_agreement
        Fraction of tasks with identical labels.
    n
        Number of tasks labelled by both annotators.
    dropped
        Task ids whose labels disagree by more than ``max_disagreement`` hops
        (dropped per protocol).
    certified
        ``{task_id: agreed_depth}`` for the surviving tasks (the lower of the two
        labels, i.e. only certify to the depth both annotators support).
    """

    kappa: float
    exact_agreement: float
    n: int
    dropped: tuple[str, ...]
    certified: Mapping[str, int]


def certify_depths(
    annotator_a: Mapping[str, int],
    annotator_b: Mapping[str, int],
    *,
    max_disagreement: int = 1,
) -> AgreementReport:
    """Apply the dual-annotation protocol and report reliability.

    Tasks labelled by both annotators are kept unless their labels differ by more
    than ``max_disagreement`` hops (those are dropped). Surviving tasks are certified
    to the *lower* of the two labels (only the depth both annotators support), and
    linear-weighted Cohen's kappa quantifies overall reliability.
    """
    shared = sorted(set(annotator_a) & set(annotator_b))
    if not shared:
        raise ValueError("annotators share no tasks")
    a = [annotator_a[t] for t in shared]
    b = [annotator_b[t] for t in shared]
    kappa = cohens_kappa(a, b, weights="linear")
    exact = float(np.mean([x == y for x, y in zip(a, b, strict=True)]))
    dropped = tuple(t for t in shared if abs(annotator_a[t] - annotator_b[t]) > max_disagreement)
    certified = {t: min(annotator_a[t], annotator_b[t]) for t in shared if t not in set(dropped)}
    return AgreementReport(
        kappa=kappa,
        exact_agreement=exact,
        n=len(shared),
        dropped=dropped,
        certified=certified,
    )
