r"""Mediation and mutual information for whether retrieval explains the compliance win.

The thesis is mechanistic: structured memory helps *because* it retrieves the
governing chain, not for some incidental reason. Mediation analysis tests exactly
that. With treatment ``T`` (structured vs. similarity arm), mediator ``M`` (did the
arm recover the governing chain), and outcome ``Y`` (did the output comply):

* the **total effect** is ``E[Y|T=1] - E[Y|T=0]``;
* the **indirect (mediated) effect** is ``a * b`` where ``a = E[M|T=1]-E[M|T=0]`` and
  ``b`` is the coefficient of ``M`` in the linear-probability regression
  ``Y ~ T + M``;
* the **proportion mediated** is ``indirect / total``.

A proportion near 1 means the arm's advantage runs almost entirely through
retrieval -- the mechanism the paper claims. :func:`mutual_information` quantifies,
in bits, how much knowing whether the chain was retrieved tells you about compliance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["MediationResult", "mediation_analysis", "mutual_information"]


def mutual_information(x: Sequence[int], y: Sequence[int]) -> float:
    r"""Return the mutual information ``I(X;Y)`` in bits for two discrete variables."""
    xa = np.asarray(x)
    ya = np.asarray(y)
    if xa.shape != ya.shape or xa.size == 0:
        raise ValueError("x and y must be non-empty and aligned")
    info = 0.0
    for xv in np.unique(xa):
        px = np.mean(xa == xv)
        for yv in np.unique(ya):
            py = np.mean(ya == yv)
            pxy = np.mean((xa == xv) & (ya == yv))
            if pxy > 0:
                info += pxy * np.log2(pxy / (px * py))
    return float(max(info, 0.0))


@dataclass(frozen=True, slots=True)
class MediationResult:
    """Decomposition of a treatment effect into direct and retrieval-mediated parts."""

    total_effect: float
    direct_effect: float
    indirect_effect: float
    proportion_mediated: float


def mediation_analysis(
    treatment: Sequence[int],
    mediator: Sequence[float],
    outcome: Sequence[int],
) -> MediationResult:
    """Decompose treatment -> outcome into direct and mediator-carried effects.

    Linear-probability (OLS) mediation: appropriate and interpretable for the binary
    outcomes here. ``treatment`` is 0/1, ``mediator`` is the retrieval signal
    (e.g. chain recovered, 0/1 or a rate), ``outcome`` is 0/1 compliance.
    """
    t = np.asarray(treatment, dtype=np.float64)
    m = np.asarray(mediator, dtype=np.float64)
    y = np.asarray(outcome, dtype=np.float64)
    if not (t.shape == m.shape == y.shape) or t.size == 0:
        raise ValueError("treatment, mediator, outcome must be non-empty and aligned")
    if set(np.unique(t)) - {0.0, 1.0}:
        raise ValueError("treatment must be binary (0/1)")
    if t.sum() == 0 or t.sum() == t.size:
        raise ValueError("need both treated and control observations")

    total = float(y[t == 1].mean() - y[t == 0].mean())
    a_path = float(m[t == 1].mean() - m[t == 0].mean())

    design = np.column_stack([np.ones_like(t), t, m])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    direct = float(beta[1])
    b_path = float(beta[2])
    indirect = a_path * b_path
    proportion = indirect / total if total != 0 else 0.0
    return MediationResult(
        total_effect=total,
        direct_effect=direct,
        indirect_effect=indirect,
        proportion_mediated=proportion,
    )
