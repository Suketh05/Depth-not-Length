r"""G-computation (standardization) for the causal effect of arm on compliance.

Where :mod:`membench.stats.mediation` asks *through what* the structured arm wins,
g-computation asks *how big the win is once confounding is removed*. Tasks are not
randomized to arms at the same depth -- deeper tasks may be over- or under-represented
in an arm's results -- so a crude difference in compliance confounds the arm effect
with depth :math:`L`. The g-formula (Robins, 1986) removes that by *standardizing*:
fit an outcome model :math:`\hat{E}[Y \mid A, L]`, then average each unit's predicted
counterfactual outcomes over the *empirical* covariate distribution.

.. math::

    \widehat{\mathrm{ATE}}
        = \frac{1}{n}\sum_{i=1}^{n}
          \Bigl[\hat{E}\bigl(Y \mid A=1, L=L_i\bigr)
              - \hat{E}\bigl(Y \mid A=0, L=L_i\bigr)\Bigr].

This is the plug-in / parametric g-formula: predict every unit's outcome under the
treated regime (:math:`A=1`) and under the control regime (:math:`A=0`) holding its
own covariates fixed, then take the difference of the means. Under the usual
identifiability assumptions -- consistency, conditional exchangeability given
:math:`L`, and positivity -- :math:`\widehat{\mathrm{ATE}}` estimates the average
treatment effect :math:`E[Y^{a=1}] - E[Y^{a=0}]`. With no confounding (``A`` balanced
across the covariate distribution) the estimate collapses to the crude difference
:math:`\bar{Y}_{A=1} - \bar{Y}_{A=0}`.

References
----------
Robins, J. (1986). A new approach to causal inference in mortality studies with a
sustained exposure period -- application to control of the healthy worker survivor
effect. *Mathematical Modelling*, 7(9-12), 1393-1512.

Hernan, M. A., & Robins, J. M. (2020). *Causal Inference: What If*, Chapter 13
(Standardization and the parametric g-formula). Boca Raton: Chapman & Hall/CRC.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["GComputationResult", "g_computation"]


@dataclass(frozen=True, slots=True)
class GComputationResult:
    """Standardized average treatment effect from the parametric g-formula.

    Attributes
    ----------
    ate
        Standardized average treatment effect, ``mean_y1 - mean_y0``.
    mean_y1
        Mean predicted outcome with everyone set to treated (``A=1``), averaged
        over the empirical covariate distribution.
    mean_y0
        Mean predicted outcome with everyone set to control (``A=0``), averaged
        over the empirical covariate distribution.
    n
        Number of observations the outcome model was fit on.
    """

    ate: float
    mean_y1: float
    mean_y0: float
    n: int


def _design_matrix(
    a: NDArray[np.float64],
    cov: NDArray[np.float64],
    *,
    interaction: bool,
) -> NDArray[np.float64]:
    """Build the OLS design ``[1, A, L, (A*L)]`` for a treatment vector ``a``.

    Parameters
    ----------
    a
        Treatment assignment column, shape ``(n,)``. Pass an all-ones or all-zeros
        vector to construct a counterfactual design.
    cov
        Covariate matrix, shape ``(n, p)``.
    interaction
        If ``True`` append one ``A * L_j`` column per covariate, allowing the
        treatment effect to be modified by the covariates.

    Returns
    -------
    numpy.ndarray
        Design matrix of shape ``(n, k)``.
    """
    n = a.shape[0]
    columns: list[NDArray[np.float64]] = [np.ones(n, dtype=np.float64), a]
    columns.extend(cov.T)
    if interaction:
        columns.extend(a * column for column in cov.T)
    return np.column_stack(columns)


def g_computation(
    treatment: Sequence[int],
    outcome: Sequence[float],
    covariates: Sequence[float] | Sequence[Sequence[float]],
    *,
    interaction: bool = False,
) -> GComputationResult:
    r"""Estimate the standardized ATE of ``treatment`` on ``outcome`` adjusting for ``covariates``.

    Fits a linear outcome model :math:`\hat{E}[Y \mid A, L]` by ordinary least
    squares, then averages the unit-level contrasts of predicted counterfactual
    outcomes (everyone treated vs. everyone control) over the empirical covariate
    distribution -- the plug-in g-formula of Robins (1986).

    Parameters
    ----------
    treatment
        Binary treatment ``A`` per observation (``1`` = treated, e.g. the structured
        arm; ``0`` = control). Both groups must be present (positivity).
    outcome
        Outcome ``Y`` per observation (e.g. 0/1 compliance, or a compliance rate).
    covariates
        Confounder(s) ``L`` to adjust for, e.g. task depth. A 1-D sequence is treated
        as a single covariate column; a 2-D sequence is ``(n, p)`` with one column
        per covariate.
    interaction
        If ``True`` include treatment-by-covariate interaction terms so the effect
        may vary with ``L`` (effect modification). Default ``False`` fits the
        additive model :math:`\beta_0 + \beta_A A + \beta_L^\top L`.

    Returns
    -------
    GComputationResult
        The standardized ATE and the two counterfactual means.

    Raises
    ------
    ValueError
        If the inputs are empty or misaligned, ``treatment`` is not binary, or only
        one treatment group is present.

    Notes
    -----
    For the additive model the contrast is constant across units, so the estimate
    equals the OLS coefficient on ``A``; the explicit averaging is what generalizes
    correctly once ``interaction=True`` makes the contrast covariate-dependent.
    """
    a = np.asarray(treatment, dtype=np.float64)
    y = np.asarray(outcome, dtype=np.float64)
    cov = np.asarray(covariates, dtype=np.float64)
    if cov.ndim == 1:
        cov = cov.reshape(-1, 1)
    if a.ndim != 1 or y.ndim != 1 or cov.ndim != 2:
        raise ValueError("treatment and outcome must be 1-D and covariates 1-D or 2-D")
    n = a.shape[0]
    if n == 0 or y.shape[0] != n or cov.shape[0] != n:
        raise ValueError("treatment, outcome, covariates must be non-empty and aligned")
    if set(np.unique(a)) - {0.0, 1.0}:
        raise ValueError("treatment must be binary (0/1)")
    if a.sum() == 0 or a.sum() == n:
        raise ValueError("need both treated and control observations (positivity)")

    design = _design_matrix(a, cov, interaction=interaction)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)

    y1 = _design_matrix(np.ones(n, dtype=np.float64), cov, interaction=interaction) @ beta
    y0 = _design_matrix(np.zeros(n, dtype=np.float64), cov, interaction=interaction) @ beta
    mean_y1 = float(y1.mean())
    mean_y0 = float(y0.mean())
    return GComputationResult(
        ate=mean_y1 - mean_y0,
        mean_y1=mean_y1,
        mean_y0=mean_y0,
        n=n,
    )
