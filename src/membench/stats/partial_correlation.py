r"""Partial and semipartial correlation with a significance test.

Partial correlation measures the linear association between two variables ``x`` and
``y`` after the linear effect of one or more covariates ``Z`` has been removed from
*both*. For a single covariate ``z`` it is

.. math::

    r_{xy \cdot z} = \frac{r_{xy} - r_{xz}\,r_{yz}}
                          {\sqrt{(1 - r_{xz}^2)(1 - r_{yz}^2)}} .

The *semipartial* (part) correlation removes ``Z`` from ``x`` only, leaving ``y``
intact -- the quantity whose square is the increment in :math:`R^2` from adding ``x``
to a model already containing ``Z``:

.. math::

    r_{y(x \cdot z)} = \frac{r_{xy} - r_{xz}\,r_{yz}}{\sqrt{1 - r_{xz}^2}} .

For several covariates the partial correlation is obtained from the **precision
matrix** :math:`P = \Sigma^{-1}` (the inverse of the correlation matrix): the partial
correlation between variables ``i`` and ``j`` controlling for all the others is

.. math::

    r_{ij \cdot \text{rest}} = -\,P_{ij} \big/ \sqrt{P_{ii}\,P_{jj}} ,

which for three variables reduces exactly to the closed form above. Significance uses
the standard ``t`` test with :math:`\mathrm{df} = n - 2 - k` for ``k`` covariates:

.. math::

    t = r \sqrt{\frac{n - 2 - k}{1 - r^2}} .

References
----------
Cohen, J., Cohen, P., West, S. G., & Aiken, L. S. (2003). *Applied Multiple
Regression/Correlation Analysis for the Behavioral Sciences* (3rd ed.), chapter 3.
Lawrence Erlbaum Associates.

Whittaker, J. (1990). *Graphical Models in Applied Multivariate Statistics*. Wiley
(the precision-matrix identity for partial correlation).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

__all__ = [
    "PartialCorrelationResult",
    "partial_correlation",
    "partial_correlation_matrix",
]


@dataclass(frozen=True, slots=True)
class PartialCorrelationResult:
    """Partial and semipartial correlation of ``x`` with ``y`` given covariates.

    Attributes
    ----------
    partial
        Partial correlation ``r_{xy.Z}`` (covariates removed from both ``x`` and
        ``y``), in ``[-1, 1]``.
    semipartial
        Semipartial (part) correlation ``r_{y(x.Z)}`` (covariates removed from ``x``
        only); its square is the increment in ``R^2`` from adding ``x``.
    n
        Number of observations.
    n_covariates
        Number of covariates ``k`` controlled for.
    df
        Degrees of freedom of the significance test, ``n - 2 - k``.
    t_statistic
        ``t`` statistic testing ``H0: partial == 0``.
    p_value
        Two-sided p-value of ``t_statistic``.
    """

    partial: float
    semipartial: float
    n: int
    n_covariates: int
    df: int
    t_statistic: float
    p_value: float


def partial_correlation_matrix(corr: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Return the full partial correlation matrix via the precision matrix.

    Each off-diagonal entry ``[i, j]`` is the partial correlation of variables ``i``
    and ``j`` controlling for *all* the remaining variables, computed as
    ``-P_ij / sqrt(P_ii * P_jj)`` where ``P`` is the inverse of ``corr``. The diagonal
    is set to ``1``.

    Parameters
    ----------
    corr
        A symmetric correlation (or covariance) matrix, shape ``(p, p)`` with
        ``p >= 2``.

    Returns
    -------
    numpy.ndarray
        The ``(p, p)`` partial correlation matrix.

    Raises
    ------
    ValueError
        If ``corr`` is not square, has fewer than two variables, or is singular
        (e.g. perfectly collinear covariates).
    """
    mat = np.asarray(corr, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("corr must be a square 2-D matrix")
    if mat.shape[0] < 2:
        raise ValueError("need at least two variables")
    try:
        precision = np.linalg.inv(mat)
    except np.linalg.LinAlgError as exc:
        raise ValueError("correlation matrix is singular; covariates may be collinear") from exc
    d = np.sqrt(np.diag(precision))
    pcorr = -precision / np.outer(d, d)
    np.fill_diagonal(pcorr, 1.0)
    result: NDArray[np.float64] = pcorr
    return result


def _residualize(
    target: NDArray[np.float64], covars: list[NDArray[np.float64]]
) -> NDArray[np.float64]:
    """Return ``target`` minus its OLS projection onto an intercept and ``covars``."""
    n = target.size
    if not covars:
        centered: NDArray[np.float64] = target - target.mean()
        return centered
    design = np.column_stack([np.ones(n), *covars])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual: NDArray[np.float64] = target - design @ beta
    return residual


def partial_correlation(
    x: Sequence[float],
    y: Sequence[float],
    covariates: Sequence[Sequence[float]] = (),
) -> PartialCorrelationResult:
    """Partial and semipartial correlation of ``x`` and ``y`` controlling for covariates.

    The partial correlation is computed by the precision-matrix method on the sample
    correlation matrix of ``[x, y, *covariates]`` (equivalently, the correlation of the
    residuals of ``x`` and ``y`` after regressing each on the covariates). The
    semipartial correlation is the correlation of ``y`` with the residual of ``x`` on
    the covariates -- the covariates are removed from ``x`` only.

    Parameters
    ----------
    x, y
        Aligned one-dimensional samples of equal length ``n``.
    covariates
        A sequence of ``k`` covariate columns, each aligned with ``x`` and ``y``. An
        empty sequence reduces the result to the ordinary Pearson correlation.

    Returns
    -------
    PartialCorrelationResult
        The partial and semipartial correlations plus the ``t`` test of the partial
        correlation.

    Raises
    ------
    ValueError
        If the samples are misaligned, any variable is constant, ``x`` is perfectly
        explained by the covariates, or there are too few observations
        (``n - 2 - k < 1``).
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.ndim != 1 or ya.ndim != 1:
        raise ValueError("x and y must be one-dimensional")
    if xa.shape != ya.shape:
        raise ValueError("x and y must have the same length")
    cov_cols = [np.asarray(c, dtype=np.float64) for c in covariates]
    for c in cov_cols:
        if c.shape != xa.shape:
            raise ValueError("each covariate must align with x and y")
    n = int(xa.size)
    k = len(cov_cols)
    if n - 2 - k < 1:
        raise ValueError("not enough observations for the number of covariates")

    matrix = np.column_stack([xa, ya, *cov_cols])
    if np.any(matrix.std(axis=0) == 0):
        raise ValueError("every variable must have non-zero variance")

    corr = np.corrcoef(matrix, rowvar=False)
    partial = float(partial_correlation_matrix(corr)[0, 1])

    # Semipartial: covariates removed from x only -> corr(y, residual of x on Z).
    resid_x = _residualize(xa, cov_cols)
    if resid_x.std() == 0:
        raise ValueError("x is fully explained by the covariates")
    semipartial = float(np.corrcoef(ya, resid_x)[0, 1])

    df = n - 2 - k
    if abs(partial) >= 1.0:
        t_stat = float("inf") if partial > 0 else float("-inf")
        p_value = 0.0
    else:
        t_stat = float(partial * np.sqrt(df / (1.0 - partial**2)))
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df))
    return PartialCorrelationResult(
        partial=partial,
        semipartial=semipartial,
        n=n,
        n_covariates=k,
        df=df,
        t_statistic=t_stat,
        p_value=p_value,
    )
