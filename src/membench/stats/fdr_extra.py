r"""Storey q-values and Benjamini-Yekutieli FDR control under dependence.

This module complements the Holm-Bonferroni and Benjamini-Hochberg (BH)
procedures already in :mod:`membench.stats.multiple_comparisons` with two
false-discovery-rate (FDR) tools that are *not* duplicated there:

* :func:`storey_pi0` / :func:`storey_qvalues` -- Storey's q-value methodology.
  The proportion of true nulls :math:`\pi_0` is estimated from the empirical
  density of large p-values, and the q-value of a feature is the minimum FDR at
  which that feature is called significant. Because :math:`\pi_0 \le 1`, Storey
  q-values are uniformly no larger than (and usually tighter than) the BH
  adjusted p-values, recovering BH exactly when :math:`\pi_0 = 1`.
* :func:`benjamini_yekutieli_adjust` -- the Benjamini-Yekutieli (BY) step-up
  procedure, which controls the FDR under *arbitrary* dependence between the test
  statistics by inflating the BH threshold by the harmonic factor
  :math:`c(m) = \sum_{i=1}^{m} 1/i`. This is the conservative choice when the
  benchmark's per-arm/per-task p-values cannot be assumed independent or PRDS.

Formulae
--------
Let :math:`p_{(1)} \le \dots \le p_{(m)}` be the ordered p-values.

Storey :math:`\pi_0` at a tuning parameter :math:`\lambda \in [0, 1)`::

    pi0(lambda) = #{ p_i > lambda } / ( m * (1 - lambda) )

For a grid of :math:`\lambda` values a natural cubic spline of
:math:`\hat\pi_0(\lambda)` against :math:`\lambda` is fitted and evaluated at the
largest :math:`\lambda` (the "smoother" of Storey & Tibshirani 2003); a
single-element grid reduces to the original fixed-:math:`\lambda` estimator of
Storey (2002). The estimate is clamped to :math:`(0, 1]`.

Storey q-value (step-up, monotone)::

    q(p_(i)) = min_{ k >= i }  pi0 * m * p_(k) / k

Benjamini-Yekutieli adjusted p-value::

    p^{BY}_(i) = min( 1, min_{ k >= i }  c(m) * m * p_(k) / k ),
    c(m) = sum_{i=1}^{m} 1/i

References
----------
.. [1] Storey, J. D. (2002). "A direct approach to false discovery rates."
       Journal of the Royal Statistical Society, Series B, 64(3), 479-498.
.. [2] Storey, J. D., & Tibshirani, R. (2003). "Statistical significance for
       genomewide studies." PNAS, 100(16), 9440-9445.
.. [3] Benjamini, Y., & Yekutieli, D. (2001). "The control of the false
       discovery rate in multiple testing under dependency." Annals of
       Statistics, 29(4), 1165-1188.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import UnivariateSpline

__all__ = [
    "StoreyQValues",
    "benjamini_yekutieli_adjust",
    "harmonic_factor",
    "storey_pi0",
    "storey_qvalues",
]

# Default lambda grid for the Storey-Tibshirani smoother: 0.05, 0.10, ..., 0.95.
DEFAULT_LAMBDAS: tuple[float, ...] = tuple(round(0.05 * j, 2) for j in range(1, 20))


@dataclass(frozen=True, slots=True)
class StoreyQValues:
    """Storey q-values together with the estimated null proportion.

    Attributes
    ----------
    pi0 : float
        Estimated proportion of true null hypotheses, in ``(0, 1]``.
    qvalues : tuple[float, ...]
        q-value for each input p-value, in the original input order. Each
        q-value lies in ``[0, 1]`` and is monotone in the underlying p-value.
    """

    pi0: float
    qvalues: tuple[float, ...]


def _as_pvalues(pvalues: Sequence[float]) -> NDArray[np.float64]:
    """Validate and return p-values as a float array.

    Parameters
    ----------
    pvalues : Sequence[float]
        Raw p-values, each in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        The p-values as a 1-D ``float64`` array.

    Raises
    ------
    ValueError
        If the sequence is empty or any value falls outside ``[0, 1]``.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    if p.size == 0:
        raise ValueError("need at least one p-value")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("p-values must lie in [0, 1]")
    return p


def harmonic_factor(m: int) -> float:
    r"""Return the Benjamini-Yekutieli harmonic factor :math:`c(m)=\sum_{i=1}^m 1/i`.

    Parameters
    ----------
    m : int
        Number of hypotheses tested (``m >= 1``).

    Returns
    -------
    float
        The ``m``-th harmonic number, the dependence-correction multiplier used
        by the BY procedure.

    Raises
    ------
    ValueError
        If ``m < 1``.
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    return float(np.sum(1.0 / np.arange(1, m + 1, dtype=np.float64)))


def _step_up(p: NDArray[np.float64], factor: float) -> NDArray[np.float64]:
    """Apply a step-up FDR transform with a constant front multiplier.

    Computes ``factor * m * p_(k) / k`` on the ordered p-values, enforces
    monotonicity by a running minimum from largest to smallest, and clips to
    ``[0, 1]``. With ``factor = 1`` this is Benjamini-Hochberg; with
    ``factor = c(m)`` it is Benjamini-Yekutieli; with ``factor = pi0`` it is the
    Storey q-value transform.

    Parameters
    ----------
    p : numpy.ndarray
        Validated 1-D array of p-values.
    factor : float
        Constant multiplier applied to every scaled p-value.

    Returns
    -------
    numpy.ndarray
        The adjusted values in the original input order.
    """
    n = p.size
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=np.float64)
    running = 1.0
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        running = min(running, factor * n * p[idx] / (rank + 1))
        adjusted[idx] = running
    return np.clip(adjusted, 0.0, 1.0)


def benjamini_yekutieli_adjust(pvalues: Sequence[float]) -> list[float]:
    r"""Return Benjamini-Yekutieli adjusted p-values (FDR under dependence).

    The BY procedure controls the FDR under arbitrary dependence by scaling the
    Benjamini-Hochberg step-up threshold by the harmonic factor
    :math:`c(m)=\sum_{i=1}^m 1/i`::

        p^{BY}_(i) = min( 1, min_{k>=i} c(m) * m * p_(k) / k )

    Parameters
    ----------
    pvalues : Sequence[float]
        Raw p-values, each in ``[0, 1]``.

    Returns
    -------
    list[float]
        Adjusted p-values in the original input order, each in ``[0, 1]`` and
        monotone in the underlying p-value.

    References
    ----------
    Benjamini & Yekutieli (2001), Annals of Statistics 29(4), 1165-1188.
    """
    p = _as_pvalues(pvalues)
    adjusted = _step_up(p, harmonic_factor(p.size))
    return [float(x) for x in adjusted]


def storey_pi0(pvalues: Sequence[float], *, lambdas: Sequence[float] = DEFAULT_LAMBDAS) -> float:
    r"""Estimate the null proportion :math:`\pi_0` (Storey 2002 / Storey-Tibshirani 2003).

    For each :math:`\lambda` the estimator is
    :math:`\hat\pi_0(\lambda) = \#\{p_i > \lambda\} / (m(1-\lambda))`. With a
    single :math:`\lambda` this is the fixed-tuning estimator of Storey (2002);
    with the default grid a natural cubic spline of :math:`\hat\pi_0(\lambda)`
    against :math:`\lambda` is fitted and evaluated at the largest :math:`\lambda`
    (the smoother of Storey & Tibshirani 2003). The result is clamped to
    ``(0, 1]``.

    Parameters
    ----------
    pvalues : Sequence[float]
        Raw p-values, each in ``[0, 1]``.
    lambdas : Sequence[float], optional
        Tuning grid; values must lie in ``[0, 1)`` and be strictly increasing.
        A grid with four or more points enables the cubic-spline smoother.

    Returns
    -------
    float
        Estimated proportion of true nulls, in ``(0, 1]``.

    Raises
    ------
    ValueError
        If ``lambdas`` is empty, out of ``[0, 1)``, not strictly increasing, or
        if the resulting estimate is non-positive.
    """
    p = _as_pvalues(pvalues)
    lam = np.asarray(lambdas, dtype=np.float64)
    if lam.size == 0:
        raise ValueError("need at least one lambda")
    if np.any(lam < 0.0) or np.any(lam >= 1.0):
        raise ValueError("lambda values must lie in [0, 1)")
    if lam.size > 1 and np.any(np.diff(lam) <= 0.0):
        raise ValueError("lambda values must be strictly increasing")

    pi0_lam = np.array([np.mean(p > value) / (1.0 - value) for value in lam], dtype=np.float64)

    if lam.size < 4:
        estimate = float(pi0_lam[-1])
    else:
        spline = UnivariateSpline(lam, pi0_lam, k=3)
        estimate = float(spline(lam[-1]))

    estimate = min(estimate, 1.0)
    if estimate <= 0.0:
        raise ValueError("estimated pi0 <= 0; supply a smaller lambda or more p-values")
    return estimate


def storey_qvalues(
    pvalues: Sequence[float],
    *,
    pi0: float | None = None,
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
) -> StoreyQValues:
    r"""Compute Storey q-values for a vector of p-values.

    The q-value of the :math:`i`-th ordered p-value is the step-up quantity
    :math:`q(p_{(i)}) = \min_{k \ge i} \pi_0\, m\, p_{(k)} / k`, which is the
    minimum FDR at which that feature is declared significant. When
    :math:`\pi_0 = 1` these coincide with the Benjamini-Hochberg adjusted
    p-values.

    Parameters
    ----------
    pvalues : Sequence[float]
        Raw p-values, each in ``[0, 1]``.
    pi0 : float, optional
        Null proportion to use. If ``None`` it is estimated with
        :func:`storey_pi0` on ``lambdas``. Must lie in ``(0, 1]`` if given.
    lambdas : Sequence[float], optional
        Grid passed to :func:`storey_pi0` when ``pi0`` is not supplied.

    Returns
    -------
    StoreyQValues
        The estimated/used ``pi0`` and the q-values in input order. q-values are
        monotone in the p-values and bounded by ``1``.

    Raises
    ------
    ValueError
        If a supplied ``pi0`` is outside ``(0, 1]``.

    References
    ----------
    Storey (2002), JRSS-B 64(3), 479-498; Storey & Tibshirani (2003), PNAS.
    """
    p = _as_pvalues(pvalues)
    if pi0 is None:
        pi0 = storey_pi0(pvalues, lambdas=lambdas)
    elif not 0.0 < pi0 <= 1.0:
        raise ValueError("pi0 must lie in (0, 1]")
    q = _step_up(p, pi0)
    return StoreyQValues(pi0=float(pi0), qvalues=tuple(float(x) for x in q))
