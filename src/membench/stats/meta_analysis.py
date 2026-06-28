r"""DerSimonian-Laird random-effects meta-analysis.

When the same comparison (e.g. a structured-vs-similarity effect) is estimated on
several datasets or repeated runs, each estimate has its own sampling variance and
the *true* effect may itself vary across studies. A fixed-effect pool assumes one
shared effect; a random-effects pool admits between-study heterogeneity and widens
the interval accordingly. The DerSimonian-Laird estimator is the standard,
closed-form moment estimator for that between-study variance.

Given ``k`` studies with effect estimates :math:`y_i` and within-study variances
:math:`v_i`, let the fixed-effect (inverse-variance) weights be :math:`w_i = 1/v_i`.

.. math::

    \hat\mu_{\text{FE}} &= \frac{\sum_i w_i y_i}{\sum_i w_i} \\
    Q &= \sum_i w_i (y_i - \hat\mu_{\text{FE}})^2 \\
    C &= \sum_i w_i - \frac{\sum_i w_i^2}{\sum_i w_i} \\
    \hat\tau^2 &= \max\!\left(0,\; \frac{Q - (k-1)}{C}\right) \\
    w_i^{*} &= \frac{1}{v_i + \hat\tau^2}, \qquad
    \hat\mu_{\text{RE}} = \frac{\sum_i w_i^{*} y_i}{\sum_i w_i^{*}} \\
    \operatorname{Var}(\hat\mu_{\text{RE}}) &= \frac{1}{\sum_i w_i^{*}}

with Higgins' inconsistency :math:`I^2 = \max(0, (Q - (k-1))/Q)` (defined as ``0``
when ``Q == 0``). The pooled confidence interval is the normal interval
:math:`\hat\mu_{\text{RE}} \pm z_{1-\alpha/2}\sqrt{\operatorname{Var}}`.

References
----------
DerSimonian, R. and Laird, N. (1986). "Meta-analysis in clinical trials."
*Controlled Clinical Trials*, 7(3), 177-188. doi:10.1016/0197-2456(86)90046-2.
Higgins, J. P. T. and Thompson, S. G. (2002). "Quantifying heterogeneity in a
meta-analysis." *Statistics in Medicine*, 21(11), 1539-1558.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = ["MetaAnalysisResult", "dersimonian_laird"]


@dataclass(frozen=True, slots=True)
class MetaAnalysisResult:
    r"""Random-effects (DerSimonian-Laird) pooled effect with heterogeneity stats.

    Attributes
    ----------
    pooled_effect
        Random-effects pooled estimate :math:`\hat\mu_{\text{RE}}`.
    ci_low, ci_high
        Bounds of the normal confidence interval at level ``confidence``.
    standard_error
        Standard error of the pooled random-effects estimate.
    fixed_effect
        Inverse-variance fixed-effect estimate :math:`\hat\mu_{\text{FE}}`.
    q
        Cochran's :math:`Q` heterogeneity statistic.
    tau_squared
        DerSimonian-Laird between-study variance estimate (clamped at 0).
    i_squared
        Higgins' :math:`I^2` inconsistency in ``[0, 1]``.
    k
        Number of studies pooled.
    confidence
        Confidence level used for the interval.
    """

    pooled_effect: float
    ci_low: float
    ci_high: float
    standard_error: float
    fixed_effect: float
    q: float
    tau_squared: float
    i_squared: float
    k: int
    confidence: float


def dersimonian_laird(
    effects: Sequence[float],
    variances: Sequence[float],
    *,
    confidence: float = 0.95,
) -> MetaAnalysisResult:
    r"""Pool study effects with the DerSimonian-Laird random-effects estimator.

    Parameters
    ----------
    effects
        Per-study effect estimates :math:`y_i` (any additive scale, e.g. a mean
        difference or a log odds ratio).
    variances
        Per-study within-study variances :math:`v_i`, strictly positive and the
        same length as ``effects``.
    confidence
        Confidence level for the pooled interval, in ``(0, 1)``.

    Returns
    -------
    MetaAnalysisResult
        Pooled random-effects estimate, its confidence interval, and the
        heterogeneity statistics (:math:`Q`, :math:`\tau^2`, :math:`I^2`).

    Raises
    ------
    ValueError
        If fewer than two studies are given, the lengths differ, any variance is
        non-positive, or ``confidence`` is not in ``(0, 1)``.

    Notes
    -----
    Implements the moment estimator of DerSimonian and Laird (1986); see the
    module docstring for the formulae and references.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    y = np.asarray(effects, dtype=np.float64)
    v = np.asarray(variances, dtype=np.float64)
    if y.shape != v.shape:
        raise ValueError("effects and variances must have the same length")
    k = int(y.size)
    if k < 2:
        raise ValueError("need at least two studies to pool")
    if np.any(v <= 0.0):
        raise ValueError("all variances must be positive")

    # Fixed-effect inverse-variance weights and pooled estimate.
    w = 1.0 / v
    sum_w = float(np.sum(w))
    fixed_effect = float(np.sum(w * y) / sum_w)

    # Cochran's Q about the fixed-effect estimate.
    q = float(np.sum(w * (y - fixed_effect) ** 2))

    # DerSimonian-Laird between-study variance (moment estimator), clamped at 0.
    c = sum_w - float(np.sum(w**2)) / sum_w
    tau_squared = max(0.0, (q - (k - 1)) / c)

    # Higgins' I^2 (0 when there is no heterogeneity signal, including Q == 0).
    i_squared = max(0.0, (q - (k - 1)) / q) if q > 0.0 else 0.0

    # Random-effects weights, pooled estimate, and its variance.
    w_star = 1.0 / (v + tau_squared)
    sum_w_star = float(np.sum(w_star))
    pooled_effect = float(np.sum(w_star * y) / sum_w_star)
    variance = 1.0 / sum_w_star
    standard_error = float(np.sqrt(variance))

    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    margin = z * standard_error
    return MetaAnalysisResult(
        pooled_effect=pooled_effect,
        ci_low=pooled_effect - margin,
        ci_high=pooled_effect + margin,
        standard_error=standard_error,
        fixed_effect=fixed_effect,
        q=q,
        tau_squared=tau_squared,
        i_squared=i_squared,
        k=k,
        confidence=confidence,
    )
