r"""Bootstrap confidence intervals appropriate for this benchmark's design.

Benchmark observations are not independent: every arm is evaluated on the *same*
tasks (a repeated-measures / clustered design). Two tools here respect that:

* :func:`bca_ci` -- the bias-corrected and accelerated bootstrap, which corrects the
  percentile interval for bias and skew (important for bounded rates near 0/1 and for
  ratios like cost-to-correct), giving second-order-accurate intervals.
* :func:`cluster_bootstrap_ci` -- resamples whole *clusters* (tasks) with replacement
  rather than individual rows, which is the statistically correct bootstrap when rows
  within a task are dependent.

All intervals are seeded for reproducibility.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

__all__ = ["ConfidenceInterval", "bca_ci", "cluster_bootstrap_ci", "percentile_ci"]

Statistic = Callable[[NDArray[np.float64]], float]


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A point estimate with a lower/upper confidence bound."""

    point: float
    low: float
    high: float
    confidence: float


def _mean(x: NDArray[np.float64]) -> float:
    return float(np.mean(x))


def _resample(
    values: NDArray[np.float64], statistic: Statistic, n: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    idx = rng.integers(0, values.size, size=(n, values.size))
    return np.array([statistic(values[row]) for row in idx], dtype=np.float64)


def percentile_ci(
    values: Sequence[float],
    statistic: Statistic = _mean,
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> ConfidenceInterval:
    """Return a basic percentile bootstrap interval (seeded)."""
    x = np.asarray(values, dtype=np.float64)
    if x.size < 2:
        raise ValueError("need at least two observations")
    rng = np.random.default_rng(seed)
    boot = _resample(x, statistic, n_resamples, rng)
    alpha = 1.0 - confidence
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return ConfidenceInterval(statistic(x), float(low), float(high), confidence)


def bca_ci(
    values: Sequence[float],
    statistic: Statistic = _mean,
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> ConfidenceInterval:
    """Bias-corrected and accelerated (BCa) bootstrap interval.

    Corrects the percentile interval for median bias (``z0``) and skew (the
    acceleration ``a``, from the jackknife), giving second-order-accurate bounds.
    Falls back to percentile bounds if the acceleration is degenerate.
    """
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    if n < 2:
        raise ValueError("need at least two observations")
    rng = np.random.default_rng(seed)
    theta_hat = statistic(x)
    boot = _resample(x, statistic, n_resamples, rng)

    # bias-correction z0 from the fraction of bootstrap stats below the estimate
    prop = float(np.mean(boot < theta_hat))
    prop = min(max(prop, 1.0 / n_resamples), 1.0 - 1.0 / n_resamples)
    z0 = stats.norm.ppf(prop)

    # acceleration from the jackknife
    jack = np.array([statistic(np.delete(x, i)) for i in range(n)], dtype=np.float64)
    jack_mean = jack.mean()
    diff = jack_mean - jack
    denom = 6.0 * (np.sum(diff**2) ** 1.5)
    a = float(np.sum(diff**3) / denom) if denom != 0 else 0.0

    alpha = 1.0 - confidence
    z_lo, z_hi = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)

    def adjust(z: float) -> float:
        return float(stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))

    a1, a2 = adjust(z_lo), adjust(z_hi)
    low, high = np.quantile(boot, [a1, a2])
    return ConfidenceInterval(theta_hat, float(low), float(high), confidence)


def cluster_bootstrap_ci(
    values: Sequence[float],
    clusters: Sequence[str],
    statistic: Statistic = _mean,
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> ConfidenceInterval:
    """Cluster bootstrap: resample whole clusters (tasks) with replacement.

    The correct bootstrap when observations within a cluster are dependent (e.g.
    several attempts or arms per task). Resamples cluster labels, pools the values of
    the chosen clusters, and recomputes the statistic.
    """
    x = np.asarray(values, dtype=np.float64)
    labels = np.asarray(clusters)
    if x.shape != labels.shape:
        raise ValueError("values and clusters must align")
    unique = np.unique(labels)
    if unique.size < 2:
        raise ValueError("need at least two clusters")
    by_cluster = {c: x[labels == c] for c in unique}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        chosen = rng.choice(unique, size=unique.size, replace=True)
        pooled = np.concatenate([by_cluster[c] for c in chosen])
        boot[b] = statistic(pooled)
    alpha = 1.0 - confidence
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return ConfidenceInterval(statistic(x), float(low), float(high), confidence)
