r"""Murphy's three-component partition of the Brier score.

The Brier score :math:`BS = \tfrac1N\sum_i (p_i - o_i)^2` of a set of probabilistic
forecasts admits an additive decomposition into *reliability* (calibration),
*resolution* (discrimination), and *uncertainty* (irreducible base-rate variance).
Grouping the :math:`N` forecasts into bins :math:`k` with :math:`n_k` members, bin
mean forecast :math:`\bar f_k`, observed frequency :math:`\bar o_k`, and overall base
rate :math:`\bar o = \tfrac1N\sum_i o_i`,

.. math::

    \underbrace{\tfrac1N\sum_k n_k(\bar f_k-\bar o_k)^2}_{\text{reliability}}
    -\underbrace{\tfrac1N\sum_k n_k(\bar o_k-\bar o)^2}_{\text{resolution}}
    +\underbrace{\bar o(1-\bar o)}_{\text{uncertainty}}
    \;=\; \tfrac1N\sum_k\sum_{i\in k}(\bar f_k-o_i)^2 ,

i.e. ``Brier = Reliability - Resolution + Uncertainty``. The identity is exact for the
*binned* Brier score (each forecast replaced by its bin mean :math:`\bar f_k`); it
coincides with the raw Brier score when forecasts are constant within every bin
(e.g. genuinely discrete forecast values). Lower reliability is better (0 = perfectly
calibrated); higher resolution is better (the forecaster separates high- from
low-frequency events); uncertainty depends only on the outcomes, not the forecasts.

Reference
---------
Murphy, A. H. (1973). "A New Vector Partition of the Probability Score."
*Journal of Applied Meteorology*, 12(4), 595-600.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "BrierDecomposition",
    "brier_decomposition",
]


def _validate(
    probs: Sequence[float], outcomes: Sequence[bool]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Coerce and check that ``probs`` and ``outcomes`` form a valid forecast set."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError("probs and outcomes must align")
    if p.ndim != 1:
        raise ValueError("probs and outcomes must be one-dimensional")
    if p.size == 0:
        raise ValueError("need at least one observation")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must be in [0, 1]")
    if np.any((y != 0.0) & (y != 1.0)):
        raise ValueError("outcomes must be binary (0/1 or bool)")
    return p, y


@dataclass(frozen=True, slots=True)
class BrierDecomposition:
    """Murphy (1973) partition of a binned Brier score.

    Attributes
    ----------
    brier : float
        The binned Brier score, ``reliability - resolution + uncertainty``. Equal to
        the raw mean Brier score when forecasts are constant within each bin.
    reliability : float
        Count-weighted mean squared gap between each bin's mean forecast and its
        observed frequency. Non-negative; 0 for perfect calibration.
    resolution : float
        Count-weighted mean squared gap between each bin's observed frequency and the
        overall base rate. Non-negative; larger is better.
    uncertainty : float
        Base-rate variance ``base_rate * (1 - base_rate)``; the irreducible term.
    base_rate : float
        Overall event frequency ``mean(outcomes)``.
    n_bins : int
        Number of equal-width probability bins requested.
    n_observations : int
        Number of forecast/outcome pairs scored.
    """

    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    base_rate: float
    n_bins: int
    n_observations: int


def brier_decomposition(
    probs: Sequence[float], outcomes: Sequence[bool], n_bins: int = 10
) -> BrierDecomposition:
    r"""Decompose a Brier score into reliability, resolution, and uncertainty.

    Forecasts are grouped into ``n_bins`` equal-width bins over :math:`[0, 1]`. Each
    bin contributes via its mean forecast :math:`\bar f_k` (not the bin midpoint), so
    the returned components satisfy ``brier == reliability - resolution + uncertainty``
    exactly for the binned forecasts (Murphy, 1973).

    Parameters
    ----------
    probs : Sequence[float]
        Forecast probabilities in :math:`[0, 1]`, one per observation.
    outcomes : Sequence[bool]
        Realised binary outcomes aligned with ``probs``.
    n_bins : int, optional
        Number of equal-width confidence bins, by default 10. Must be >= 1.

    Returns
    -------
    BrierDecomposition
        The Brier score and its three additive components.

    Raises
    ------
    ValueError
        If ``n_bins < 1``, the inputs are misaligned/empty, probabilities fall
        outside :math:`[0, 1]`, or outcomes are not binary.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    p, y = _validate(probs, outcomes)
    n_obs = p.size
    base_rate = float(y.mean())

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    reliability = 0.0
    resolution = 0.0
    binned = np.empty_like(p)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        mean_forecast = float(p[mask].mean())
        observed = float(y[mask].mean())
        binned[mask] = mean_forecast
        reliability += count * (mean_forecast - observed) ** 2
        resolution += count * (observed - base_rate) ** 2

    reliability /= n_obs
    resolution /= n_obs
    uncertainty = base_rate * (1.0 - base_rate)
    brier = float(np.mean((binned - y) ** 2))

    return BrierDecomposition(
        brier=brier,
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        base_rate=base_rate,
        n_bins=n_bins,
        n_observations=n_obs,
    )
