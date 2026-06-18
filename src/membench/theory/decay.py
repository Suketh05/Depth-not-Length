r"""The similarity-decay model: how retrievability falls with causal distance.

The benchmark's thesis starts from a single empirical claim (Definition 1 of the
theory note): the resemblance of a needed node to the current query decays
geometrically with the number of causal hops between them,

.. math::

    s_i = s_0 \, \rho^{\,i},

where :math:`s_0 \in (0, 1]` is the retrievability of the node closest to the
query and :math:`\rho \in (0, 1)` is the per-hop decay rate. Each step away from
where the agent is standing makes the next needed fact look a little less like the
thing it is searching for.

This module provides the model (:class:`SimilarityDecayModel`) and a log-linear
estimator (:func:`fit_similarity_decay`) that recovers :math:`(s_0, \rho)` from
measured (distance, similarity) pairs — for example, cosine similarities measured
along known dependency chains in an embedding space. Fitting in log-space turns
the geometric law into a straight line :math:`\log s_i = \log s_0 + i \log \rho`,
so ordinary least squares gives a closed-form estimate with an :math:`R^2` that
quantifies how well the geometric assumption actually holds on the data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

__all__ = ["DecayFit", "SimilarityDecayModel", "fit_similarity_decay"]


@dataclass(frozen=True, slots=True)
class SimilarityDecayModel:
    r"""A geometric similarity-decay law :math:`s_i = s_0 \rho^i`.

    Parameters
    ----------
    s0
        Retrievability at distance 0 (the node closest to the query). Positive;
        typically in :math:`(0, 1]` for a cosine-similarity scale.
    rho
        Per-hop decay rate. Positive. The geometric-decay regime the theory
        assumes is :math:`\rho \in (0, 1)`; values :math:`\ge 1` are representable
        (an empirical fit might find no decay) and flagged by :attr:`is_decaying`.
    """

    s0: float
    rho: float

    def __post_init__(self) -> None:
        if not self.s0 > 0:
            raise ValueError(f"s0 must be positive, got {self.s0}")
        if not self.rho > 0:
            raise ValueError(f"rho must be positive, got {self.rho}")

    @property
    def is_decaying(self) -> bool:
        r"""True when :math:`\rho < 1`, i.e. retrievability genuinely falls with depth."""
        return self.rho < 1.0

    @property
    def half_life(self) -> float:
        r"""Causal distance at which retrievability halves, :math:`\ln(1/2)/\ln\rho`.

        Infinite when there is no decay (:math:`\rho \ge 1`).
        """
        if self.rho >= 1.0:
            return math.inf
        return math.log(0.5) / math.log(self.rho)

    def retrievability(self, distance: int | float) -> float:
        r"""Return :math:`s_0 \rho^{\text{distance}}`, the retrievability at ``distance`` hops."""
        return self.s0 * self.rho**distance

    def retrievabilities(self, max_distance: int) -> NDArray[np.float64]:
        r"""Return retrievabilities for distances :math:`i = 1, \dots, \text{max\_distance}`.

        Distance 0 (the query node itself) is excluded: a chain of depth ``d`` has
        ``d`` upstream hops to recover, indexed 1..d, which is what the recovery
        bounds in :mod:`membench.theory.recovery` consume.
        """
        if max_distance < 1:
            raise ValueError(f"max_distance must be >= 1, got {max_distance}")
        distances = np.arange(1, max_distance + 1, dtype=np.float64)
        return np.asarray(self.s0 * self.rho**distances, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class DecayFit:
    r"""Result of fitting :class:`SimilarityDecayModel` to measured data.

    Attributes
    ----------
    model
        The fitted decay law.
    r_squared
        Coefficient of determination of the log-linear fit; how well the geometric
        assumption holds on the data (1.0 is a perfect geometric decay).
    n_points
        Number of (distance, similarity) pairs used.
    stderr_log_rho
        Standard error of the slope estimate :math:`\log \rho`, propagated for
        confidence intervals on the decay rate.
    p_value
        Two-sided p-value for a non-zero slope (i.e. for decay being present).
    """

    model: SimilarityDecayModel
    r_squared: float
    n_points: int
    stderr_log_rho: float
    p_value: float

    def predict(self, distances: Sequence[float]) -> NDArray[np.float64]:
        """Predicted retrievabilities at the given distances under the fitted model."""
        arr = np.asarray(distances, dtype=np.float64)
        return np.asarray(self.model.s0 * self.model.rho**arr, dtype=np.float64)


def fit_similarity_decay(
    distances: Sequence[float],
    similarities: Sequence[float],
) -> DecayFit:
    r"""Estimate :math:`(s_0, \rho)` from measured (distance, similarity) pairs.

    Fits the geometric law in log-space by ordinary least squares:
    :math:`\log s_i = \log s_0 + i \log \rho`. The slope yields
    :math:`\rho = e^{\text{slope}}` and the intercept yields
    :math:`s_0 = e^{\text{intercept}}`.

    Parameters
    ----------
    distances
        Causal distances :math:`i` (non-negative). At least two distinct values.
    similarities
        Measured similarities at those distances, strictly positive (cosine
        similarities should be shifted/clipped into :math:`(0, 1]` upstream so the
        logarithm is defined).

    Returns
    -------
    DecayFit
        The fitted model with goodness-of-fit and slope-uncertainty statistics.

    Raises
    ------
    ValueError
        If the inputs are mismatched, too few, contain non-positive similarities,
        or have no variation in distance (the slope would be undefined).
    """
    x = np.asarray(distances, dtype=np.float64)
    y = np.asarray(similarities, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"distances and similarities must align: {x.shape} vs {y.shape}")
    if x.size < 2:
        raise ValueError("need at least two points to fit a decay")
    if np.any(y <= 0):
        raise ValueError("similarities must be strictly positive to fit in log-space")
    if np.ptp(x) == 0:
        raise ValueError("distances must vary; cannot fit a slope to a single distance")

    log_y = np.log(y)
    result = stats.linregress(x, log_y)
    rho = float(np.exp(result.slope))
    s0 = float(np.exp(result.intercept))
    return DecayFit(
        model=SimilarityDecayModel(s0=s0, rho=rho),
        r_squared=float(result.rvalue**2),
        n_points=int(x.size),
        stderr_log_rho=float(result.stderr),
        p_value=float(result.pvalue),
    )
