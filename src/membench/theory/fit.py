r"""Assemble the theory from measurements and compare prediction to observation.

The decay law (:mod:`membench.theory.decay`) is fit from embedding similarities;
this module supplies the remaining free parameter — the structured per-link
reliability :math:`q` — from observed full-chain recovery rates of the structured
arm, then assembles a :class:`~membench.theory.recovery.RecoveryModel`, predicts
the crossover depth, and compares that prediction against the empirically measured
crossover.

The structured recovery law :math:`P_\text{struct}(d) = q^d` has no intercept
(:math:`P = 1` at :math:`d = 0`), so :math:`q` is estimated by ordinary least
squares *through the origin* on :math:`(d, \log P)`:

.. math::

    \widehat{\log q} = \frac{\sum_d d \,\log P_d}{\sum_d d^2},
    \qquad \widehat{q} = e^{\widehat{\log q}}.

The headline empirical result of the paper is then a single sentence: *the theory,
parameterised only by measured* :math:`\rho, s_0, q`, *predicts a crossover at*
:math:`d^\star`, *and we observe the accuracy/cost curves cross there.*
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from membench.theory.crossover import CrossoverResult, find_crossover_depth
from membench.theory.decay import DecayFit
from membench.theory.recovery import RecoveryModel

__all__ = [
    "CrossoverComparison",
    "LinkReliabilityFit",
    "build_recovery_model",
    "compare_crossover",
    "estimate_link_reliability",
    "predict_crossover",
]


@dataclass(frozen=True, slots=True)
class LinkReliabilityFit:
    r"""Estimated structured per-link reliability :math:`q` with goodness-of-fit.

    Attributes
    ----------
    q
        Estimated per-link recovery probability, clipped to ``(0, 1]``.
    r_squared
        Coefficient of determination of the through-origin log-linear fit.
    n_points
        Number of (depth, recovery-rate) pairs used.
    """

    q: float
    r_squared: float
    n_points: int


def estimate_link_reliability(
    depths: Sequence[int],
    recovery_rates: Sequence[float],
) -> LinkReliabilityFit:
    r"""Estimate :math:`q` from observed structured full-chain recovery rates.

    Parameters
    ----------
    depths
        Chain depths (>= 1) at which recovery was measured.
    recovery_rates
        Observed fraction of chains fully recovered at each depth, in ``(0, 1]``.

    Returns
    -------
    LinkReliabilityFit
        The estimate with a through-origin :math:`R^2`.

    Raises
    ------
    ValueError
        If inputs are mismatched, empty, contain non-positive depths, or contain
        recovery rates outside ``(0, 1]`` (zero recovery has no finite log).
    """
    d = np.asarray(depths, dtype=np.float64)
    p = np.asarray(recovery_rates, dtype=np.float64)
    if d.shape != p.shape:
        raise ValueError(f"depths and recovery_rates must align: {d.shape} vs {p.shape}")
    if d.size == 0:
        raise ValueError("need at least one (depth, recovery_rate) pair")
    if np.any(d < 1):
        raise ValueError("depths must be >= 1")
    if np.any(p <= 0) or np.any(p > 1):
        raise ValueError("recovery_rates must be in (0, 1]")

    log_p = np.log(p)
    # OLS through the origin: slope = sum(d * log_p) / sum(d^2).
    slope = float(np.sum(d * log_p) / np.sum(d**2))
    q = float(min(1.0, np.exp(slope)))

    # R^2 relative to the through-origin model's predictions.
    pred = slope * d
    ss_res = float(np.sum((log_p - pred) ** 2))
    ss_tot = float(np.sum(log_p**2))  # about zero, the no-decay null in log-space
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return LinkReliabilityFit(q=q, r_squared=r_squared, n_points=int(d.size))


def build_recovery_model(decay_fit: DecayFit, link_fit: LinkReliabilityFit) -> RecoveryModel:
    """Assemble a recovery model from a fitted decay law and a fitted ``q``."""
    return RecoveryModel(decay=decay_fit.model, q=link_fit.q)


def predict_crossover(
    decay_fit: DecayFit,
    link_fit: LinkReliabilityFit,
    tau: float,
    *,
    max_depth: int = 64,
) -> CrossoverResult:
    """Predict the crossover depth from the fitted decay law and link reliability."""
    model = build_recovery_model(decay_fit, link_fit)
    return find_crossover_depth(model, tau, max_depth=max_depth)


@dataclass(frozen=True, slots=True)
class CrossoverComparison:
    """Predicted vs. empirically measured crossover depth.

    Attributes
    ----------
    predicted
        Crossover depth predicted by the fitted theory.
    measured
        Crossover depth observed in the benchmark (e.g. from the segmented-logistic
        changepoint estimator).
    absolute_error
        ``|predicted - measured|`` in hops.
    agrees
        Whether the prediction lands within ``tolerance`` hops of the measurement.
    """

    predicted: float
    measured: float
    absolute_error: float
    agrees: bool


def compare_crossover(
    predicted: float,
    measured: float,
    *,
    tolerance: float = 1.0,
) -> CrossoverComparison:
    """Compare a predicted crossover depth against the measured one.

    Parameters
    ----------
    predicted
        Theory-predicted crossover depth.
    measured
        Empirically measured crossover depth.
    tolerance
        Maximum absolute difference (in hops) for the two to count as agreeing.
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    err = abs(predicted - measured)
    return CrossoverComparison(
        predicted=predicted,
        measured=measured,
        absolute_error=err,
        agrees=err <= tolerance,
    )
