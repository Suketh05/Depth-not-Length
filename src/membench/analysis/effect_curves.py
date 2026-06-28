r"""Parametric fits of the compliance-vs-depth curve and the inter-arm crossover.

The headline figure of the benchmark is *compliance as a function of reasoning
depth*: how reliably each retrieval arm honours the governing decisions as the
causal chain it must recover grows longer. To summarise such a curve with a few
interpretable numbers (and to locate the depth at which one arm overtakes
another) this module fits two standard nonlinear models to the measured
``(depth, compliance)`` points and extracts where two fitted curves cross.

Models
------
**Four-parameter logistic (sigmoid).** Compliance typically holds near a high
plateau at shallow depth, then falls through a transition to a low plateau as the
chain lengthens. The four-parameter logistic captures exactly this S-shaped fall:

.. math::

    f(d) = L_{\min} + \frac{L_{\max} - L_{\min}}{1 + \exp\!\big(k\,(d - d_0)\big)},

with lower asymptote :math:`L_{\min}`, upper asymptote :math:`L_{\max}`, midpoint
(inflection depth) :math:`d_0`, and growth rate :math:`k`. For :math:`k > 0` the
curve decreases from :math:`L_{\max}` to :math:`L_{\min}`; at :math:`d = d_0` it
takes the value :math:`(L_{\min} + L_{\max})/2`. This is the standard logistic
of Verhulst, in the four-parameter form used throughout dose-response and growth
modelling (Ratkowsky, 1990).

**Exponential decay with offset.** A memoryless alternative in which each extra
hop multiplies the recoverable signal by a constant factor:

.. math::

    f(d) = A\,e^{-\lambda d} + C,

with amplitude :math:`A`, decay rate :math:`\lambda`, and floor :math:`C`. This is
the same geometric attenuation the theory assumes for raw similarity recovery
(:mod:`membench.theory.decay`), here fit directly in compliance space.

Both fits are obtained by nonlinear least squares
(:func:`scipy.optimize.curve_fit`, a Levenberg-Marquardt / trust-region solver),
and goodness of fit is reported as the coefficient of determination
:math:`R^2 = 1 - \mathrm{SS}_{\text{res}}/\mathrm{SS}_{\text{tot}}`.

Crossover
---------
Given two fitted curves (e.g. a structured-graph arm and a lexical baseline), the
*crossover depth* is the depth at which their compliance is equal — below it one
arm wins, above it the other. With both curves continuous and monotone-ish, the
difference changes sign at most once on the scanned interval, and the root is
found with Brent's method (:func:`scipy.optimize.brentq`; Brent, 1973).

References
----------
.. [1] Verhulst, P.-F. (1845). "Recherches mathématiques sur la loi
       d'accroissement de la population." *Nouveaux Mémoires de l'Académie Royale
       des Sciences et Belles-Lettres de Bruxelles* 18, 1-42. (The logistic law.)
.. [2] Ratkowsky, D. A. (1990). *Handbook of Nonlinear Regression Models.*
       Marcel Dekker. (Four-parameter logistic and exponential-decay forms.)
.. [3] Brent, R. P. (1973). *Algorithms for Minimization without Derivatives*,
       ch. 4. Prentice-Hall. (Root finding used for the crossover.)
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

__all__ = [
    "CrossoverPoint",
    "ExponentialDecayFit",
    "LogisticFit",
    "find_curve_crossover",
    "fit_exponential_decay",
    "fit_logistic",
]


def _logistic_curve(
    depth: NDArray[np.float64],
    lower: float,
    upper: float,
    midpoint: float,
    growth_rate: float,
) -> NDArray[np.float64]:
    r"""Vectorised four-parameter logistic :math:`L_{\min}+(L_{\max}-L_{\min})/(1+e^{k(d-d_0)})`."""
    value = lower + (upper - lower) / (1.0 + np.exp(growth_rate * (depth - midpoint)))
    return np.asarray(value, dtype=np.float64)


def _exponential_decay_curve(
    depth: NDArray[np.float64],
    amplitude: float,
    decay_rate: float,
    offset: float,
) -> NDArray[np.float64]:
    r"""Vectorised exponential decay :math:`A e^{-\lambda d} + C`."""
    value = amplitude * np.exp(-decay_rate * depth) + offset
    return np.asarray(value, dtype=np.float64)


def _r_squared(observed: NDArray[np.float64], predicted: NDArray[np.float64]) -> float:
    r"""Coefficient of determination :math:`1 - \text{SS}_{res}/\text{SS}_{tot}`."""
    residual_ss = float(np.sum((observed - predicted) ** 2))
    total_ss = float(np.sum((observed - float(np.mean(observed))) ** 2))
    if total_ss == 0.0:
        # Constant target: a perfect fit reproduces it exactly, otherwise R^2 is undefined (->0).
        return 1.0 if residual_ss == 0.0 else 0.0
    return 1.0 - residual_ss / total_ss


def _validate_pairs(
    depths: Sequence[float], compliance: Sequence[float], *, min_points: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Coerce and check aligned ``(depth, compliance)`` inputs for a fit."""
    x = np.asarray(depths, dtype=np.float64)
    y = np.asarray(compliance, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"depths and compliance must align: {x.shape} vs {y.shape}")
    if x.ndim != 1:
        raise ValueError(f"depths and compliance must be 1-D, got ndim={x.ndim}")
    if x.size < min_points:
        raise ValueError(f"need at least {min_points} points to fit, got {x.size}")
    if np.ptp(x) == 0:
        raise ValueError("depths must vary; cannot fit a curve to a single depth")
    return x, y


@dataclass(frozen=True, slots=True)
class LogisticFit:
    r"""A fitted four-parameter logistic compliance curve.

    Attributes
    ----------
    lower
        Lower asymptote :math:`L_{\min}` (compliance as depth :math:`\to \infty`
        when ``growth_rate`` > 0).
    upper
        Upper asymptote :math:`L_{\max}` (compliance at shallow depth).
    midpoint
        Inflection depth :math:`d_0`, where the curve sits halfway between the
        asymptotes.
    growth_rate
        Logistic rate :math:`k`. Positive values give a curve that falls with
        depth; the steepness grows with :math:`|k|`.
    r_squared
        Coefficient of determination on the fitted points (1.0 is a perfect fit).
    n_points
        Number of ``(depth, compliance)`` pairs used.
    """

    lower: float
    upper: float
    midpoint: float
    growth_rate: float
    r_squared: float
    n_points: int

    def predict(self, depth: float) -> float:
        """Return the fitted compliance at ``depth``."""
        return self.lower + (self.upper - self.lower) / (
            1.0 + math.exp(self.growth_rate * (depth - self.midpoint))
        )


@dataclass(frozen=True, slots=True)
class ExponentialDecayFit:
    r"""A fitted exponential-decay compliance curve :math:`A e^{-\lambda d} + C`.

    Attributes
    ----------
    amplitude
        Decaying component :math:`A` (compliance above the floor at depth 0).
    decay_rate
        Per-hop decay rate :math:`\lambda`. Positive values decay with depth.
    offset
        Asymptotic floor :math:`C` (compliance as depth :math:`\to \infty`).
    r_squared
        Coefficient of determination on the fitted points (1.0 is a perfect fit).
    n_points
        Number of ``(depth, compliance)`` pairs used.
    """

    amplitude: float
    decay_rate: float
    offset: float
    r_squared: float
    n_points: int

    def predict(self, depth: float) -> float:
        """Return the fitted compliance at ``depth``."""
        return self.amplitude * math.exp(-self.decay_rate * depth) + self.offset


@dataclass(frozen=True, slots=True)
class CrossoverPoint:
    r"""Where two compliance curves meet.

    Attributes
    ----------
    exists
        Whether the two curves cross within the scanned depth interval.
    depth
        Depth at which the curves are equal (``nan`` when ``exists`` is False).
    compliance
        Common compliance value at the crossover (``nan`` when ``exists`` is
        False).
    """

    exists: bool
    depth: float
    compliance: float


def fit_logistic(depths: Sequence[float], compliance: Sequence[float]) -> LogisticFit:
    r"""Fit a four-parameter logistic to measured ``(depth, compliance)`` pairs.

    Fits :math:`f(d) = L_{\min} + (L_{\max}-L_{\min})/(1+e^{k(d-d_0)})` by nonlinear
    least squares (:func:`scipy.optimize.curve_fit`). Initial guesses are taken from
    the data: asymptotes from the compliance range, midpoint from the mean depth,
    and a unit growth rate.

    Parameters
    ----------
    depths
        Reasoning depths (the x-axis). At least four points with some variation.
    compliance
        Compliance rate measured at each depth, typically in :math:`[0, 1]`.

    Returns
    -------
    LogisticFit
        The fitted parameters with the goodness-of-fit :math:`R^2`.

    Raises
    ------
    ValueError
        If the inputs are misaligned, have fewer than four points, or all share a
        single depth.
    RuntimeError
        Propagated from :func:`scipy.optimize.curve_fit` if the least-squares
        solver fails to converge.
    """
    x, y = _validate_pairs(depths, compliance, min_points=4)
    p0 = [float(y.min()), float(y.max()), float(x.mean()), 1.0]
    popt, _ = optimize.curve_fit(_logistic_curve, x, y, p0=p0, maxfev=10_000)
    lower, upper, midpoint, growth_rate = (float(value) for value in popt)
    predicted = _logistic_curve(x, lower, upper, midpoint, growth_rate)
    return LogisticFit(
        lower=lower,
        upper=upper,
        midpoint=midpoint,
        growth_rate=growth_rate,
        r_squared=_r_squared(y, predicted),
        n_points=int(x.size),
    )


def fit_exponential_decay(
    depths: Sequence[float], compliance: Sequence[float]
) -> ExponentialDecayFit:
    r"""Fit an exponential-decay curve to measured ``(depth, compliance)`` pairs.

    Fits :math:`f(d) = A e^{-\lambda d} + C` by nonlinear least squares
    (:func:`scipy.optimize.curve_fit`), with initial guesses drawn from the data:
    the floor from the minimum compliance, the amplitude from the compliance range,
    and a unit decay rate.

    Parameters
    ----------
    depths
        Reasoning depths (the x-axis). At least three points with some variation.
    compliance
        Compliance rate measured at each depth, typically in :math:`[0, 1]`.

    Returns
    -------
    ExponentialDecayFit
        The fitted parameters with the goodness-of-fit :math:`R^2`.

    Raises
    ------
    ValueError
        If the inputs are misaligned, have fewer than three points, or all share a
        single depth.
    RuntimeError
        Propagated from :func:`scipy.optimize.curve_fit` if the least-squares
        solver fails to converge.
    """
    x, y = _validate_pairs(depths, compliance, min_points=3)
    p0 = [float(np.ptp(y)) or 1.0, 1.0, float(y.min())]
    popt, _ = optimize.curve_fit(_exponential_decay_curve, x, y, p0=p0, maxfev=10_000)
    amplitude, decay_rate, offset = (float(value) for value in popt)
    predicted = _exponential_decay_curve(x, amplitude, decay_rate, offset)
    return ExponentialDecayFit(
        amplitude=amplitude,
        decay_rate=decay_rate,
        offset=offset,
        r_squared=_r_squared(y, predicted),
        n_points=int(x.size),
    )


def find_curve_crossover(
    curve_a: Callable[[float], float],
    curve_b: Callable[[float], float],
    *,
    depth_min: float,
    depth_max: float,
) -> CrossoverPoint:
    r"""Find the depth at which two compliance curves are equal.

    Locates the root of :math:`h(d) = \text{curve\_a}(d) - \text{curve\_b}(d)` on
    ``[depth_min, depth_max]``. If ``h`` does not change sign across the interval
    (one arm dominates throughout), no crossover exists; otherwise the root is
    bracketed and refined with Brent's method (:func:`scipy.optimize.brentq`).

    Parameters
    ----------
    curve_a, curve_b
        Compliance-vs-depth functions for the two arms (e.g. the ``predict``
        method of a :class:`LogisticFit` or :class:`ExponentialDecayFit`).
    depth_min, depth_max
        Inclusive depth interval to search; ``depth_min < depth_max``.

    Returns
    -------
    CrossoverPoint
        The crossover depth and shared compliance, or ``exists=False`` with ``nan``
        coordinates when the curves do not cross on the interval.

    Raises
    ------
    ValueError
        If ``depth_min`` is not strictly less than ``depth_max``.
    """
    if not depth_min < depth_max:
        raise ValueError(f"need depth_min < depth_max, got [{depth_min}, {depth_max}]")

    def _difference(depth: float) -> float:
        return curve_a(depth) - curve_b(depth)

    diff_lo = _difference(depth_min)
    diff_hi = _difference(depth_max)
    if diff_lo == 0.0:
        return CrossoverPoint(exists=True, depth=depth_min, compliance=curve_a(depth_min))
    if diff_hi == 0.0:
        return CrossoverPoint(exists=True, depth=depth_max, compliance=curve_a(depth_max))
    if diff_lo * diff_hi > 0.0:
        return CrossoverPoint(exists=False, depth=math.nan, compliance=math.nan)

    root = float(optimize.brentq(_difference, depth_min, depth_max))
    return CrossoverPoint(exists=True, depth=root, compliance=curve_a(root))
