r"""Local sensitivity and elasticity of the recovery laws and crossover depth.

The recovery bounds in :mod:`membench.theory.recovery` and the crossover analysis
in :mod:`membench.theory.crossover` are smooth, closed-form functions of three
parameters: the head retrievability :math:`s_0`, the per-hop similarity-decay rate
:math:`\rho`, and the structured per-link reliability :math:`q`. To know *which*
of these the conclusions actually hinge on, we differentiate the recovery
quantities with respect to each parameter (a *local sensitivity analysis*) and
report the dimensionless *elasticities* alongside the raw gradients.

**Gradients (calculus on the closed forms).** With the depth-``d`` triangular
exponent :math:`T = d(d+1)/2`,

.. math::

    P_\text{sim}(d) = s_0^{\,d}\,\rho^{\,T}, \qquad
    \frac{\partial P_\text{sim}}{\partial s_0} = d\, s_0^{\,d-1} \rho^{\,T},\qquad
    \frac{\partial P_\text{sim}}{\partial \rho} = T\, s_0^{\,d}\, \rho^{\,T-1},

    P_\text{struct}(d) = q^{\,d}, \qquad
    \frac{\partial P_\text{struct}}{\partial q} = d\, q^{\,d-1},

and the gap :math:`g(d) = P_\text{struct}(d) - P_\text{sim}(d)` differentiates
term by term. Because :math:`P_\text{sim}` does not depend on :math:`q` (and
:math:`P_\text{struct}` depends on neither :math:`s_0` nor :math:`\rho`) several
partials are identically zero, which the gap's gradient inherits.

**Elasticities (logarithmic derivatives).** The elasticity of a positive
quantity :math:`f` with respect to a parameter :math:`x` is
:math:`\varepsilon_x = \partial \ln f / \partial \ln x = (x/f)\,\partial f /
\partial x`, the percentage change in :math:`f` per one-percent change in
:math:`x` (Chiang & Wainwright, *Fundamental Methods of Mathematical Economics*,
4th ed., 2005, ch. 8). For the power-law recovery bounds the elasticities collapse
to the exponents themselves -- a textbook property of monomials:

.. math::

    \varepsilon_{s_0}[P_\text{sim}] = d, \qquad
    \varepsilon_{\rho}[P_\text{sim}] = T = \tfrac{d(d+1)}{2}, \qquad
    \varepsilon_{q}[P_\text{struct}] = d .

These exact, parameter-free elasticities are what the unit tests pin as golden
values.

**Crossover sensitivity (finite differences).** The crossover depth
:math:`d^\star(\tau)` is found numerically (Brent's method on the rising edge of
``g``), so it has no convenient closed-form derivative. We instead estimate
:math:`\partial d^\star / \partial x` by a second-order *central finite
difference*, :math:`f'(x) \approx (f(x+h) - f(x-h)) / 2h` with error
:math:`O(h^2)` (Nocedal & Wright, *Numerical Optimization*, 2nd ed., 2006,
§8.1). The same central-difference primitive is exposed publicly so callers can
cross-check the analytic recovery gradients above against a numerical estimate.

References
----------
.. [1] A. C. Chiang and K. Wainwright. *Fundamental Methods of Mathematical
       Economics*, 4th ed. McGraw-Hill, 2005. (Elasticity as a logarithmic
       derivative.)
.. [2] J. Nocedal and S. J. Wright. *Numerical Optimization*, 2nd ed. Springer,
       2006, §8.1. (Central finite differences and their :math:`O(h^2)` error.)
.. [3] A. Saltelli et al. *Global Sensitivity Analysis: The Primer*. Wiley, 2008.
       (Local vs. global sensitivity framing.)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from membench.theory.crossover import find_crossover_depth, g
from membench.theory.decay import SimilarityDecayModel
from membench.theory.recovery import (
    RecoveryModel,
    similarity_recovery,
    structured_recovery,
)

__all__ = [
    "CrossoverSensitivity",
    "RecoverySensitivity",
    "central_difference",
    "crossover_sensitivity",
    "gap_sensitivity",
    "similarity_sensitivity",
    "structured_sensitivity",
]


def _check_depth(d: int) -> None:
    if d < 1:
        raise ValueError(f"depth d must be >= 1, got {d}")


def _elasticity(x: float, value: float, grad: float) -> float:
    r"""Return the elasticity :math:`(x/f)\,\partial f/\partial x`.

    A zero gradient yields a zero elasticity (the parameter is inert). A non-zero
    gradient at a zero value -- the gap exactly at its crossover -- yields a signed
    infinity, the honest statement that the elasticity diverges there.
    """
    if grad == 0.0:
        return 0.0
    if value == 0.0:
        return math.copysign(math.inf, grad * x)
    return x * grad / value


@dataclass(frozen=True, slots=True)
class RecoverySensitivity:
    r"""Analytic gradient and elasticities of a recovery quantity at a point.

    All partial derivatives are evaluated analytically from the closed forms; the
    elasticities are the corresponding logarithmic derivatives
    :math:`(x/f)\,\partial f/\partial x`.

    Attributes
    ----------
    quantity
        Which recovery quantity was differentiated (``"similarity_recovery"``,
        ``"structured_recovery"`` or ``"gap"``).
    d
        Chain depth at which the sensitivity was evaluated.
    value
        The quantity's value :math:`f(d)` at the point.
    grad_s0, grad_rho, grad_q
        Partial derivatives :math:`\partial f/\partial s_0`,
        :math:`\partial f/\partial \rho`, :math:`\partial f/\partial q`. A partial
        is exactly ``0.0`` when ``f`` does not depend on that parameter.
    grad_d
        Partial derivative with respect to the (continuous) depth ``d``; negative
        for decaying recovery, since chains get harder to recover with depth.
    elasticity_s0, elasticity_rho, elasticity_q
        Elasticities with respect to :math:`s_0`, :math:`\rho`, :math:`q`.
    """

    quantity: str
    d: int
    value: float
    grad_s0: float
    grad_rho: float
    grad_q: float
    grad_d: float
    elasticity_s0: float
    elasticity_rho: float
    elasticity_q: float


@dataclass(frozen=True, slots=True)
class CrossoverSensitivity:
    r"""Finite-difference sensitivity of the crossover depth :math:`d^\star`.

    Attributes
    ----------
    tau
        Overhead ratio :math:`\tau = W/V` the crossover was solved for.
    d_star
        The continuous crossover depth :math:`d^\star` at the base point.
    grad_s0, grad_rho, grad_q
        Central-difference estimates of :math:`\partial d^\star/\partial s_0`,
        :math:`\partial d^\star/\partial \rho`, :math:`\partial d^\star/\partial q`.
        Raising :math:`s_0` or :math:`\rho` strengthens similarity recovery and so
        pushes the crossover deeper (positive); raising :math:`q` strengthens
        structured recovery and pulls it shallower (negative).
    step
        The finite-difference step :math:`h` used.
    """

    tau: float
    d_star: float
    grad_s0: float
    grad_rho: float
    grad_q: float
    step: float


def central_difference(f: Callable[[float], float], x: float, *, h: float = 1e-6) -> float:
    r"""Second-order central finite-difference estimate of :math:`f'(x)`.

    Uses :math:`f'(x) \approx (f(x+h) - f(x-h)) / 2h`, whose truncation error is
    :math:`O(h^2)` (Nocedal & Wright, 2006, §8.1).

    Parameters
    ----------
    f
        Scalar function of a single real argument.
    x
        Point at which to estimate the derivative.
    h
        Step size; must be positive.

    Returns
    -------
    float
        The estimated derivative :math:`f'(x)`.
    """
    if h <= 0:
        raise ValueError(f"step h must be positive, got {h}")
    return (f(x + h) - f(x - h)) / (2.0 * h)


def similarity_sensitivity(d: int, model: SimilarityDecayModel) -> RecoverySensitivity:
    r"""Analytic sensitivity of :func:`similarity_recovery` at depth ``d``.

    Differentiates :math:`P_\text{sim}(d) = s_0^{d}\rho^{T}` with
    :math:`T = d(d+1)/2`. The :math:`q` partial is identically zero (similarity
    recovery does not use structure) and the elasticities are the exact exponents
    :math:`\varepsilon_{s_0} = d`, :math:`\varepsilon_{\rho} = T`.

    Parameters
    ----------
    d
        Chain depth (>= 1).
    model
        The similarity-decay law supplying :math:`s_0` and :math:`\rho`.

    Returns
    -------
    RecoverySensitivity
        Gradients and elasticities of :math:`P_\text{sim}` at ``d``.
    """
    _check_depth(d)
    s0 = model.s0
    rho = model.rho
    t = d * (d + 1) / 2.0
    value = similarity_recovery(d, model)
    grad_s0 = d * s0 ** (d - 1) * rho**t
    grad_rho = t * s0**d * rho ** (t - 1)
    grad_d = value * (math.log(s0) + (d + 0.5) * math.log(rho))
    return RecoverySensitivity(
        quantity="similarity_recovery",
        d=d,
        value=value,
        grad_s0=grad_s0,
        grad_rho=grad_rho,
        grad_q=0.0,
        grad_d=grad_d,
        elasticity_s0=_elasticity(s0, value, grad_s0),
        elasticity_rho=_elasticity(rho, value, grad_rho),
        elasticity_q=0.0,
    )


def structured_sensitivity(d: int, q: float) -> RecoverySensitivity:
    r"""Analytic sensitivity of :func:`structured_recovery` at depth ``d``.

    Differentiates :math:`P_\text{struct}(d) = q^{d}`. Only the :math:`q` partial
    is non-zero, :math:`\partial P_\text{struct}/\partial q = d\,q^{d-1} > 0`
    (recovery increases in ``q``), and the elasticity is the exact exponent
    :math:`\varepsilon_q = d`.

    Parameters
    ----------
    d
        Chain depth (>= 1).
    q
        Per-link recovery probability, in :math:`(0, 1]`.

    Returns
    -------
    RecoverySensitivity
        Gradients and elasticities of :math:`P_\text{struct}` at ``d``.
    """
    _check_depth(d)
    value = structured_recovery(d, q)  # validates q in (0, 1]
    grad_q = d * q ** (d - 1)
    grad_d = value * math.log(q)  # q^d ln q; 0 at q = 1
    return RecoverySensitivity(
        quantity="structured_recovery",
        d=d,
        value=value,
        grad_s0=0.0,
        grad_rho=0.0,
        grad_q=grad_q,
        grad_d=grad_d,
        elasticity_s0=0.0,
        elasticity_rho=0.0,
        elasticity_q=_elasticity(q, value, grad_q),
    )


def gap_sensitivity(d: int, model: RecoveryModel) -> RecoverySensitivity:
    r"""Analytic sensitivity of the gap :math:`g(d) = P_\text{struct} - P_\text{sim}`.

    The gap differentiates term by term, so its partials are the structured
    partial minus the similarity partial along each axis. Because the two terms
    decouple across the parameters, the :math:`s_0` and :math:`\rho` partials come
    entirely (with a sign flip) from :math:`P_\text{sim}` and the :math:`q` partial
    entirely from :math:`P_\text{struct}`.

    Parameters
    ----------
    d
        Chain depth (>= 1).
    model
        The recovery model bundling the decay law and structured reliability
        :math:`q`.

    Returns
    -------
    RecoverySensitivity
        Gradients and elasticities of the gap ``g`` at ``d``.
    """
    _check_depth(d)
    sim = similarity_sensitivity(d, model.decay)
    struct = structured_sensitivity(d, model.q)
    value = g(float(d), model)
    grad_s0 = -sim.grad_s0
    grad_rho = -sim.grad_rho
    grad_q = struct.grad_q
    grad_d = struct.grad_d - sim.grad_d
    return RecoverySensitivity(
        quantity="gap",
        d=d,
        value=value,
        grad_s0=grad_s0,
        grad_rho=grad_rho,
        grad_q=grad_q,
        grad_d=grad_d,
        elasticity_s0=_elasticity(model.decay.s0, value, grad_s0),
        elasticity_rho=_elasticity(model.decay.rho, value, grad_rho),
        elasticity_q=_elasticity(model.q, value, grad_q),
    )


def crossover_sensitivity(
    model: RecoveryModel,
    tau: float,
    *,
    h: float = 1e-4,
    max_depth: int = 64,
) -> CrossoverSensitivity:
    r"""Finite-difference sensitivity of the crossover depth :math:`d^\star`.

    Perturbs each of :math:`s_0`, :math:`\rho`, :math:`q` by :math:`\pm h`,
    re-solves :func:`~membench.theory.crossover.find_crossover_depth`, and takes a
    central difference of the continuous crossover ``d_star_continuous``. Use an
    interior base point so the perturbed :math:`q \pm h` stays within
    :math:`(0, 1]`.

    Parameters
    ----------
    model
        The recovery model defining the base point :math:`(s_0, \rho, q)`.
    tau
        Overhead ratio :math:`\tau = W/V \ge 0`.
    h
        Finite-difference step for the parameter perturbations.
    max_depth
        Largest integer depth scanned by the crossover solver.

    Returns
    -------
    CrossoverSensitivity
        Central-difference gradients of :math:`d^\star` at the base point.

    Raises
    ------
    ValueError
        If a crossover does not exist at the base point or at any perturbed point
        (so the derivative is undefined), or if ``h`` is not positive.
    """
    if h <= 0:
        raise ValueError(f"step h must be positive, got {h}")

    def d_star_at(s0: float, rho: float, q: float) -> float:
        result = find_crossover_depth(
            RecoveryModel(decay=SimilarityDecayModel(s0=s0, rho=rho), q=q),
            tau,
            max_depth=max_depth,
        )
        if result.d_star_continuous is None:
            raise ValueError("no crossover at this point; d* is not differentiable here")
        return result.d_star_continuous

    s0 = model.decay.s0
    rho = model.decay.rho
    q = model.q
    base = d_star_at(s0, rho, q)
    grad_s0 = (d_star_at(s0 + h, rho, q) - d_star_at(s0 - h, rho, q)) / (2.0 * h)
    grad_rho = (d_star_at(s0, rho + h, q) - d_star_at(s0, rho - h, q)) / (2.0 * h)
    grad_q = (d_star_at(s0, rho, q + h) - d_star_at(s0, rho, q - h)) / (2.0 * h)
    return CrossoverSensitivity(
        tau=tau,
        d_star=base,
        grad_s0=grad_s0,
        grad_rho=grad_rho,
        grad_q=grad_q,
        step=h,
    )
