r"""The crossover depth :math:`d^\star`: where structure starts to pay.

On pure recovery the structured store always looks better, so to be honest it
must be charged for what it costs. Let :math:`V` be the value of solving a task
correctly and :math:`W` the extra per-chain overhead of building and maintaining
typed links. Structured memory is the better choice when its net recovery gain
clears that overhead:

.. math::

    V\,\bigl(q^{d} - s_0^{d}\rho^{d(d+1)/2}\bigr) > W
    \quad\Longleftrightarrow\quad
    g(d) := P_\text{struct}(d) - P_\text{sim}(d) > \tau,
    \qquad \tau = W/V.

**An honesty note the theory sketch glosses over.** Proposition 2 asserts ``g`` is
increasing, which holds only in the :math:`q = 1` limit (no structured decay).
For :math:`q < 1` both terms vanish as :math:`d \to \infty`, so ``g`` rises, peaks,
then falls back toward zero — the win is an *interval*
:math:`[d^\star, d_\text{upper}]`, not a half-line. We therefore solve for the
crossover numerically and report the full win region rather than asserting a
half-line that the model does not actually produce. The smallest integer depth in
that region is :math:`d^\star`; a continuous root on the rising edge (via Brent's
method) gives a sub-integer crossover estimate for plotting and for comparison
against the empirically measured changepoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from membench.theory.recovery import RecoveryModel

__all__ = ["CrossoverResult", "find_crossover_depth", "g", "net_value_gain", "overhead_ratio"]


def overhead_ratio(maintenance_cost: float, task_value: float) -> float:
    r"""Overhead ratio :math:`\tau = W/V` (per-chain link cost over value-of-correct)."""
    if task_value <= 0:
        raise ValueError(f"task_value V must be positive, got {task_value}")
    if maintenance_cost < 0:
        raise ValueError(f"maintenance_cost W must be non-negative, got {maintenance_cost}")
    return maintenance_cost / task_value


def g(d: float, model: RecoveryModel) -> float:
    r"""Net recovery gain :math:`g(d) = P_\text{struct}(d) - P_\text{sim}(d)`.

    Defined for continuous ``d`` (the closed forms accept real exponents) so the
    crossover can be located between integers.
    """
    p_struct = model.q**d
    p_sim = model.decay.s0**d * model.decay.rho ** (d * (d + 1) / 2)
    return float(p_struct - p_sim)


def net_value_gain(
    d: float, model: RecoveryModel, task_value: float, maintenance_cost: float
) -> float:
    r"""Net value of structured over similarity memory: :math:`V g(d) - W`."""
    return task_value * g(d, model) - maintenance_cost


@dataclass(frozen=True, slots=True)
class CrossoverResult:
    r"""The crossover analysis at a given overhead ratio :math:`\tau`.

    Attributes
    ----------
    tau
        The overhead ratio the gain was tested against.
    exists
        Whether structured memory's net gain ever clears :math:`\tau`.
    d_star
        Smallest integer depth in the win region (``None`` if it never wins).
    d_star_continuous
        Sub-integer crossover on the rising edge of ``g`` (``None`` if no win).
    win_region
        Inclusive integer depth interval where ``g(d) > tau`` (``None`` if empty).
    peak_depth
        Integer depth maximising ``g`` over the scanned range (the depth of
        maximal structured advantage).
    g_curve
        ``g`` evaluated at integer depths ``1..max_depth``.
    """

    tau: float
    exists: bool
    d_star: int | None
    d_star_continuous: float | None
    win_region: tuple[int, int] | None
    peak_depth: int
    g_curve: NDArray[np.float64]


def find_crossover_depth(
    model: RecoveryModel,
    tau: float,
    *,
    max_depth: int = 64,
) -> CrossoverResult:
    r"""Locate the crossover depth :math:`d^\star` for overhead ratio ``tau``.

    Scans integer depths ``1..max_depth`` for the contiguous region where
    :math:`g(d) > \tau`, takes its smallest integer as :math:`d^\star`, and refines
    a continuous crossover on the rising edge with Brent's method.

    Parameters
    ----------
    model
        The recovery model (decay law + structured per-link reliability ``q``).
    tau
        Overhead ratio :math:`\tau = W/V \ge 0`. ``tau = 0`` asks where structured
        recovery merely first exceeds similarity recovery.
    max_depth
        Largest integer depth scanned.

    Returns
    -------
    CrossoverResult
        The crossover depth, win region, and diagnostic curve.
    """
    if tau < 0:
        raise ValueError(f"tau must be non-negative, got {tau}")
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")

    depths = np.arange(1, max_depth + 1)
    g_curve = np.array([g(int(d), model) for d in depths], dtype=np.float64)
    above = g_curve > tau
    peak_depth = int(depths[int(np.argmax(g_curve))])

    if not above.any():
        return CrossoverResult(
            tau=tau,
            exists=False,
            d_star=None,
            d_star_continuous=None,
            win_region=None,
            peak_depth=peak_depth,
            g_curve=g_curve,
        )

    first_idx = int(np.argmax(above))  # first True
    d_star = int(depths[first_idx])

    # Contiguous win region starting at d_star (the win is an interval for q < 1).
    last_idx = first_idx
    while last_idx + 1 < above.size and above[last_idx + 1]:
        last_idx += 1
    win_region = (d_star, int(depths[last_idx]))

    # Continuous crossover on the rising edge: g(d) - tau changes sign in
    # (d_star - 1, d_star] unless the model already wins at d = 1.
    d_continuous: float | None
    lo = float(d_star - 1)
    if lo < 1e-9 or g(lo, model) - tau >= 0:
        d_continuous = float(d_star)
    else:
        d_continuous = float(optimize.brentq(lambda d: g(d, model) - tau, lo, float(d_star)))

    return CrossoverResult(
        tau=tau,
        exists=True,
        d_star=d_star,
        d_star_continuous=d_continuous,
        win_region=win_region,
        peak_depth=peak_depth,
        g_curve=g_curve,
    )
