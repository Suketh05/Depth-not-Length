r"""Closed-form brackets for the crossover depth :math:`d^\star`.

The numeric solver in :mod:`membench.theory.crossover` locates the depth at which
structured recovery overtakes similarity recovery by *scanning* integer depths.
This module derives the same crossover analytically, giving an :math:`O(1)`
lower/upper bracket on :math:`d^\star` plus a closed-form "win window" of depths,
so the solver's answer can be independently checked and so plotting / sweeps can
skip the scan.

Setup
-----
With the geometric similarity-decay law :math:`s_i = s_0 \rho^i` (Definition 1 of
the membench theory note) the two recovery regimes are

.. math::

    P_\text{struct}(d) = q^{\,d},
    \qquad
    P_\text{sim}(d) = s_0^{\,d}\, \rho^{\,d(d+1)/2},

and structured memory clears its per-chain overhead ratio :math:`\tau = W/V` when

.. math::

    g(d) = P_\text{struct}(d) - P_\text{sim}(d) > \tau .

Take logs of the *advantage* :math:`P_\text{struct}/P_\text{sim}`. Writing
:math:`a = \ln(q/s_0)` and :math:`k = -\ln\rho > 0` (since :math:`\rho < 1`),

.. math::

    \log\frac{P_\text{struct}(d)}{P_\text{sim}(d)}
        = a\,d + \tfrac{k}{2}\,d(d+1)
        = \tfrac{k}{2}\,d^{2} + \bigl(a + \tfrac{k}{2}\bigr) d ,

an upward parabola with roots at :math:`d = 0` and

.. math::

    d_0 = \frac{2\ln(q/s_0)}{\ln\rho} - 1 .

So :math:`g(d) > 0` **iff** :math:`d > d_0` (Proposition 1, the *depth ceiling*):
below :math:`d_0` similarity wins, above it structure wins, and the advantage
ratio diverges as :math:`d \to \infty`.

The bracket
-----------
*Lower bound.* Winning at overhead :math:`\tau \ge 0` requires :math:`g(d) > \tau
\ge 0`, hence :math:`g(d) > 0`, hence :math:`d > d_0`. Every integer depth
:math:`\le \lfloor d_0 \rfloor` therefore provably loses, so

.. math::

    d^\star \ge \lfloor d_0 \rfloor + 1 .

At :math:`\tau = 0` this is *exact*: the smallest winning depth is precisely
:math:`\lfloor d_0\rfloor + 1`.

*Upper bound.* Because :math:`P_\text{sim} > 0`, a winning depth needs the
structured term alone to clear the overhead: :math:`q^{d} \ge g(d) + P_\text{sim}
> \tau`, hence :math:`d < \ln\tau / \ln q` for :math:`0 < \tau < q`. Every winning
depth — and so the whole win window, and in particular :math:`d^\star` — lies
below that bound, giving

.. math::

    d^\star \le \Bigl\lfloor \frac{\ln\tau}{\ln q} \Bigr\rfloor .

For :math:`\tau \ge q` the structured term never clears the overhead
(:math:`q^d \le q \le \tau`) and no crossover exists. For :math:`q = 1` the
advantage is monotone (Proposition 2 holds only in this limit), so the win window
is the half-line :math:`[d^\star, \infty)` and the upper bound is unbounded
(reported as ``max_depth``).

The win window is the contiguous depth interval in which a win is *possible*: the
true win region computed by :func:`~membench.theory.crossover.find_crossover_depth`
is always a subset of it.

References
----------
membench theory note, Definition 1 (geometric similarity decay) and
Propositions 1-2 (depth ceiling and the structure-vs-overhead crossover); see the
companion derivations in :mod:`membench.theory.recovery` and
:mod:`membench.theory.crossover`. The reduction of a difference of a geometric
sequence and a super-exponential (Gaussian-tailed) sequence to a single sign
change of a quadratic in log-space is the standard log-linearisation argument
(e.g. de Bruijn, *Asymptotic Methods in Analysis*, 1981, ch. 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from membench.theory.recovery import RecoveryModel

__all__ = ["CrossoverBounds", "bracket_crossover_depth", "tau_zero_crossover"]

# Guard against float undershoot when flooring the q^d == tau depth: a true upper
# bound rounded a hair low would wrongly exclude the deepest winning depth.
_FLOOR_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class CrossoverBounds:
    r"""Closed-form bracket of the crossover depth :math:`d^\star`.

    Attributes
    ----------
    tau
        The overhead ratio :math:`\tau = W/V` the bracket was derived for.
    d_lower
        Closed-form lower bound on :math:`d^\star` (every shallower integer depth
        provably loses). Exact at :math:`\tau = 0`.
    d_upper
        Closed-form upper bound on :math:`d^\star` (every deeper integer depth
        provably loses, because the structured term :math:`q^d` itself drops below
        :math:`\tau`). Clipped to ``max_depth``; equals ``d_lower`` at
        :math:`\tau = 0` and ``max_depth`` in the monotone :math:`q = 1` limit.
    win_window
        Inclusive integer interval of depths where a win is *possible* (a depth
        outside it provably cannot clear the overhead). ``None`` when no crossover
        exists. The numeric solver's ``win_region`` is always a subset of this.
    """

    tau: float
    d_lower: int
    d_upper: int
    win_window: tuple[int, int] | None


def tau_zero_crossover(model: RecoveryModel) -> float:
    r"""Return the continuous zero-overhead crossover :math:`d_0`.

    This is the nonzero root of the log-advantage parabola, i.e. the depth at
    which :math:`P_\text{struct}(d) = P_\text{sim}(d)`:

    .. math::

        d_0 = \frac{2\ln(q/s_0)}{\ln\rho} - 1 .

    Parameters
    ----------
    model
        Recovery model (decay law + structured per-link reliability ``q``); its
        decay must be genuinely decaying (:math:`\rho < 1`).

    Returns
    -------
    float
        The continuous crossover depth :math:`d_0`. For :math:`d > d_0` structured
        recovery exceeds similarity recovery; for :math:`d < d_0` it does not.

    Raises
    ------
    ValueError
        If the decay model does not decay (:math:`\rho \ge 1`), so the quadratic
        depth-ceiling argument does not apply.
    """
    if not model.decay.is_decaying:
        raise ValueError(
            "crossover bounds require a decaying similarity model (rho < 1), "
            f"got rho={model.decay.rho}"
        )
    ln_q = math.log(model.q)
    ln_s0 = math.log(model.decay.s0)
    ln_rho = math.log(model.decay.rho)
    return 2.0 * (ln_q - ln_s0) / ln_rho - 1.0


def bracket_crossover_depth(
    model: RecoveryModel,
    tau: float,
    *,
    max_depth: int = 64,
) -> CrossoverBounds:
    r"""Bracket the crossover depth :math:`d^\star` in closed form.

    Derives an :math:`O(1)` lower/upper bound on the smallest depth at which
    structured recovery clears the overhead ratio ``tau``, together with the
    closed-form win window (see the module docstring for the derivation). The
    bounds are guaranteed to bracket
    :func:`~membench.theory.crossover.find_crossover_depth`'s ``d_star`` whenever a
    crossover exists, and the win window always contains the solver's ``win_region``.

    Parameters
    ----------
    model
        Recovery model (decay law + structured per-link reliability ``q``). Its
        decay must satisfy :math:`\rho < 1`.
    tau
        Overhead ratio :math:`\tau = W/V \ge 0`. ``tau = 0`` brackets the depth at
        which structure merely first exceeds similarity.
    max_depth
        Largest integer depth considered; unbounded upper bounds (at ``tau = 0`` or
        ``q = 1``) are clipped to this, matching the solver's scan range.

    Returns
    -------
    CrossoverBounds
        The closed-form bracket and win window.

    Raises
    ------
    ValueError
        If ``tau`` is negative, ``max_depth`` is below 1, or the decay model does
        not decay (:math:`\rho \ge 1`).
    """
    if tau < 0:
        raise ValueError(f"tau must be non-negative, got {tau}")
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")

    d0 = tau_zero_crossover(model)  # also validates rho < 1
    d_lower = max(1, math.floor(d0) + 1)

    q = model.q
    ln_q = math.log(q)

    if tau == 0.0:
        # Rising crossover is pinned exactly; the win window runs to the scan edge.
        d_upper = min(d_lower, max_depth)
        win_hi = max_depth
    elif tau >= q:
        # q^d <= q <= tau for every d >= 1: structure can never clear the overhead.
        return CrossoverBounds(tau=tau, d_lower=d_lower, d_upper=d_lower, win_window=None)
    elif ln_q == 0.0:
        # q == 1: q^d = 1 > tau for all d, so the win window is unbounded above.
        d_upper = max_depth
        win_hi = max_depth
    else:
        # Winners satisfy q^d > tau, i.e. d < ln(tau)/ln(q).
        u = math.log(tau) / ln_q
        d_upper = min(math.floor(u + _FLOOR_EPS), max_depth)
        win_hi = d_upper

    win_window = (d_lower, win_hi) if d_lower <= win_hi else None
    return CrossoverBounds(tau=tau, d_lower=d_lower, d_upper=d_upper, win_window=win_window)
