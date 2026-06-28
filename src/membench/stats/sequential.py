r"""Always-valid sequential testing of a Bernoulli mean by betting.

Benchmark rates (compliance, merge-ready correctness) accrue one task at a time, and
we want to peek as the data arrives without inflating the type-I error the way a
fixed-sample test would. The *betting* / test-martingale construction gives an
**anytime-valid** answer: a nonnegative process whose value can be inspected at every
step, with a hard guarantee that the false-rejection probability over the *entire*
(possibly infinite) run stays below ``alpha``.

Capital process
---------------
To test :math:`H_0:\ \mu = m_0` for outcomes :math:`x_t \in \{0, 1\}` with a
predictable bet fraction :math:`\lambda \in (0, 1)`, maintain the capital (wealth)

.. math::

    e_t \;=\; \prod_{s=1}^{t} \bigl(1 + \lambda\,(x_s - m_0)\bigr).

Because :math:`\mathbb{E}[x_s - m_0 \mid \mathcal{F}_{s-1}] = 0` under :math:`H_0`,
each factor has conditional mean one, so :math:`(e_t)` is a nonnegative martingale
with :math:`\mathbb{E}[e_t] = 1` -- i.e. :math:`e_t` is an *e-value*. **Ville's
inequality** then bounds the chance the capital ever reaches :math:`1/\alpha`:

.. math::

    \mathbb{P}_{H_0}\!\left(\exists\, t:\ e_t \ge 1/\alpha\right) \le \alpha,

so the level-:math:`\alpha` sequential test rejects the first time :math:`e_t \ge
1/\alpha` and is valid at every stopping time.

Confidence sequence
-------------------
Inverting the family of tests over candidate means yields a *confidence sequence*: a
sequence of intervals :math:`(C_t)` covering the true mean simultaneously for all
:math:`t` with probability :math:`\ge 1 - \alpha`. For two-sided coverage we use the
hedged capital process of Waudby-Smith & Ramdas -- the equal mixture of an upward and
a downward bet,

.. math::

    M_t(m) = \tfrac12 \prod_{s\le t}\!\bigl(1 + \lambda(x_s - m)\bigr)
           + \tfrac12 \prod_{s\le t}\!\bigl(1 - \lambda(x_s - m)\bigr),

which is itself a nonnegative mean-one e-process for every fixed :math:`m`. The
confidence sequence is :math:`C_t = \{m : \max_{s\le t} M_s(m) < 1/\alpha\}`, reported
as a running intersection so the interval only shrinks.

References
----------
* Waudby-Smith, I. & Ramdas, A. (2024). "Estimating means of bounded random variables
  by betting." *Journal of the Royal Statistical Society: Series B* 86(1), 1-27.
* Howard, S. R., Ramdas, A., McAuliffe, J. & Sekhon, J. (2021). "Time-uniform,
  nonparametric, nonasymptotic confidence sequences." *Annals of Statistics* 49(2),
  1055-1080.
* Shafer, G. (2021). "Testing by betting: A strategy for statistical and scientific
  communication." *JRSS-A* 184(2), 407-431. (Ville's inequality, 1939.)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ConfidenceSequence",
    "SequentialTestResult",
    "betting_capital_process",
    "confidence_sequence",
    "sequential_test",
]


def _as_binary(outcomes: Sequence[int] | Sequence[float]) -> NDArray[np.float64]:
    """Validate ``outcomes`` as a non-empty 1-D Bernoulli sequence and return it."""
    arr = np.asarray(outcomes, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("outcomes must be a non-empty 1-D sequence")
    if not bool(np.all((arr == 0.0) | (arr == 1.0))):
        raise ValueError("outcomes must be Bernoulli (each value 0 or 1)")
    return arr


def betting_capital_process(
    outcomes: Sequence[int] | Sequence[float],
    null_mean: float,
    *,
    bet: float = 0.5,
    direction: str = "up",
) -> NDArray[np.float64]:
    r"""Return the capital path :math:`e_1, \dots, e_n` for betting against ``null_mean``.

    Parameters
    ----------
    outcomes : sequence of {0, 1}
        The observed Bernoulli outcomes, in order.
    null_mean : float
        The hypothesised mean :math:`m_0`, in the open interval ``(0, 1)``.
    bet : float, optional
        Fixed (predictable) bet fraction :math:`\lambda \in (0, 1)`, by default
        ``0.5``. Restricting to ``(0, 1)`` keeps every factor strictly positive for
        any ``null_mean``, so the capital stays a valid nonnegative e-process.
    direction : {"up", "down"}, optional
        ``"up"`` bets that the true mean exceeds ``null_mean`` (factor
        :math:`1 + \lambda(x - m_0)`); ``"down"`` bets it is smaller (factor
        :math:`1 - \lambda(x - m_0)`). By default ``"up"``.

    Returns
    -------
    numpy.ndarray
        The running capital :math:`e_t`, one entry per outcome. Under ``H_0`` each
        :math:`e_t` is an e-value with :math:`\mathbb{E}[e_t] = 1`.
    """
    x = _as_binary(outcomes)
    if not 0.0 < null_mean < 1.0:
        raise ValueError("null_mean must be in the open interval (0, 1)")
    if not 0.0 < bet < 1.0:
        raise ValueError("bet must be in the open interval (0, 1)")
    if direction not in ("up", "down"):
        raise ValueError("direction must be 'up' or 'down'")
    sign = 1.0 if direction == "up" else -1.0
    factors = 1.0 + sign * bet * (x - null_mean)
    return np.asarray(np.cumprod(factors), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SequentialTestResult:
    r"""Outcome of an always-valid sequential test of a Bernoulli mean.

    Attributes
    ----------
    null_mean : float
        The hypothesised mean :math:`m_0` that was tested.
    alpha : float
        The target type-I error level; the test rejects when ``capital`` reaches
        :math:`1/\alpha`.
    capital : float
        The final capital :math:`e_n` (the e-value at the last observation).
    rejected : bool
        Whether the capital reached :math:`1/\alpha` at any step.
    stopping_time : int or None
        The 1-indexed step at which the threshold was first crossed, or ``None`` if
        it never was.
    n : int
        The number of observations.
    capital_path : numpy.ndarray
        The full capital trajectory :math:`e_1, \dots, e_n`.
    """

    null_mean: float
    alpha: float
    capital: float
    rejected: bool
    stopping_time: int | None
    n: int
    capital_path: NDArray[np.float64]


def sequential_test(
    outcomes: Sequence[int] | Sequence[float],
    null_mean: float,
    *,
    alpha: float = 0.05,
    bet: float = 0.5,
    direction: str = "up",
) -> SequentialTestResult:
    r"""Run a betting-based always-valid test of ``H_0: mean == null_mean``.

    Rejects the first time the capital process reaches :math:`1/\alpha`. By Ville's
    inequality the probability of *ever* rejecting under :math:`H_0` is at most
    ``alpha``, so the verdict is valid even under continuous monitoring / optional
    stopping.

    Parameters
    ----------
    outcomes : sequence of {0, 1}
        The observed Bernoulli outcomes, in order.
    null_mean : float
        The hypothesised mean :math:`m_0`, in ``(0, 1)``.
    alpha : float, optional
        Target type-I error level in ``(0, 1)``, by default ``0.05``.
    bet : float, optional
        Fixed bet fraction in ``(0, 1)``, by default ``0.5``.
    direction : {"up", "down"}, optional
        Whether the alternative places the mean above or below ``null_mean``, by
        default ``"up"``.

    Returns
    -------
    SequentialTestResult
        The capital trajectory, the reject decision, and the stopping time.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in the open interval (0, 1)")
    path = betting_capital_process(outcomes, null_mean, bet=bet, direction=direction)
    threshold = 1.0 / alpha
    crossings = np.flatnonzero(path >= threshold)
    rejected = crossings.size > 0
    stopping_time = int(crossings[0]) + 1 if rejected else None
    return SequentialTestResult(
        null_mean=null_mean,
        alpha=alpha,
        capital=float(path[-1]),
        rejected=rejected,
        stopping_time=stopping_time,
        n=int(path.size),
        capital_path=path,
    )


@dataclass(frozen=True, slots=True)
class ConfidenceSequence:
    r"""A two-sided confidence sequence for a Bernoulli mean.

    Attributes
    ----------
    lower : numpy.ndarray
        Running lower bounds :math:`L_t`, one per observation (monotone non-decreasing).
    upper : numpy.ndarray
        Running upper bounds :math:`U_t`, one per observation (monotone non-increasing).
    confidence : float
        Simultaneous coverage level :math:`1 - \alpha`.
    """

    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    confidence: float

    @property
    def final(self) -> tuple[float, float]:
        """Return the final interval ``(L_n, U_n)``."""
        return float(self.lower[-1]), float(self.upper[-1])


def confidence_sequence(
    outcomes: Sequence[int] | Sequence[float],
    *,
    confidence: float = 0.95,
    bet: float = 0.5,
    grid_size: int = 999,
) -> ConfidenceSequence:
    r"""Compute the hedged-capital confidence sequence for a Bernoulli mean.

    For each candidate mean ``m`` on a uniform grid of the open interval ``(0, 1)``
    the hedged capital process :math:`M_t(m)` is evaluated; ``m`` is retained in the
    interval at step ``t`` while :math:`\max_{s \le t} M_s(m) < 1/\alpha`. The
    reported interval is the running intersection over time, so it only shrinks.

    Parameters
    ----------
    outcomes : sequence of {0, 1}
        The observed Bernoulli outcomes, in order.
    confidence : float, optional
        Simultaneous coverage level :math:`1 - \alpha` in ``(0, 1)``, by default
        ``0.95``.
    bet : float, optional
        Fixed bet fraction in ``(0, 1)``, by default ``0.5``.
    grid_size : int, optional
        Number of interior grid points used to invert the test, by default ``999``
        (grid resolution :math:`1/(grid\_size + 1)`).

    Returns
    -------
    ConfidenceSequence
        The running lower and upper bounds and the coverage level.
    """
    x = _as_binary(outcomes)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in the open interval (0, 1)")
    if not 0.0 < bet < 1.0:
        raise ValueError("bet must be in the open interval (0, 1)")
    if grid_size < 1:
        raise ValueError("grid_size must be a positive integer")

    alpha = 1.0 - confidence
    threshold = 1.0 / alpha
    grid = np.linspace(0.0, 1.0, grid_size + 2)[1:-1]

    diff = x[None, :] - grid[:, None]  # (grid, n)
    up = np.cumprod(1.0 + bet * diff, axis=1)
    down = np.cumprod(1.0 - bet * diff, axis=1)
    hedged = 0.5 * up + 0.5 * down
    running_max = np.maximum.accumulate(hedged, axis=1)
    in_set = running_max < threshold  # (grid, n)

    grid_col = grid[:, None]
    lowers = np.where(in_set, grid_col, np.inf).min(axis=0)
    uppers = np.where(in_set, grid_col, -np.inf).max(axis=0)
    empty = ~in_set.any(axis=0)
    lowers = np.where(empty, 0.0, lowers)
    uppers = np.where(empty, 1.0, uppers)

    lower = np.asarray(np.maximum.accumulate(lowers), dtype=np.float64)
    upper = np.asarray(np.minimum.accumulate(uppers), dtype=np.float64)
    return ConfidenceSequence(lower=lower, upper=upper, confidence=confidence)
