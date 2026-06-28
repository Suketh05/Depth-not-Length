r"""Bayes factors for binomial proportions.

A Bayes factor :math:`BF_{10} = m_1 / m_0` is the ratio of the marginal likelihoods
of the data under the alternative (:math:`H_1`) and the null (:math:`H_0`). Unlike a
p-value it can quantify evidence *for* the null, which is exactly what an "arm A is no
better than arm B" claim needs.

**Single proportion vs a point null.** For ``s`` successes in ``n`` Bernoulli trials,
:math:`H_0` fixes the rate at a point ``p0`` while :math:`H_1` places a conjugate
``Beta(a, b)`` prior on the rate. The marginal likelihood under the prior is the
Beta-Binomial: integrating the binomial likelihood against the Beta prior gives

.. math::

    m_1 = \binom{n}{s}\,\frac{B(a+s,\; b+n-s)}{B(a,\; b)},

and the point null contributes the ordinary binomial pmf
:math:`m_0 = \binom{n}{s}\,p_0^{s}(1-p_0)^{n-s}`. The binomial coefficient cancels, so

.. math::

    BF_{10} = \frac{B(a+s,\; b+n-s)}{B(a,\; b)\; p_0^{s}\,(1-p_0)^{n-s}}.

This is the standard conjugate Bayes factor for a proportion (Jeffreys, 1961; Kass &
Raftery, 1995). The default prior is Jeffreys' ``Beta(1/2, 1/2)``; the uniform
``Beta(1, 1)`` reduces ``B(a, b)`` to one.

**Two proportions (BIC approximation).** Comparing whether two groups share a rate
(:math:`H_0`, one pooled parameter) or differ (:math:`H_1`, two free parameters) has no
simple closed form, so we use the Schwarz/BIC approximation
:math:`BF_{10} \approx \exp\!\big((BIC_0 - BIC_1)/2\big)` with
:math:`BIC = k\ln N - 2\ln \hat{L}` evaluated at the maximum-likelihood proportions and
total sample size :math:`N = n_a + n_b` (Kass & Raftery, 1995; Wagenmakers, 2007).

Evidence is labelled on the Jeffreys scale as refined by Lee & Wagenmakers (2013):
anecdotal (1-3), moderate (3-10), strong (10-30), very strong (30-100), extreme (>100),
with the reciprocal favouring :math:`H_0`.

References
----------
Jeffreys, H. (1961). *Theory of Probability* (3rd ed.). Oxford University Press.
Kass, R. E., & Raftery, A. E. (1995). Bayes factors. *Journal of the American
Statistical Association*, 90(430), 773-795.
Wagenmakers, E.-J. (2007). A practical solution to the pervasive problems of p values.
*Psychonomic Bulletin & Review*, 14(5), 779-804.
Lee, M. D., & Wagenmakers, E.-J. (2013). *Bayesian Cognitive Modeling*. Cambridge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.special import betaln

__all__ = ["BayesFactor", "bayes_factor_proportion", "bayes_factor_two_proportion_bic"]


@dataclass(frozen=True, slots=True)
class BayesFactor:
    """A Bayes factor with its natural-log value and a Jeffreys-scale label.

    Attributes
    ----------
    bf10 : float
        The Bayes factor :math:`m_1 / m_0` (evidence for :math:`H_1` when ``> 1``).
    log_bf : float
        The natural logarithm of ``bf10`` (numerically stable for extreme evidence).
    interpretation : str
        Jeffreys-scale verdict, e.g. ``"strong evidence for H1"``.
    """

    bf10: float
    log_bf: float
    interpretation: str


def _interpret(log_bf: float) -> str:
    """Return the Jeffreys-scale verdict for a natural-log Bayes factor."""
    if log_bf == 0.0:
        return "no evidence either way"
    favour = "H1" if log_bf > 0.0 else "H0"
    ratio = math.exp(abs(log_bf))
    if ratio < 3.0:
        strength = "anecdotal"
    elif ratio < 10.0:
        strength = "moderate"
    elif ratio < 30.0:
        strength = "strong"
    elif ratio < 100.0:
        strength = "very strong"
    else:
        strength = "extreme"
    return f"{strength} evidence for {favour}"


def _xlogy(x: float, y: float) -> float:
    """Return ``x * log(y)`` with the convention ``0 * log(0) = 0``."""
    if x == 0.0:
        return 0.0
    return x * math.log(y)


def _binom_loglik(successes: int, trials: int, p: float) -> float:
    """Return the binomial log-likelihood ``s*log(p) + (n-s)*log(1-p)``."""
    return _xlogy(successes, p) + _xlogy(trials - successes, 1.0 - p)


def bayes_factor_proportion(
    successes: int,
    trials: int,
    *,
    p0: float = 0.5,
    prior: tuple[float, float] = (0.5, 0.5),
) -> BayesFactor:
    r"""Return the Beta-Binomial Bayes factor for a proportion vs a point null ``p0``.

    Computes :math:`BF_{10} = B(a+s, b+n-s) / [B(a, b)\, p_0^{s}(1-p_0)^{n-s}]`, the
    ratio of the conjugate Beta marginal likelihood (:math:`H_1`) to the point-null
    binomial likelihood (:math:`H_0`). Evaluated in log-space for stability.
    """
    if trials < 0 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials")
    if not 0.0 < p0 < 1.0:
        raise ValueError("p0 must be in (0, 1)")
    a0, b0 = prior
    if a0 <= 0.0 or b0 <= 0.0:
        raise ValueError("prior parameters must be positive")
    failures = trials - successes
    log_m1 = float(betaln(a0 + successes, b0 + failures) - betaln(a0, b0))
    log_m0 = successes * math.log(p0) + failures * math.log(1.0 - p0)
    log_bf = log_m1 - log_m0
    return BayesFactor(bf10=math.exp(log_bf), log_bf=log_bf, interpretation=_interpret(log_bf))


def bayes_factor_two_proportion_bic(
    successes_a: int,
    trials_a: int,
    successes_b: int,
    trials_b: int,
) -> BayesFactor:
    r"""Return the BIC-approximated Bayes factor comparing two proportions.

    :math:`H_0` assigns both groups a single pooled rate (one parameter); :math:`H_1`
    gives each its own rate (two parameters). Using
    :math:`BIC = k\ln N - 2\ln\hat{L}` at the MLE proportions with
    :math:`N = n_a + n_b`, the Schwarz approximation gives
    :math:`BF_{10} \approx \exp\!\big((BIC_0 - BIC_1)/2\big)` (Wagenmakers, 2007).
    """
    if trials_a <= 0 or trials_b <= 0:
        raise ValueError("each group needs at least one trial")
    if not (0 <= successes_a <= trials_a and 0 <= successes_b <= trials_b):
        raise ValueError("require 0 <= successes <= trials")
    total = trials_a + trials_b
    pooled = (successes_a + successes_b) / total
    loglik_null = _binom_loglik(successes_a, trials_a, pooled) + _binom_loglik(
        successes_b, trials_b, pooled
    )
    loglik_alt = _binom_loglik(successes_a, trials_a, successes_a / trials_a) + _binom_loglik(
        successes_b, trials_b, successes_b / trials_b
    )
    log_n = math.log(total)
    bic_null = 1 * log_n - 2.0 * loglik_null
    bic_alt = 2 * log_n - 2.0 * loglik_alt
    log_bf = (bic_null - bic_alt) / 2.0
    return BayesFactor(bf10=math.exp(log_bf), log_bf=log_bf, interpretation=_interpret(log_bf))
