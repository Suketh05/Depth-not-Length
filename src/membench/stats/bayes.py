r"""Bayesian Beta-Binomial inference for per-arm rates.

Compliance and merge-ready correctness are binomial outcomes measured on a few dozen
tasks -- small n, where normal-approximation intervals misbehave near 0 and 1. The
Beta-Binomial conjugate model is the right tool: with a ``Beta(a, b)`` prior and
``s`` successes in ``n`` trials, the posterior is ``Beta(a+s, b+n-s)``, giving exact
credible intervals and -- crucially for the comparison story -- the directly
defensible statement ``P(rate_Brief > rate_competitor)`` via the two posteriors.

The default prior is Jeffreys' ``Beta(1/2, 1/2)`` (a standard objective prior for a
proportion); a uniform ``Beta(1, 1)`` is also available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

__all__ = ["BetaPosterior", "beta_binomial_posterior", "prob_superiority"]


@dataclass(frozen=True, slots=True)
class BetaPosterior:
    """A Beta posterior over a binomial rate."""

    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        """Return the posterior mean ``alpha / (alpha + beta)``."""
        return self.alpha / (self.alpha + self.beta)

    def credible_interval(self, cred: float = 0.95) -> tuple[float, float]:
        """Return the equal-tailed credible interval at level ``cred``."""
        if not 0.0 < cred < 1.0:
            raise ValueError("cred must be in (0, 1)")
        tail = (1.0 - cred) / 2.0
        lo, hi = stats.beta.ppf([tail, 1.0 - tail], self.alpha, self.beta)
        return float(lo), float(hi)

    def sample(self, n: int, seed: int = 0) -> NDArray[np.float64]:
        """Draw ``n`` posterior samples of the rate (seeded)."""
        return np.asarray(stats.beta.rvs(self.alpha, self.beta, size=n, random_state=seed))


def beta_binomial_posterior(
    successes: int, trials: int, *, prior: tuple[float, float] = (0.5, 0.5)
) -> BetaPosterior:
    """Return the Beta posterior for ``successes`` of ``trials`` under a Beta prior."""
    if trials < 0 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials")
    a0, b0 = prior
    if a0 <= 0 or b0 <= 0:
        raise ValueError("prior parameters must be positive")
    return BetaPosterior(alpha=a0 + successes, beta=b0 + (trials - successes))


def prob_superiority(
    a: BetaPosterior,
    b: BetaPosterior,
    *,
    n_samples: int = 200_000,
    seed: int = 0,
) -> float:
    r"""Return ``P(rate_a > rate_b)`` from the two posteriors via Monte Carlo.

    The headline comparison statement (e.g. "Brief's compliance exceeds the
    competitor's with probability 0.98"). Seeded for reproducibility.
    """
    sa = a.sample(n_samples, seed=seed)
    sb = b.sample(n_samples, seed=seed + 1)
    return float(np.mean(sa > sb))
