r"""Model selection for the similarity-decay law.

The theory assumes geometric decay (:math:`s_i = s_0 \rho^i`). A reviewer will ask
whether that functional form is actually the best description of the data, or merely
assumed. This module fits three candidate decay laws to measured (distance,
similarity) pairs and ranks them by AIC/BIC:

* **geometric** :math:`s_0 \rho^i` -- linear in log-space (the theory's law);
* **power-law** :math:`s_0 (i+1)^{-\alpha}` -- linear in log-log-space;
* **stretched-exponential** :math:`s_0 e^{-(\lambda i)^\beta}` -- a flexible
  three-parameter alternative (non-linear fit).

All three are scored by the Gaussian log-likelihood of their log-space residuals, so
AIC/BIC are comparable. Geometric winning (or tying) is the honest justification for
the model the theory uses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

__all__ = ["ModelFit", "aic", "bic", "compare_decay_models"]


def _gaussian_loglik(rss: float, n: int) -> float:
    if rss <= 0:
        rss = 1e-12
    return -0.5 * n * (math.log(2 * math.pi) + math.log(rss / n) + 1.0)


def aic(rss: float, n: int, k: int) -> float:
    """Return the Akaike Information Criterion from residual sum of squares."""
    return 2 * k - 2 * _gaussian_loglik(rss, n)


def bic(rss: float, n: int, k: int) -> float:
    """Return the Bayesian Information Criterion from residual sum of squares."""
    return k * math.log(n) - 2 * _gaussian_loglik(rss, n)


@dataclass(frozen=True, slots=True)
class ModelFit:
    """A fitted decay model with its information criteria."""

    name: str
    params: dict[str, float]
    rss: float
    aic: float
    bic: float


def _linear_logspace_rss(
    x: NDArray[np.float64], log_s: NDArray[np.float64]
) -> tuple[float, float, float]:
    # least squares log_s = intercept + slope * x
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, log_s, rcond=None)
    resid = log_s - design @ beta
    return float(beta[0]), float(beta[1]), float(np.sum(resid**2))


def compare_decay_models(
    distances: Sequence[float], similarities: Sequence[float]
) -> dict[str, ModelFit]:
    """Fit and rank geometric / power-law / stretched-exp decay laws by AIC/BIC."""
    i = np.asarray(distances, dtype=np.float64)
    s = np.asarray(similarities, dtype=np.float64)
    if i.shape != s.shape or i.size < 4:
        raise ValueError("need at least four aligned (distance, similarity) points")
    if np.any(s <= 0):
        raise ValueError("similarities must be strictly positive (log-space fit)")
    log_s = np.log(s)
    n = int(i.size)
    fits: dict[str, ModelFit] = {}

    # Parameter counts include the Gaussian variance sigma^2 (k = n_coef + 1), the
    # standard convention for likelihood-based AIC/BIC on a regression fit.
    # geometric: log s = log s0 + i log rho  (2 coefficients + sigma^2)
    b0, b1, rss = _linear_logspace_rss(i, log_s)
    fits["geometric"] = ModelFit(
        "geometric", {"s0": math.exp(b0), "rho": math.exp(b1)}, rss, aic(rss, n, 3), bic(rss, n, 3)
    )

    # power-law: log s = log s0 - alpha log(i+1)  (2 coefficients + sigma^2)
    b0p, b1p, rssp = _linear_logspace_rss(np.log(i + 1.0), log_s)
    fits["power_law"] = ModelFit(
        "power_law", {"s0": math.exp(b0p), "alpha": -b1p}, rssp, aic(rssp, n, 3), bic(rssp, n, 3)
    )

    # stretched-exponential: log s = log s0 - (lambda i)^beta  (3 coefficients + sigma^2)
    def model(
        x: NDArray[np.float64], log_s0: float, lam: float, beta: float
    ) -> NDArray[np.float64]:
        return np.asarray(log_s0 - (np.abs(lam) * x) ** np.abs(beta), dtype=np.float64)

    try:
        popt, _ = optimize.curve_fit(model, i, log_s, p0=[float(log_s[0]), 0.5, 1.0], maxfev=10000)
        rss_s = float(np.sum((log_s - model(i, *popt)) ** 2))
        fits["stretched_exp"] = ModelFit(
            "stretched_exp",
            {"s0": math.exp(popt[0]), "lambda": abs(popt[1]), "beta": abs(popt[2])},
            rss_s,
            aic(rss_s, n, 4),
            bic(rss_s, n, 4),
        )
    except (RuntimeError, ValueError):  # pragma: no cover - fit failure path
        fits["stretched_exp"] = ModelFit("stretched_exp", {}, math.inf, math.inf, math.inf)

    return fits
