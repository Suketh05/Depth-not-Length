r"""Logistic model of compliance with task-cluster-robust standard errors.

The benchmark's outcomes are binary (complied or not) and clustered (every arm runs
on the same tasks), so the proper model is logistic regression with standard errors
that account for the task clustering. Fitting

.. math::

    \operatorname{logit} \Pr(\text{comply}) = \beta_0 + \beta_1\,\text{structured}
        + \beta_2\,\text{depth} + \beta_3\,(\text{structured}\times\text{depth})

answers two questions at once: :math:`\beta_1` is the structured-memory advantage,
and the interaction :math:`\beta_3` is the headline -- *does that advantage grow with
depth?* A positive, significant :math:`\beta_3` is the depth thesis in one
coefficient. Standard errors are cluster-robust by ``task_id``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
import statsmodels.formula.api as smf

__all__ = ["LogisticFit", "fit_compliance_on_arm_depth"]


@dataclass(frozen=True, slots=True)
class LogisticFit:
    """Fitted logistic coefficients and cluster-robust p-values."""

    params: dict[str, float]
    pvalues: dict[str, float]
    n_obs: int
    n_clusters: int

    @property
    def structured_advantage(self) -> float:
        """Log-odds advantage of the structured arm at depth 0 (the ``structured`` term)."""
        return self.params["structured"]

    @property
    def depth_interaction(self) -> float:
        """Coefficient on structured*depth: how the advantage grows per hop of depth."""
        return self.params["structured:depth"]

    @property
    def interaction_p(self) -> float:
        """P-value for the structured*depth interaction (the depth-thesis test)."""
        return self.pvalues["structured:depth"]


def fit_compliance_on_arm_depth(
    compliance: Sequence[int],
    structured: Sequence[int],
    depth: Sequence[int],
    task_ids: Sequence[str],
) -> LogisticFit:
    """Fit logit(comply) ~ structured*depth with task-cluster-robust SEs.

    Parameters
    ----------
    compliance
        Binary outcome per observation (1 = complied).
    structured
        1 if the structured (graph) arm, 0 if a similarity arm.
    depth
        Causal-chain depth of the task.
    task_ids
        Cluster id per observation (standard errors are robust to within-task
        correlation).
    """
    frame = pd.DataFrame(
        {
            "comply": list(compliance),
            "structured": list(structured),
            "depth": list(depth),
            "task_id": list(task_ids),
        }
    )
    if frame.empty:
        raise ValueError("need at least one observation")
    if frame["comply"].nunique() < 2:
        raise ValueError("outcome has no variation; cannot fit a logistic model")

    model = smf.logit("comply ~ structured * depth", data=frame)
    result = model.fit(disp=False, cov_type="cluster", cov_kwds={"groups": frame["task_id"]})
    # statsmodels names the interaction term "structured:depth".
    params = {str(k): float(v) for k, v in result.params.items()}
    pvalues = {str(k): float(v) for k, v in result.pvalues.items()}
    return LogisticFit(
        params=params,
        pvalues=pvalues,
        n_obs=int(frame.shape[0]),
        n_clusters=int(frame["task_id"].nunique()),
    )
