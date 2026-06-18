"""Tests for the cluster-robust logistic compliance model."""

from __future__ import annotations

import numpy as np
import pytest

from membench.stats.regression import fit_compliance_on_arm_depth


def _synthetic_dataset() -> tuple[list[int], list[int], list[int], list[str]]:
    """Structured advantage that GROWS with depth (the depth thesis)."""
    rng = np.random.default_rng(0)
    comply: list[int] = []
    structured: list[int] = []
    depth: list[int] = []
    task_ids: list[str] = []
    for d in (1, 2, 3):
        for i in range(40):
            # similarity compliance decays with depth; structured stays high
            p_sim = max(0.05, 0.9 - 0.3 * d)
            p_str = 0.95
            for arm, p in ((0, p_sim), (1, p_str)):
                comply.append(int(rng.random() < p))
                structured.append(arm)
                depth.append(d)
                task_ids.append(f"d{d}-t{i}")
    return comply, structured, depth, task_ids


class TestFit:
    def test_structured_advantage_grows_with_depth(self) -> None:
        fit = fit_compliance_on_arm_depth(*_synthetic_dataset())
        # the interaction (advantage grows with depth) is positive and significant
        assert fit.depth_interaction > 0
        assert fit.interaction_p < 0.05
        assert fit.n_clusters > 1

    def test_reports_obs_and_clusters(self) -> None:
        fit = fit_compliance_on_arm_depth(*_synthetic_dataset())
        assert fit.n_obs == 3 * 40 * 2
        assert "structured" in fit.params and "structured:depth" in fit.params

    def test_rejects_constant_outcome(self) -> None:
        with pytest.raises(ValueError, match="no variation"):
            fit_compliance_on_arm_depth([1, 1, 1], [0, 1, 0], [1, 2, 3], ["a", "b", "c"])

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            fit_compliance_on_arm_depth([], [], [], [])
