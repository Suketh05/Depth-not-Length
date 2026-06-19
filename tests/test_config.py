"""Tests for the benchmark config schema and the fairness lock."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from membench.config.schema import BenchmarkConfig, load_config


def _valid() -> dict[str, object]:
    return {
        "datasets": ["synthetic"],
        "arms": ["none", "brief_graph_3hop"],
        "budget_tokens_by_dataset": {"synthetic": 150},
    }


class TestSchema:
    def test_loads_default_config(self) -> None:
        cfg = load_config(Path("configs/default.yaml"))
        assert "brief_graph_3hop" in cfg.arms
        assert cfg.budget_for("dcbench") == 250
        assert cfg.offline is True

    def test_minimal_valid(self) -> None:
        cfg = BenchmarkConfig.model_validate(_valid())
        assert cfg.models == ["claude"] and cfg.seed == 0

    def test_missing_budget_for_dataset_rejected(self) -> None:
        bad = _valid() | {"datasets": ["synthetic", "dcbench"]}
        with pytest.raises(ValidationError, match="missing entries"):
            BenchmarkConfig.model_validate(bad)

    def test_empty_arms_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkConfig.model_validate(_valid() | {"arms": []})

    def test_bad_retry_policy_rejected(self) -> None:
        with pytest.raises(ValidationError, match="retry_policy"):
            BenchmarkConfig.model_validate(_valid() | {"retry_policy": "bogus"})

    def test_negative_budget_rejected(self) -> None:
        with pytest.raises(ValidationError, match="non-negative"):
            BenchmarkConfig.model_validate(
                _valid() | {"budget_tokens_by_dataset": {"synthetic": -1}}
            )

    def test_fairness_lock_is_per_dataset(self) -> None:
        # The schema has no per-arm budget field at all -- the fairness lock is structural.
        assert "arm" not in str(BenchmarkConfig.model_fields["budget_tokens_by_dataset"])
