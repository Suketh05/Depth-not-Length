"""Typed run configuration with the fairness lock validated at load time.

A run is fully described by which datasets, arms, and models to sweep, the per-dataset
token budget, and the seed. This module makes that a validated pydantic schema rather
than scattered arguments, and enforces the *fairness lock* structurally: the budget is
keyed by dataset (never by arm), so every arm provably sees the same budget for a given
dataset, and the validator rejects a config that omits a budget for a swept dataset or
leaves arms/models empty. Configuration is data; this module only loads and validates
it.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

__all__ = ["BenchmarkConfig", "load_config"]

_RETRY_POLICIES = ("single_shot", "retry_until_correct")


class BenchmarkConfig(BaseModel):
    """A validated benchmark run specification."""

    datasets: list[str] = Field(min_length=1)
    arms: list[str] = Field(min_length=1)
    models: list[str] = Field(default_factory=lambda: ["claude"], min_length=1)
    budget_tokens_by_dataset: dict[str, int]
    seed: int = 0
    per_depth: int = Field(default=10, gt=0)
    retry_policy: str = "single_shot"
    offline: bool = True

    @model_validator(mode="after")
    def _check_fairness_lock(self) -> BenchmarkConfig:
        if self.retry_policy not in _RETRY_POLICIES:
            raise ValueError(f"retry_policy must be one of {_RETRY_POLICIES}")
        missing = [d for d in self.datasets if d not in self.budget_tokens_by_dataset]
        if missing:
            raise ValueError(f"budget_tokens_by_dataset is missing entries for: {missing}")
        for dataset, budget in self.budget_tokens_by_dataset.items():
            if budget < 0:
                raise ValueError(f"budget for {dataset!r} must be non-negative")
        # The budget is per-dataset, never per-arm: this is the fairness lock by
        # construction -- every arm receives the same budget for a given dataset.
        return self

    def budget_for(self, dataset: str) -> int:
        """Return the shared token budget for a dataset (identical across arms)."""
        return self.budget_tokens_by_dataset[dataset]


def load_config(path: str | Path) -> BenchmarkConfig:
    """Load and validate a benchmark configuration from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return BenchmarkConfig.model_validate(raw)
