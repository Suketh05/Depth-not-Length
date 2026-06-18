"""Scored-result aggregation and the benchmark's output tables.

Turns the runner's raw :class:`~membench.agents.runner.AttemptRecord` rows into
scored rows (joining compliance / retrieval / correctness), then aggregates them
into the headline tables -- all plain group-by-means over one row shape, no bespoke
per-table logic:

* **headline** -- model fixed (Claude), rows = arms, per dataset (compliance).
* **depth crossover** -- rows = arms, cols = depth, cells = full-chain recovery,
  on the spec-strippable datasets.
* **robustness** -- arm fixed (the graph arm), rows = models, per dataset.
* **ablation** -- the four-arm ablation cells.
* **competitor matrix** -- measured (Tier-1) arms in one block; the cited
  vendor-reported (Tier-2) claims kept in a separate, labelled block (the
  ``assert_single_tier`` guard forbids mixing them).
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

from membench.agents.runner import AttemptRecord
from membench.competitors import ProvenanceTier, VendorClaim, assert_single_tier
from membench.metrics.compliance import score_compliance
from membench.metrics.correctness import score_correctness
from membench.metrics.retrieval import score_retrieval
from membench.types import Task

_K = TypeVar("_K")

__all__ = [
    "ABLATION_CELLS",
    "ABLATION_DATASETS",
    "ScoredRow",
    "TwoTierMatrix",
    "ablation_table",
    "competitor_matrix",
    "depth_crossover_table",
    "headline_table",
    "robustness_table",
    "score_records",
]

_CLAUDE_PREFIX = "claude"
_STRIPPABLE = ("dcbench", "swebench", "synthetic")
# The four-arm ablation is an *ecological* test on the real spec-stripped coding
# datasets only -- never synthetic. Synthetic is the controlled depth-crossover test;
# blending its flat, near-total separation into the ablation would overstate recovery.
ABLATION_DATASETS = ("dcbench", "swebench")
# The ablation's structured arm is brief_graph_3hop: plain brief_graph is 2-hop and
# cannot reach a depth-3 chain, so an ("brief_graph", 3) cell would render Brief as
# failing to recover -- a latent bug, not a measurement.
ABLATION_CELLS: dict[tuple[str, int], str] = {
    ("none", 1): "full_spec/none",
    ("none", 3): "stripped/none",
    ("brief_graph_3hop", 3): "stripped/brief",
    ("random_context", 3): "stripped/random",
}


@dataclass(frozen=True, slots=True)
class ScoredRow:
    """One attempt with its scored metrics -- the canonical analysis row."""

    dataset: str
    arm: str
    model: str
    depth: int
    spec_variant: str
    compliance_rate: float
    chain_recovered: bool
    recall: float
    precision: float
    correct: bool
    total_tokens: int
    dollars: float


def score_records(
    records: Iterable[AttemptRecord], tasks_by_id: Mapping[str, Task]
) -> list[ScoredRow]:
    """Join compliance / retrieval / correctness onto raw attempt records."""
    rows: list[ScoredRow] = []
    for rec in records:
        task = tasks_by_id[rec.task_id]
        compliance = score_compliance(rec.response_text, rec.governing_decisions, task.corpus_by_id)
        retrieval = score_retrieval(rec.retrieved_ids, rec.governing_decisions)
        if task.is_coding:
            correct = score_correctness(rec.response_text, compliance).merge_ready
        else:
            correct = compliance.rate == 1.0
        rows.append(
            ScoredRow(
                dataset=rec.dataset,
                arm=rec.arm,
                model=rec.model,
                depth=rec.depth,
                spec_variant=rec.spec_variant,
                compliance_rate=compliance.rate,
                chain_recovered=retrieval.chain_recovered,
                recall=retrieval.recall,
                precision=retrieval.precision,
                correct=correct,
                total_tokens=rec.input_tokens + rec.output_tokens,
                dollars=rec.dollars,
            )
        )
    return rows


def _grouped_mean(pairs: Iterable[tuple[_K, float]]) -> dict[_K, float]:
    groups: dict[_K, list[float]] = {}
    for key, value in pairs:
        groups.setdefault(key, []).append(value)
    return {key: statistics.mean(values) for key, values in groups.items()}


def headline_table(rows: Sequence[ScoredRow]) -> dict[tuple[str, str], float]:
    """Mean compliance per (dataset, arm) for the Claude model line."""
    return _grouped_mean(
        ((r.dataset, r.arm), r.compliance_rate) for r in rows if r.model.startswith(_CLAUDE_PREFIX)
    )


def depth_crossover_table(rows: Sequence[ScoredRow]) -> dict[tuple[str, int], float]:
    """Mean full-chain recovery per (arm, depth) on the spec-strippable datasets."""
    return _grouped_mean(
        ((r.arm, r.depth), float(r.chain_recovered))
        for r in rows
        if r.dataset in _STRIPPABLE and r.model.startswith(_CLAUDE_PREFIX)
    )


def robustness_table(
    rows: Sequence[ScoredRow], graph_arm: str = "brief_graph_3hop"
) -> dict[tuple[str, str], float]:
    """Mean compliance per (dataset, model) with the memory arm fixed to the graph arm."""
    return _grouped_mean(
        ((r.dataset, r.model), r.compliance_rate) for r in rows if r.arm == graph_arm
    )


def ablation_table(
    rows: Sequence[ScoredRow],
    cells: Mapping[tuple[str, int], str] | None = None,
) -> dict[str, float]:
    """Relabel specific (arm, depth) cells into the four-arm ablation.

    Restricted to the ecological datasets (dcbench + swebench); synthetic is excluded
    so the controlled-crossover separation is never conflated with measured recovery.
    """
    cells = cells or ABLATION_CELLS
    grouped: dict[str, list[float]] = {}
    for r in rows:
        if r.dataset not in ABLATION_DATASETS or not r.model.startswith(_CLAUDE_PREFIX):
            continue
        label = cells.get((r.arm, r.depth))
        if label is not None:
            grouped.setdefault(label, []).append(r.compliance_rate)
    return {label: statistics.mean(values) for label, values in grouped.items()}


@dataclass(frozen=True, slots=True)
class TwoTierMatrix:
    """Measured (Tier-1) and vendor-reported (Tier-2) results, kept separate."""

    measured: dict[str, float]  # arm -> measured compliance
    measured_tier: ProvenanceTier
    vendor_reported: list[VendorClaim]
    vendor_tier: ProvenanceTier


def competitor_matrix(
    rows: Sequence[ScoredRow], vendor_claims: Sequence[VendorClaim]
) -> TwoTierMatrix:
    """Build the two-tier competitor matrix; the tiers are never merged.

    Measured arms (one block) and vendor-reported claims (a separate, cited block)
    are returned distinctly; :func:`assert_single_tier` enforces that each block is a
    single provenance tier.
    """
    measured_by_arm = _grouped_mean((r.arm, r.compliance_rate) for r in rows)
    # Guard: the measured block is uniformly Tier-1 and vendor block uniformly Tier-2.
    measured_tier = assert_single_tier([ProvenanceTier.MEASURED] * max(len(measured_by_arm), 1))
    vendor_tier = (
        assert_single_tier([c.tier for c in vendor_claims])
        if vendor_claims
        else (ProvenanceTier.VENDOR_REPORTED)
    )
    return TwoTierMatrix(
        measured=measured_by_arm,
        measured_tier=measured_tier,
        vendor_reported=list(vendor_claims),
        vendor_tier=vendor_tier,
    )
