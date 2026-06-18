"""Expands configs/*.json into concrete run cells. This is the one piece of
code configs/ needs -- everything else in this directory is data. De-
duplicates cells that appear in more than one config file (e.g.
headline_robustness.json's dcbench/swebench memory_line cells are the same
cells as depth_crossover.json's depth=1 cells) so run.py never executes the
same (dataset, arm, model, spec_variant, depth) combination twice.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

_CONFIGS_DIR = Path(__file__).parent


def _load(name: str) -> dict:
    with open(_CONFIGS_DIR / name) as f:
        return json.load(f)


@dataclasses.dataclass(frozen=True)
class RunCell:
    dataset: str
    arm: str
    model_name: str  # configs/models.json key: "claude" | "gpt" | "open_weight"
    model_id: str
    model_backend: str
    model_implemented: bool
    spec_variant: str
    depth: int
    budget_tokens: int
    code_search_tool: str
    retry_policy: str


def load_run_cells() -> list[RunCell]:
    """Returns every unique run cell implied by configs/headline_robustness.json
    and configs/depth_crossover.json, with budget_tokens/code_search_tool/
    retry_policy injected identically from configs/common.json -- the
    fairness lock. For longmemeval, spec_variant/depth are inert: its loader
    reads depth off the dataset's own structure per task, not per cell."""
    common = _load("common.json")
    models = _load("models.json")
    headline = _load("headline_robustness.json")
    depth_crossover = _load("depth_crossover.json")

    budgets = common["budget_tokens_by_dataset"]
    code_search_tool = common["code_search_tool"]
    retry_policy = common["retry_policy"]

    def make(dataset: str, arm: str, model_name: str, spec_variant: str, depth: int) -> RunCell:
        model = models[model_name]
        return RunCell(
            dataset=dataset,
            arm=arm,
            model_name=model_name,
            model_id=model["model_id"],
            model_backend=model["backend"],
            model_implemented=model["implemented"],
            spec_variant=spec_variant,
            depth=depth,
            budget_tokens=budgets[dataset],
            code_search_tool=code_search_tool,
            retry_policy=retry_policy,
        )

    cells: dict[tuple, RunCell] = {}

    def add(cell: RunCell) -> None:
        key = (cell.dataset, cell.arm, cell.model_name, cell.spec_variant, cell.depth)
        cells[key] = cell

    cond = headline["condition"]
    for dataset in headline["datasets"]:
        if dataset == "longmemeval":
            spec_variant, depth = "full", 1  # inert placeholders, see docstring
        else:
            spec_variant = cond["dcbench_swebench_spec_variant"]
            depth = cond["dcbench_swebench_depth"]
        for arm in headline["memory_line"]["arms"]:
            add(make(dataset, arm, headline["memory_line"]["model"], spec_variant, depth))
        for model_name in headline["model_line"]["models"]:
            add(make(dataset, headline["model_line"]["arm"], model_name, spec_variant, depth))

    for dataset in depth_crossover["datasets"]:
        for arm in depth_crossover["arms"]:
            for depth_str, spec_variant in depth_crossover["spec_variants_by_depth"].items():
                add(make(dataset, arm, depth_crossover["model"], spec_variant, int(depth_str)))

    return list(cells.values())
