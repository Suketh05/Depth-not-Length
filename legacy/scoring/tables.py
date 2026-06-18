"""The four output tables (CLAUDE.md), each a groupby over results/*.jsonl.
No bespoke per-table scoring logic -- every function below is filter, then
groupby-mean, against the exact row shape agent/runner.py already writes.
Nothing here re-scores anything; it only aggregates rows that already carry
compliance_rate, dataset, arm, model, depth, and spec_variant.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

_RESULTS_DIR = Path(__file__).parent.parent / "results"

# response.model strings look like "claude-sonnet-4-6"; this is how rows are
# attributed to the "claude" line of the two-line grid without needing a
# separate model-name field on the row.
_CLAUDE_MODEL_PREFIX = "claude"

# The ablation's "stripped" condition uses depth=3 (CLAUDE.md doesn't pin an
# exact depth for it; depth=3 is the sharpest test of the recovery claim --
# see configs/depth_crossover.json, which is what actually produces these
# rows). Not a separate run: this is a relabeled filter over the same
# depth-crossover data depth_crossover_table() also reads.
_ABLATION_CELLS = {
    ("none", 1): "full_spec/none",
    ("none", 3): "stripped/none",
    ("brief", 3): "stripped/brief",
    ("random_context", 3): "stripped/random_context",
}


def load_rows(results_dir: Path = _RESULTS_DIR) -> list[dict]:
    rows = []
    for path in sorted(Path(results_dir).glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _mean(values: list[float]):
    return statistics.mean(values) if values else None


def _groupby_mean(rows: list[dict], key_fields: list[str], value_field: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in key_fields)
        groups[key].append(row[value_field])
    return {key: _mean(vals) for key, vals in groups.items()}


def _is_native_condition(row: dict) -> bool:
    """longmemeval has no stripped condition, so every row qualifies; for
    dcbench/swebench, "native" means the unstripped d=1 control."""
    return row["dataset"] == "longmemeval" or (row["depth"] == 1 and row["spec_variant"] == "full")


def headline_table(rows: list[dict]) -> dict:
    """Model fixed (Claude), rows = memory arms, per dataset."""
    filtered = [
        r for r in rows if r["model"].startswith(_CLAUDE_MODEL_PREFIX) and _is_native_condition(r)
    ]
    return _groupby_mean(filtered, ["dataset", "arm"], "compliance_rate")


def model_robustness_table(rows: list[dict]) -> dict:
    """Brief fixed, rows = models, per dataset."""
    filtered = [r for r in rows if r["arm"] == "brief" and _is_native_condition(r)]
    return _groupby_mean(filtered, ["dataset", "model"], "compliance_rate")


def depth_crossover_table(rows: list[dict]) -> dict:
    """rows = arms, cols = d=1/d=2/d=3+, cells = compliance. dcbench +
    swebench only, model fixed to Claude -- part of the model-fixed line,
    not a third sweep dimension."""
    filtered = [
        r
        for r in rows
        if r["dataset"] in ("dcbench", "swebench") and r["model"].startswith(_CLAUDE_MODEL_PREFIX)
    ]
    return _groupby_mean(filtered, ["arm", "depth"], "compliance_rate")


def ablation_table(rows: list[dict]) -> dict:
    """The 4-arm ablation, dcbench + swebench only -- a relabeled slice of
    the same data depth_crossover_table() reads, not a separate experiment."""
    filtered = [
        r
        for r in rows
        if r["dataset"] in ("dcbench", "swebench")
        and r["model"].startswith(_CLAUDE_MODEL_PREFIX)
        and (r["arm"], r["depth"]) in _ABLATION_CELLS
    ]
    raw = _groupby_mean(filtered, ["arm", "depth"], "compliance_rate")
    return {_ABLATION_CELLS[key]: value for key, value in raw.items()}


def _print_table(name: str, table: dict) -> None:
    print(f"\n{name}")
    for key, value in sorted(table.items(), key=lambda kv: str(kv[0])):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    rows = load_rows()
    print(f"loaded {len(rows)} rows from {_RESULTS_DIR}/*.jsonl")
    _print_table("headline (model=claude, rows=arm, per dataset)", headline_table(rows))
    _print_table("model_robustness (arm=brief, rows=model, per dataset)", model_robustness_table(rows))
    _print_table("depth_crossover (rows=arm, cols=depth, dcbench+swebench)", depth_crossover_table(rows))
    _print_table("ablation (dcbench+swebench only)", ablation_table(rows))
