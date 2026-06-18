"""Shared helpers for dataset loaders.

Every loader emits :class:`~membench.types.Task` objects and nothing else, so the
runner stays a flat loop. This module locates the vendored data and reads the
per-task depth labels (the certified maximum depth each task may be stripped to).
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["DATA_DIR", "max_depth_by_task_id"]

DATA_DIR = Path(__file__).parent / "data"
_DEPTH_LABELS_PATH = DATA_DIR / "depth_labels.jsonl"


def max_depth_by_task_id(dataset: str, path: Path | None = None) -> dict[str, int]:
    """Return ``{task_id: certified_max_depth}`` for a dataset from depth_labels.jsonl.

    A task may only be stripped to a depth at or below its certified maximum; the
    loaders drop tasks not certified to the requested depth rather than stripping
    past what was checked.
    """
    labels_path = path or _DEPTH_LABELS_PATH
    labels: dict[str, int] = {}
    with open(labels_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["dataset"] == dataset:
                labels[row["task_id"]] = int(row["max_depth"])
    return labels
