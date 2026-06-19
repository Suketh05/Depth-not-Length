"""Reproducibility manifest: capture exactly what produced a run.

A defensible benchmark records the conditions of every run so a number can be
reproduced and audited. :func:`capture_manifest` snapshots the package version, the
Python and platform identity, the compute device (honest provenance, never asserted),
the resolved config, the versions of the load-bearing dependencies, and the git commit
-- enough that another machine can reconstruct the run. Timestamps are passed in rather
than read from the clock, so capturing a manifest stays a pure function (and the test
suite reproducible).
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from membench import __version__
from membench.device import DeviceInfo, detect_device

__all__ = ["RunManifest", "capture_manifest", "save_manifest"]

_TRACKED_PACKAGES = ("numpy", "scipy", "scikit-learn", "pandas", "statsmodels", "pydantic")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


@dataclass(frozen=True, slots=True)
class RunManifest:
    """A snapshot of the conditions that produced a benchmark run."""

    membench_version: str
    python_version: str
    platform: str
    device: DeviceInfo
    config: dict[str, Any]
    package_versions: dict[str, str]
    git_commit: str | None
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of the manifest."""
        return asdict(self)


def capture_manifest(
    config: dict[str, Any] | None = None,
    *,
    created_at: str | None = None,
    device: DeviceInfo | None = None,
) -> RunManifest:
    """Capture the current run's reproducibility manifest.

    ``config`` is the resolved run configuration (e.g.
    ``BenchmarkConfig.model_dump()``); ``created_at`` is an optional caller-supplied
    timestamp (kept out of this function so it stays pure).
    """
    return RunManifest(
        membench_version=__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),
        device=device or detect_device(),
        config=config or {},
        package_versions=_package_versions(),
        git_commit=_git_commit(),
        created_at=created_at,
    )


def save_manifest(manifest: RunManifest, path: str | Path) -> Path:
    """Write a manifest to a JSON file (creating parent dirs)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return out
