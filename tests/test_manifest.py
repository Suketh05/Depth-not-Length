"""Tests for the reproducibility manifest."""

from __future__ import annotations

import json
from pathlib import Path

from membench.manifest import capture_manifest, save_manifest


class TestCapture:
    def test_captures_core_provenance(self) -> None:
        m = capture_manifest(config={"datasets": ["synthetic"]}, created_at="2026-06-18T00:00:00Z")
        assert m.membench_version == "0.1.0"
        assert m.python_version
        assert m.device.kind in {"cpu", "metal", "cuda"}
        assert m.config == {"datasets": ["synthetic"]}
        assert m.created_at == "2026-06-18T00:00:00Z"

    def test_tracks_key_package_versions(self) -> None:
        m = capture_manifest()
        assert "numpy" in m.package_versions and "scipy" in m.package_versions

    def test_round_trips_to_json(self, tmp_path: Path) -> None:
        m = capture_manifest(config={"seed": 0})
        path = save_manifest(m, tmp_path / "manifest.json")
        loaded = json.loads(path.read_text())
        assert loaded["membench_version"] == "0.1.0"
        assert loaded["device"]["kind"] in {"cpu", "metal", "cuda"}
        assert loaded["config"] == {"seed": 0}

    def test_pure_without_timestamp(self) -> None:
        # No clock read inside: created_at defaults to None unless supplied.
        assert capture_manifest().created_at is None
