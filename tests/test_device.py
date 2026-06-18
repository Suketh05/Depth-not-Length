"""Tests for compute-device detection (run provenance)."""

from __future__ import annotations

import pytest

from membench.device import DeviceInfo, detect_device


def test_returns_a_known_device_kind() -> None:
    info = detect_device()
    assert isinstance(info, DeviceInfo)
    assert info.kind in {"cpu", "metal", "cuda"}
    assert info.name and info.detail


def test_cuda_detected_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert detect_device().kind == "cuda"


def test_is_deterministic_in_a_fixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert detect_device() == detect_device()
