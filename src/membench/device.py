"""Compute-device detection for honest run provenance.

The benchmark's reported numbers come from the deterministic CPU/NumPy reference
pipeline; native kernels (C/Rust/CUDA) are optional accelerators for scale. To keep
hardware claims honest, every run records the device it actually ran on via
:func:`detect_device` -- and that device is whatever this machine is (CPU, Apple
Metal, or an NVIDIA CUDA GPU), never an asserted one.
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass

__all__ = ["DeviceInfo", "detect_device"]


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """The compute device a run executed on."""

    kind: str  # "cpu" | "metal" | "cuda"
    name: str
    detail: str


def detect_device() -> DeviceInfo:
    """Detect the available compute device for provenance (best-effort, no side effects).

    Prefers a visible NVIDIA CUDA GPU, then Apple Metal (Apple-Silicon macOS), else
    CPU. This only *labels* the environment; it never changes where work runs.
    """
    if shutil.which("nvidia-smi") is not None or os.environ.get("CUDA_VISIBLE_DEVICES"):
        return DeviceInfo(kind="cuda", name="nvidia-gpu", detail="detected via nvidia-smi/env")
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return DeviceInfo(kind="metal", name="apple-silicon", detail=platform.platform())
    return DeviceInfo(kind="cpu", name=platform.machine() or "cpu", detail=platform.platform())
