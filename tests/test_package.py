"""Smoke tests: the package imports and its subpackages are wired."""

from __future__ import annotations

import importlib

import membench


def test_version_is_exposed() -> None:
    assert membench.__version__ == "0.1.0"


def test_all_subpackages_import() -> None:
    for sub in (
        "theory",
        "retrieval",
        "agents",
        "datasets",
        "metrics",
        "stats",
        "analysis",
        "config",
    ):
        module = importlib.import_module(f"membench.{sub}")
        assert module.__doc__, f"membench.{sub} is missing a module docstring"
