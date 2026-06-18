"""Shared tokenisation for the lexical retrieval arms (BM25, TF-IDF)."""

from __future__ import annotations

import re

__all__ = ["tokenize"]

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Split text into lowercased alphanumeric word tokens."""
    return _TOKEN_RE.findall(text.lower())
