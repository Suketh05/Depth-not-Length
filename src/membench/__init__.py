"""membench: a depth-stratified benchmark for agent memory.

This package measures whether *structured* memory (typed records joined by
explicit links, as in Brief's context graph) recovers deep causal context that
*similarity*-based retrieval cannot, and quantifies the depth at which the two
regimes cross over on both accuracy and token cost.

The public surface is intentionally small; see :mod:`membench.types` for the two
core contracts (``Task`` and ``MemorySystem``) that every dataset loader and
memory arm conforms to.
"""

from __future__ import annotations

__version__ = "0.1.0"
