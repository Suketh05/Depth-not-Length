"""Provenance tiers and the cited vendor-reported claims registry.

The benchmark separates two tiers of numbers and never mixes them:

* **Tier 1 — measured.** Produced by this harness, with every arm on the same
  tasks, budget, and model. These are the load-bearing, apples-to-apples results.
* **Tier 2 — vendor-reported.** Numbers a competitor (or its paper) published under
  *its own* conditions. Useful context, but not comparable to Tier 1, so they live
  in clearly labelled separate columns with a citation and the original conditions.

This module loads the Tier-2 registry (``competitors/registry.yaml``), validates
it, and provides the guard the analysis layer uses to refuse to merge the two tiers
into a single comparison.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

__all__ = [
    "MixedTierError",
    "ProvenanceTier",
    "VendorClaim",
    "assert_single_tier",
    "load_vendor_claims",
    "verified_claims",
]

_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "competitors" / "registry.yaml"


class ProvenanceTier(str, Enum):
    """How a reported number was produced."""

    MEASURED = "measured"
    VENDOR_REPORTED = "vendor_reported"


class VendorClaim(BaseModel):
    """One vendor-reported (Tier-2) claim with its provenance."""

    system: str
    category: str
    metric: str
    value: float | None = None
    baseline_value: float | None = None
    unit: str
    source_url: str
    conditions: str
    accessed: str
    needs_citation: bool = True

    @property
    def tier(self) -> ProvenanceTier:
        """Vendor claims are always Tier-2 by construction."""
        return ProvenanceTier.VENDOR_REPORTED


class _Registry(BaseModel):
    claims: list[VendorClaim] = Field(default_factory=list)


class MixedTierError(ValueError):
    """Raised when measured and vendor-reported numbers would be mixed in one view."""


def load_vendor_claims(path: Path | None = None) -> list[VendorClaim]:
    """Load and validate the vendor-reported claims registry."""
    registry_path = path or _REGISTRY_PATH
    raw = yaml.safe_load(registry_path.read_text()) or {}
    return _Registry.model_validate(raw).claims


def verified_claims(claims: Iterable[VendorClaim]) -> list[VendorClaim]:
    """Return only claims with a verified value (``needs_citation`` is false)."""
    return [c for c in claims if not c.needs_citation and c.value is not None]


def assert_single_tier(tiers: Iterable[ProvenanceTier]) -> ProvenanceTier:
    """Return the single tier present, or raise if measured and vendor numbers mix.

    The analysis layer calls this before rendering any comparison so a measured
    number and a vendor-reported number can never share a column.
    """
    distinct = set(tiers)
    if not distinct:
        raise ValueError("no tiers provided")
    if len(distinct) > 1:
        raise MixedTierError(
            "refusing to mix measured and vendor-reported numbers in one comparison; "
            "keep them in separate labelled columns"
        )
    return distinct.pop()
