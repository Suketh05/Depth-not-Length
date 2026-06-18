"""Tests for the provenance tiers and the vendor-reported claims registry."""

from __future__ import annotations

import pytest

from membench.competitors import (
    MixedTierError,
    ProvenanceTier,
    VendorClaim,
    assert_single_tier,
    load_vendor_claims,
    verified_claims,
)


class TestRegistryLoads:
    def test_loads_and_validates(self) -> None:
        claims = load_vendor_claims()
        assert claims, "registry should not be empty"
        assert all(isinstance(c, VendorClaim) for c in claims)
        assert all(c.tier is ProvenanceTier.VENDOR_REPORTED for c in claims)

    def test_every_claim_has_a_source(self) -> None:
        for claim in load_vendor_claims():
            assert claim.source_url.startswith("http")
            assert claim.conditions

    def test_brief_published_numbers_are_verified_and_present(self) -> None:
        verified = verified_claims(load_vendor_claims())
        briefs = {c.metric: c for c in verified if c.system == "Brief"}
        assert briefs["agent_decision_compliance"].value == pytest.approx(0.95)
        assert briefs["agent_decision_compliance"].baseline_value == pytest.approx(0.46)

    def test_unverified_competitor_numbers_excluded_until_cited(self) -> None:
        # Competitor entries are placeholders pending a verified primary source.
        verified = verified_claims(load_vendor_claims())
        assert not any(c.system in {"Mem0", "Zep", "GraphRAG"} for c in verified)


class TestTierSeparation:
    def test_single_tier_ok(self) -> None:
        assert (
            assert_single_tier([ProvenanceTier.MEASURED, ProvenanceTier.MEASURED])
            is ProvenanceTier.MEASURED
        )

    def test_mixed_tiers_rejected(self) -> None:
        with pytest.raises(MixedTierError, match="refusing to mix"):
            assert_single_tier([ProvenanceTier.MEASURED, ProvenanceTier.VENDOR_REPORTED])

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="no tiers"):
            assert_single_tier([])
