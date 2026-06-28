"""Tests for the P_comply = P_ret * kappa factorization scorer."""

from __future__ import annotations

import pytest

from membench.analysis.tables import ScoredRow
from membench.metrics.factorization import (
    Factorization,
    factorize_rates,
    factorize_rows,
    wilson_interval,
)


def _row(arm: str, depth: int, recall: float, compliance: float) -> ScoredRow:
    return ScoredRow(
        dataset="synthetic",
        arm=arm,
        model="claude",
        depth=depth,
        spec_variant="stripped",
        compliance_rate=compliance,
        chain_recovered=recall == 1.0,
        recall=recall,
        precision=recall,
        correct=compliance == 1.0,
        total_tokens=100,
        dollars=0.01,
    )


class TestFactorizeRates:
    def test_golden_kappa_and_reconstruction(self) -> None:
        # Hand-computed: recall=0.8, compliance=0.6
        #   kappa = compliance / recall = 0.6 / 0.8 = 0.75
        #   reconstructed P_comply = P_ret * kappa = 0.8 * 0.75 = 0.6
        result = factorize_rates([0.8], [0.6])
        assert result.kappa == pytest.approx(0.75)
        assert result.p_comply == pytest.approx(0.6)
        assert result.p_ret == pytest.approx(0.8)
        # Reconstruction identity: P_ret * kappa == observed compliance.
        assert result.p_comply == pytest.approx(result.p_comply_observed)

    def test_kappa_in_unit_interval_region(self) -> None:
        # compliance <= recall => use-factor is a genuine probability in [0, 1].
        result = factorize_rates([0.8, 0.5], [0.6, 0.1])
        assert result.kappa is not None
        assert 0.0 <= result.kappa <= 1.0

    def test_recall_zero_yields_undefined_kappa_no_crash(self) -> None:
        result = factorize_rates([0.0, 0.0], [0.0, 0.0])
        assert result.kappa is None
        assert result.kappa_ci is None
        assert result.p_ret == 0.0
        assert result.p_comply == 0.0  # product with zero retrieval

    def test_wilson_ci_golden_for_pooled_counts(self) -> None:
        # 10 rows, each recall=0.8 / compliance=0.6 => pooled successes=6, trials=8.
        # Wilson 95% (z=1.959964) on (6, 8), p_hat=0.75, hand-computed:
        #   center = (0.75 + z^2/16) / (1 + z^2/8) = 0.9900912 / 1.4801824 = 0.668898
        #   margin = z*sqrt(0.75*0.25/8 + z^2/256) / (1 + z^2/8) = 0.259612
        #   low = 0.409286, high = 0.928510
        rows_recall = [0.8] * 10
        rows_compliance = [0.6] * 10
        result = factorize_rates(rows_recall, rows_compliance)
        assert result.kappa == pytest.approx(0.75)
        assert result.kappa_ci is not None
        low, high = result.kappa_ci
        assert low == pytest.approx(0.409286, abs=1e-4)
        assert high == pytest.approx(0.928510, abs=1e-4)
        # Invariants: CI brackets the point estimate and stays inside [0, 1].
        assert 0.0 <= low <= result.kappa <= high <= 1.0

    def test_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            factorize_rates([0.5], [0.5, 0.5])
        with pytest.raises(ValueError, match="at least one"):
            factorize_rates([], [])
        with pytest.raises(ValueError, match="recall values"):
            factorize_rates([1.5], [0.5])
        with pytest.raises(ValueError, match="compliance values"):
            factorize_rates([0.5], [-0.1])


class TestWilsonInterval:
    def test_wilson_value(self) -> None:
        # Hand-computed Wilson 95% interval for 8 successes in 10 trials (p_hat=0.8,
        # z=1.959964): center=0.716737, margin=0.226578 => (0.490159, 0.943315).
        low, high = wilson_interval(8, 10)
        assert low == pytest.approx(0.490, abs=1e-3)
        assert high == pytest.approx(0.943, abs=1e-3)

    def test_bounds_clamped_to_unit_interval(self) -> None:
        low, high = wilson_interval(10, 10)
        assert low >= 0.0
        assert high == pytest.approx(1.0)

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="trials must be positive"):
            wilson_interval(0, 0)
        with pytest.raises(ValueError, match="non-negative"):
            wilson_interval(-1, 5)
        with pytest.raises(ValueError, match="confidence"):
            wilson_interval(1, 5, confidence=1.5)


class TestFactorizeRows:
    def test_groups_by_arm_and_depth(self) -> None:
        rows = [
            _row("brief_graph", 1, 0.8, 0.6),
            _row("brief_graph", 1, 0.8, 0.6),
            _row("none", 2, 0.0, 0.0),
        ]
        groups = factorize_rows(rows)
        assert [(g.arm, g.depth) for g in groups] == [("brief_graph", 1), ("none", 2)]

        brief = groups[0].factorization
        assert isinstance(brief, Factorization)
        assert brief.n == 2
        assert brief.kappa == pytest.approx(0.75)
        assert brief.p_comply == pytest.approx(0.6)

        none = groups[1].factorization
        assert none.kappa is None  # recall=0 => undefined, no crash
        assert none.kappa_ci is None
