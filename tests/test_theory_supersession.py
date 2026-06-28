"""Tests for the supersession-edge head-recovery model.

Golden values are hand-computed from the closed forms
``P_struct(n) = q**(n - 1)`` (series-system reliability over the ``n - 1``
supersession hops) and ``P_sim(n) = 1 / n`` (uniform pick among ``n``
near-identical versions), never by running the module under test.
"""

from __future__ import annotations

import math

import pytest

from membench.theory.supersession import (
    SupersessionModel,
    SupersessionRecovery,
    edge_hops,
    similarity_supersession_recovery,
    structured_supersession_recovery,
)


class TestEdgeHops:
    def test_chain_of_n_has_n_minus_one_edges(self) -> None:
        assert edge_hops(1) == 0
        assert edge_hops(2) == 1
        assert edge_hops(5) == 4

    def test_rejects_bad_length(self) -> None:
        with pytest.raises(ValueError, match="chain length"):
            edge_hops(0)


class TestStructuredRecovery:
    def test_golden_n2(self) -> None:
        # GOLDEN (hand-computed): n = 2, q = 0.95.
        # edge_hops = n - 1 = 1, so P_struct = q**1 = 0.95.
        assert structured_supersession_recovery(2, 0.95) == pytest.approx(0.95)

    def test_golden_n4(self) -> None:
        # GOLDEN (hand-computed): n = 4, q = 0.9.
        # edge_hops = 3, so P_struct = 0.9**3 = 0.729.
        assert structured_supersession_recovery(4, 0.9) == pytest.approx(0.729)

    def test_single_decision_is_certain(self) -> None:
        # n = 1: no supersession edge to traverse, head recovered with prob 1.
        assert structured_supersession_recovery(1, 0.7) == 1.0

    def test_matches_q_to_the_hops(self) -> None:
        for n in range(1, 7):
            assert structured_supersession_recovery(n, 0.85) == pytest.approx(0.85 ** (n - 1))

    def test_rejects_bad_q(self) -> None:
        with pytest.raises(ValueError, match="q must be"):
            structured_supersession_recovery(3, 0.0)

    def test_rejects_bad_length(self) -> None:
        with pytest.raises(ValueError, match="chain length"):
            structured_supersession_recovery(0, 0.9)


class TestSimilarityRecovery:
    def test_uniform_baseline(self) -> None:
        # GOLDEN: uniform pick among n indistinguishable versions = 1 / n.
        assert similarity_supersession_recovery(2) == pytest.approx(0.5)
        assert similarity_supersession_recovery(4) == pytest.approx(0.25)

    def test_fixed_baseline_overrides(self) -> None:
        # A measured band (e.g. 0.66) is independent of n when supplied.
        assert similarity_supersession_recovery(8, baseline=0.66) == pytest.approx(0.66)

    def test_rejects_bad_baseline(self) -> None:
        with pytest.raises(ValueError, match="baseline"):
            similarity_supersession_recovery(3, baseline=1.5)


class TestSupersessionModel:
    def test_golden_recovery_n2(self) -> None:
        # GOLDEN (hand-computed), n = 2, q = 0.95, uniform baseline:
        #   edge_hops      = 1
        #   p_struct       = 0.95**1 = 0.95
        #   p_sim          = 1 / 2   = 0.5
        #   advantage      = 0.95 - 0.5 = 0.45
        #   structured_wins = 0.95 >= 0.5 -> True
        model = SupersessionModel(q=0.95)
        rec = model.recovery(2)
        assert isinstance(rec, SupersessionRecovery)
        assert rec.n == 2
        assert rec.edge_hops == 1
        assert rec.q == 0.95
        assert rec.p_struct == pytest.approx(0.95)
        assert rec.p_sim == pytest.approx(0.5)
        assert rec.advantage == pytest.approx(0.45)
        assert rec.structured_wins is True

    def test_break_even_q_golden(self) -> None:
        # GOLDEN: break-even solves q**(n-1) = 1/n -> q = n**(-1/(n-1)).
        # n = 2: 2**(-1)      = 0.5
        # n = 4: 4**(-1/3)    = 0.6299605249...
        model = SupersessionModel(q=0.99)
        assert model.break_even_q(2) == pytest.approx(0.5)
        assert model.break_even_q(4) == pytest.approx(4.0 ** (-1.0 / 3.0))

    def test_break_even_at_q_makes_advantage_zero(self) -> None:
        # At q == break-even, structured exactly ties similarity.
        n = 5
        # break-even for uniform baseline = n**(-1/(n-1))
        q_be = n ** (-1.0 / (n - 1))
        model = SupersessionModel(q=q_be)
        # The crossover is a knife-edge: the gain is zero to floating-point.
        assert model.advantage(n) == pytest.approx(0.0, abs=1e-12)

    def test_single_decision_break_even_is_zero(self) -> None:
        assert SupersessionModel(q=0.5).break_even_q(1) == 0.0

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 10])
    def test_probabilities_in_unit_interval(self, n: int) -> None:
        model = SupersessionModel(q=0.8)
        assert 0.0 <= model.p_struct(n) <= 1.0
        assert 0.0 <= model.p_sim(n) <= 1.0

    def test_structured_dominates_for_high_q(self) -> None:
        # INVARIANT: for q comfortably above break-even, structured >= similarity
        # across the whole chain (and strictly so for n >= 2).
        model = SupersessionModel(q=0.97)
        for n in range(2, 12):
            assert model.p_struct(n) > model.p_sim(n)
            assert model.recovery(n).structured_wins is True

    def test_similarity_can_win_for_low_q(self) -> None:
        # Below break-even the similarity baseline is the better bet.
        model = SupersessionModel(q=0.3)
        n = 3  # break-even q = 3**(-1/2) ~ 0.577 > 0.3
        assert model.p_struct(n) < model.p_sim(n)
        assert model.recovery(n).structured_wins is False

    def test_advantage_matches_components(self) -> None:
        model = SupersessionModel(q=0.9)
        for n in range(1, 8):
            assert model.advantage(n) == pytest.approx(model.p_struct(n) - model.p_sim(n))

    def test_fixed_band_baseline_constant_across_n(self) -> None:
        model = SupersessionModel(q=0.92, similarity_baseline=0.66)
        assert model.p_sim(3) == pytest.approx(0.66)
        assert model.p_sim(9) == pytest.approx(0.66)

    def test_rejects_bad_q(self) -> None:
        with pytest.raises(ValueError, match="q must be"):
            SupersessionModel(q=1.5)

    def test_rejects_bad_baseline(self) -> None:
        with pytest.raises(ValueError, match="baseline"):
            SupersessionModel(q=0.9, similarity_baseline=-0.1)


def test_uniform_break_even_closed_form() -> None:
    # Cross-check the analytic break-even against an independent computation.
    for n in range(2, 9):
        q_be = n ** (-1.0 / (n - 1))
        assert SupersessionModel(q=0.9).break_even_q(n) == pytest.approx(q_be)
        # sanity: q_be**(n-1) == 1/n
        assert q_be ** (n - 1) == pytest.approx(1.0 / n)
        assert math.isfinite(q_be)
