"""Tests for the token-economics model."""

from __future__ import annotations

import itertools

import pytest

from membench.theory.decay import SimilarityDecayModel
from membench.theory.information import TokenEconomics, useful_signal
from membench.theory.recovery import RecoveryModel


def _econ(tokens_per_item: float = 50.0, q: float = 0.97) -> TokenEconomics:
    rec = RecoveryModel(decay=SimilarityDecayModel(s0=0.9, rho=0.5), q=q)
    return TokenEconomics(recovery=rec, tokens_per_item=tokens_per_item)


class TestUsefulSignal:
    def test_product(self) -> None:
        assert useful_signal(100.0, 0.4) == pytest.approx(40.0)

    @pytest.mark.parametrize(("tokens", "precision"), [(-1.0, 0.5), (10.0, 1.1), (10.0, -0.1)])
    def test_validation(self, tokens: float, precision: float) -> None:
        with pytest.raises(ValueError):
            useful_signal(tokens, precision)


class TestTokenEconomics:
    def test_rejects_nonpositive_tokens_per_item(self) -> None:
        rec = RecoveryModel(decay=SimilarityDecayModel(s0=0.9, rho=0.5), q=0.9)
        with pytest.raises(ValueError, match="tokens_per_item"):
            TokenEconomics(recovery=rec, tokens_per_item=0.0)

    def test_structured_tokens_linear_in_depth(self) -> None:
        econ = _econ(tokens_per_item=50.0)
        assert econ.structured_tokens(1) == pytest.approx(50.0)
        assert econ.structured_tokens(4) == pytest.approx(200.0)

    def test_similarity_tokens_superlinear(self) -> None:
        econ = _econ()
        # T_sim grows faster than linearly because 1/s_i = rho^-i explodes.
        per_depth = [econ.similarity_tokens(d) for d in range(1, 6)]
        increments = [b - a for a, b in itertools.pairwise(per_depth)]
        assert all(b > a for a, b in itertools.pairwise(increments))  # convex / accelerating

    def test_similarity_precision_decreases_with_depth(self) -> None:
        econ = _econ()
        precisions = [econ.similarity_precision(d) for d in range(1, 6)]
        assert all(b < a for a, b in itertools.pairwise(precisions))
        assert econ.structured_precision() == 1.0

    def test_token_cost_ratio_grows(self) -> None:
        econ = _econ()
        ratios = [econ.token_cost_ratio(d) for d in range(1, 6)]
        assert all(b > a for a, b in itertools.pairwise(ratios))
        assert ratios[0] == pytest.approx(1.0 / (0.9 * 0.5))  # d=1: (1/s_1)/1

    def test_cost_to_correct_ratio_diverges(self) -> None:
        econ = _econ()
        ratios = econ.cost_to_correct_ratio_curve(6)
        assert all(b > a for a, b in itertools.pairwise(ratios))
        assert ratios[-1] > ratios[0] * 10  # strong divergence by depth 6

    def test_similarity_useful_signal_equals_chain_payload_per_attempt(self) -> None:
        # U_sim = T_sim * precision = c * sum(1/s_i) * (d / sum(1/s_i)) = c*d,
        # i.e. equal *useful* payload per attempt; the difference is raw token cost.
        econ = _econ(tokens_per_item=50.0)
        for d in range(1, 5):
            assert econ.similarity_useful_signal(d) == pytest.approx(50.0 * d)
            assert econ.structured_useful_signal(d) == pytest.approx(50.0 * d)


class TestReturnOnTokens:
    def test_rot_is_reciprocal_of_cost_to_correct(self) -> None:
        econ = _econ()
        for d in range(1, 6):
            assert econ.structured_return_on_tokens(d) == pytest.approx(
                1.0 / econ.structured_cost_to_correct(d)
            )
            assert econ.similarity_return_on_tokens(d) == pytest.approx(
                1.0 / econ.similarity_cost_to_correct(d)
            )

    def test_rot_ratio_equals_cost_to_correct_ratio(self) -> None:
        # Same math, buyer-facing lens: "x times more shipped-right work per token".
        econ = _econ()
        for d in range(1, 6):
            assert econ.return_on_tokens_ratio(d) == pytest.approx(econ.cost_to_correct_ratio(d))

    def test_rot_ratio_favors_structure_and_grows_with_depth(self) -> None:
        econ = _econ()
        ratios = [econ.return_on_tokens_ratio(d) for d in range(1, 6)]
        assert ratios[0] > 1.0
        assert all(b > a for a, b in itertools.pairwise(ratios))

    def test_cost_per_correct_dollars(self) -> None:
        econ = _econ()
        d, price = 3, 5e-6
        assert econ.cost_per_correct_dollars(d, price, structured=True) == pytest.approx(
            econ.structured_cost_to_correct(d) * price
        )
        # Structured ships a merge-ready task for fewer dollars than similarity.
        assert econ.cost_per_correct_dollars(
            d, price, structured=True
        ) < econ.cost_per_correct_dollars(d, price, structured=False)

    def test_cost_per_correct_rejects_negative_price(self) -> None:
        with pytest.raises(ValueError, match="dollars_per_token"):
            _econ().cost_per_correct_dollars(2, -1.0, structured=True)
