"""Tests for the always-valid sequential (betting e-process) test."""

from __future__ import annotations

import numpy as np
import pytest

from membench.stats.sequential import (
    betting_capital_process,
    confidence_sequence,
    sequential_test,
)


class TestBettingCapitalProcess:
    def test_hand_computed_path(self) -> None:
        # GOLDEN (computed by hand, independently of the implementation):
        # bet lambda = 0.5, null mean m0 = 0.5, outcomes [1, 1, 0, 1].
        # Per-step factor f = 1 + lambda * (x - m0):
        #   x=1 -> 1 + 0.5*(1-0.5) = 1.25
        #   x=0 -> 1 + 0.5*(0-0.5) = 0.75
        # Capital is the running product:
        #   e_1 = 1.25
        #   e_2 = 1.25 * 1.25 = 1.5625
        #   e_3 = 1.5625 * 0.75 = 1.171875
        #   e_4 = 1.171875 * 1.25 = 1.46484375
        path = betting_capital_process([1, 1, 0, 1], 0.5, bet=0.5)
        assert path == pytest.approx([1.25, 1.5625, 1.171875, 1.46484375])

    def test_final_closed_form(self) -> None:
        # With m0 = 0.5 and lambda = 0.5 the final capital is exactly
        # 1.25**(#ones) * 0.75**(#zeros). Here 3 ones, 2 zeros:
        #   1.25**3 * 0.75**2 = 1.953125 * 0.5625 = 1.0986328125
        path = betting_capital_process([1, 0, 1, 0, 1], 0.5, bet=0.5)
        assert path[-1] == pytest.approx(1.25**3 * 0.75**2)
        assert path[-1] == pytest.approx(1.0986328125)

    def test_direction_down_is_reciprocal_step(self) -> None:
        # Downward factor 1 - lambda*(x - m0): for x=1 -> 0.75, x=0 -> 1.25
        # (mirror image of the upward bet).
        down = betting_capital_process([1, 0], 0.5, bet=0.5, direction="down")
        assert down == pytest.approx([0.75, 0.75 * 1.25])

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="Bernoulli"):
            betting_capital_process([0, 2, 1], 0.5)
        with pytest.raises(ValueError, match="non-empty"):
            betting_capital_process([], 0.5)
        with pytest.raises(ValueError, match="null_mean"):
            betting_capital_process([1, 0], 1.5)
        with pytest.raises(ValueError, match="bet"):
            betting_capital_process([1, 0], 0.5, bet=1.5)
        with pytest.raises(ValueError, match="direction"):
            betting_capital_process([1, 0], 0.5, direction="sideways")


class TestSequentialTest:
    def test_rejection_stopping_time_golden(self) -> None:
        # GOLDEN stopping time: with all ones, m0=0.5, lambda=0.5 the capital is
        # e_t = 1.25**t. We reject at the first t with 1.25**t >= 1/alpha = 20.
        #   1.25**13 = 18.1899 < 20
        #   1.25**14 = 22.7374 >= 20  -> first crossing at step 14.
        res = sequential_test([1] * 20, 0.5, alpha=0.05, bet=0.5)
        assert res.rejected is True
        assert res.stopping_time == 14
        assert res.capital == pytest.approx(res.capital_path[-1])
        assert res.capital == pytest.approx(1.25**20)
        assert res.n == 20

    def test_no_rejection_under_matching_data(self) -> None:
        # Balanced data centred on the null: capital should hover near 1, no reject.
        res = sequential_test([1, 0] * 25, 0.5, alpha=0.05, bet=0.5)
        assert res.rejected is False
        assert res.stopping_time is None
        # 25 ones and 25 zeros -> 1.25**25 * 0.75**25 = 0.9375**25.
        assert res.capital == pytest.approx(0.9375**25)

    def test_alpha_validation(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            sequential_test([1, 0], 0.5, alpha=0.0)

    def test_type_i_e_value_expectation_at_most_one(self) -> None:
        # Type-I invariant: under H0 the e-value is a mean-one martingale, so the
        # average final capital over many seeded null sequences must be <= 1 (up to
        # Monte-Carlo error). E[e_t] = 1 exactly for a fixed bet.
        rng = np.random.default_rng(20240607)
        m0 = 0.5
        n_seqs, length = 5000, 25
        finals = np.array(
            [
                sequential_test(rng.binomial(1, m0, length), m0, alpha=0.05, bet=0.5).capital
                for _ in range(n_seqs)
            ]
        )
        assert finals.mean() == pytest.approx(1.0, abs=0.1)
        assert finals.mean() <= 1.1

    def test_type_i_rejection_rate_bounded(self) -> None:
        # Ville's inequality: P(ever reject | H0) <= alpha. Empirically the
        # rejection rate over null runs should not materially exceed alpha.
        rng = np.random.default_rng(11)
        m0, alpha = 0.5, 0.1
        rejects = [
            sequential_test(rng.binomial(1, m0, 60), m0, alpha=alpha, bet=0.5).rejected
            for _ in range(2000)
        ]
        assert np.mean(rejects) <= alpha + 0.02


class TestConfidenceSequence:
    def test_contains_empirical_mean_and_shrinks(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.binomial(1, 0.7, 400)
        cs = confidence_sequence(data, confidence=0.95, bet=0.5)
        low, high = cs.final
        phat = float(np.mean(data))
        # The capital is minimised near the empirical mean, so the interval brackets it.
        assert low <= phat <= high
        assert 0.0 <= low < high <= 1.0
        # Anytime-valid interval is a running intersection -> monotone shrinking.
        assert np.all(np.diff(cs.lower) >= -1e-12)
        assert np.all(np.diff(cs.upper) <= 1e-12)
        # With 400 observations the interval should be reasonably tight.
        assert high - low < 0.3

    def test_simultaneous_coverage_meets_guarantee(self) -> None:
        # The defining guarantee: P(true mean in C_t for all t) >= 1 - alpha. We
        # check the (conservative) final running-intersection covers the truth on at
        # least 1 - alpha of seeded runs.
        true_mean = 0.7
        covered = 0
        n_runs = 300
        for seed in range(n_runs):
            rng = np.random.default_rng(seed)
            data = rng.binomial(1, true_mean, 200)
            low, high = confidence_sequence(data, confidence=0.95, bet=0.5).final
            covered += int(low <= true_mean <= high)
        assert covered / n_runs >= 0.95

    def test_excludes_means_with_large_capital(self) -> None:
        # Membership consistency tied to the (independently computed) hedged capital:
        # a candidate mean whose hedged capital exceeds 1/alpha must be excluded.
        data = [1] * 60 + [0] * 5
        cs = confidence_sequence(data, confidence=0.95, bet=0.5)
        low, high = cs.final
        threshold = 1.0 / 0.05
        for m in (0.05, 0.5):
            x = np.asarray(data, dtype=float)
            up = np.prod(1.0 + 0.5 * (x - m))
            down = np.prod(1.0 - 0.5 * (x - m))
            hedged = 0.5 * up + 0.5 * down
            assert hedged >= threshold  # strong evidence against this mean
            assert not (low <= m <= high)  # ...so it is outside the interval

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            confidence_sequence([1, 0], confidence=1.0)
        with pytest.raises(ValueError, match="grid_size"):
            confidence_sequence([1, 0], grid_size=0)
