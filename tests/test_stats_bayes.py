"""Tests for Bayesian Beta-Binomial inference."""

from __future__ import annotations

import pytest

from membench.stats.bayes import beta_binomial_posterior, prob_superiority


class TestPosterior:
    def test_uniform_prior_posterior_mean(self) -> None:
        # Beta(1,1) prior, 8/10 -> Beta(9,3), mean 9/12 = 0.75
        post = beta_binomial_posterior(8, 10, prior=(1.0, 1.0))
        assert post.mean == pytest.approx(0.75)

    def test_jeffreys_default(self) -> None:
        post = beta_binomial_posterior(5, 10)
        assert post.alpha == pytest.approx(5.5) and post.beta == pytest.approx(5.5)
        assert post.mean == pytest.approx(0.5)

    def test_credible_interval_brackets_mean(self) -> None:
        post = beta_binomial_posterior(18, 20)
        lo, hi = post.credible_interval(0.95)
        assert lo < post.mean < hi <= 1.0

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="successes"):
            beta_binomial_posterior(5, 3)
        with pytest.raises(ValueError, match="prior"):
            beta_binomial_posterior(1, 2, prior=(0.0, 1.0))


class TestProbSuperiority:
    def test_clear_winner(self) -> None:
        brief = beta_binomial_posterior(19, 20)  # ~0.95
        competitor = beta_binomial_posterior(8, 20)  # ~0.4
        assert prob_superiority(brief, competitor) > 0.99

    def test_symmetric_is_half(self) -> None:
        a = beta_binomial_posterior(10, 20)
        b = beta_binomial_posterior(10, 20)
        assert prob_superiority(a, b) == pytest.approx(0.5, abs=0.02)

    def test_deterministic(self) -> None:
        a = beta_binomial_posterior(15, 20)
        b = beta_binomial_posterior(10, 20)
        assert prob_superiority(a, b, seed=0) == prob_superiority(a, b, seed=0)
