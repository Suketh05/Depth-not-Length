"""Tests for Cohen's kappa and the dual-annotation certification protocol."""

from __future__ import annotations

import pytest
from sklearn.metrics import cohen_kappa_score

from membench.datasets.depth_labeling import certify_depths, cohens_kappa


class TestCohensKappa:
    def test_perfect_agreement(self) -> None:
        assert cohens_kappa([1, 2, 3, 2], [1, 2, 3, 2]) == pytest.approx(1.0)

    def test_matches_sklearn_unweighted(self) -> None:
        a, b = [1, 2, 3, 1, 2, 3, 1], [1, 2, 2, 1, 3, 3, 2]
        assert cohens_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b), abs=1e-9)

    def test_matches_sklearn_linear_weighted(self) -> None:
        a, b = [1, 2, 3, 1, 2, 3, 1], [1, 2, 2, 1, 3, 3, 2]
        assert cohens_kappa(a, b, weights="linear") == pytest.approx(
            cohen_kappa_score(a, b, weights="linear"), abs=1e-9
        )

    def test_matches_sklearn_quadratic_weighted(self) -> None:
        a, b = [1, 2, 3, 1, 2, 3, 2], [1, 3, 2, 1, 2, 3, 1]
        assert cohens_kappa(a, b, weights="quadratic") == pytest.approx(
            cohen_kappa_score(a, b, weights="quadratic"), abs=1e-9
        )

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            cohens_kappa([1, 2], [1])

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            cohens_kappa([], [])


class TestCertifyDepths:
    def test_drops_large_disagreements_and_certifies_lower(self) -> None:
        a = {"t1": 3, "t2": 2, "t3": 3}
        b = {"t1": 3, "t2": 3, "t3": 1}  # t3 disagrees by 2 hops -> dropped
        report = certify_depths(a, b, max_disagreement=1)
        assert report.dropped == ("t3",)
        assert report.certified == {"t1": 3, "t2": 2}  # lower of the two for t2
        assert report.n == 3

    def test_reports_exact_agreement_and_kappa(self) -> None:
        a = {"t1": 1, "t2": 2, "t3": 3}
        b = {"t1": 1, "t2": 2, "t3": 3}
        report = certify_depths(a, b)
        assert report.exact_agreement == 1.0
        assert report.kappa == pytest.approx(1.0)

    def test_requires_shared_tasks(self) -> None:
        with pytest.raises(ValueError, match="share no tasks"):
            certify_depths({"a": 1}, {"b": 1})
