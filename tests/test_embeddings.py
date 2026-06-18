"""Tests for the deterministic hashing embedding provider and cosine helpers."""

from __future__ import annotations

import numpy as np
import pytest

from membench.retrieval.embeddings import (
    HashingEmbeddingProvider,
    cosine_similarity,
    cosine_similarity_matrix,
)


@pytest.fixture
def provider() -> HashingEmbeddingProvider:
    return HashingEmbeddingProvider(dim=512)


class TestHashingEmbeddingProvider:
    def test_dim_and_name(self, provider: HashingEmbeddingProvider) -> None:
        assert provider.dim == 512
        assert provider.name == "hashing-512"

    def test_deterministic_across_instances(self) -> None:
        a = HashingEmbeddingProvider(dim=256).embed_one("withAuditLog on every export")
        b = HashingEmbeddingProvider(dim=256).embed_one("withAuditLog on every export")
        np.testing.assert_array_equal(a, b)

    def test_vectors_are_unit_norm(self, provider: HashingEmbeddingProvider) -> None:
        vecs = provider.embed(["audit logging middleware", "date range picker component"])
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-9)

    def test_empty_text_is_zero_vector(self, provider: HashingEmbeddingProvider) -> None:
        vec = provider.embed_one("")
        assert np.linalg.norm(vec) == 0.0

    def test_similar_text_more_similar_than_unrelated(
        self, provider: HashingEmbeddingProvider
    ) -> None:
        base = provider.embed_one("all data exports must call withAuditLog for compliance")
        near = provider.embed_one("every export endpoint should call withAuditLog")
        far = provider.embed_one("use DateRangePicker for the calendar UI")
        assert cosine_similarity(base, near) > cosine_similarity(base, far)

    def test_shape(self, provider: HashingEmbeddingProvider) -> None:
        assert provider.embed(["a", "b", "c"]).shape == (3, 512)

    @pytest.mark.parametrize(("dim", "rng"), [(0, (3, 5)), (16, (5, 3)), (16, (-1, 2))])
    def test_validation(self, dim: int, rng: tuple[int, int]) -> None:
        with pytest.raises(ValueError):
            HashingEmbeddingProvider(dim=dim, char_ngram_range=rng)

    def test_word_only_mode(self) -> None:
        prov = HashingEmbeddingProvider(dim=128, char_ngram_range=(0, 0))
        # identical token sets -> identical vectors regardless of char n-grams
        assert cosine_similarity(prov.embed_one("alpha beta"), prov.embed_one("beta alpha")) == (
            pytest.approx(1.0)
        )


class TestCosineHelpers:
    def test_cosine_of_orthogonal_is_zero(self) -> None:
        assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0

    def test_cosine_with_zero_vector(self) -> None:
        assert cosine_similarity(np.zeros(3), np.ones(3)) == 0.0

    def test_matrix_shape_and_values(self) -> None:
        q = np.array([[1.0, 0.0], [0.0, 1.0]])
        c = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        sim = cosine_similarity_matrix(q, c)
        assert sim.shape == (2, 3)
        assert sim[0, 0] == pytest.approx(1.0)
        assert sim[0, 1] == pytest.approx(1.0 / np.sqrt(2))
        assert sim[0, 2] == pytest.approx(0.0)

    def test_matrix_handles_zero_rows(self) -> None:
        sim = cosine_similarity_matrix(np.zeros((1, 3)), np.ones((2, 3)))
        np.testing.assert_allclose(sim, 0.0)
