r"""Embedding providers for the dense and graph-similarity arms.

Dense retrieval and the empirical decay measurement both need to turn text into
vectors. To keep the benchmark reproducible and runnable with no network or API
key, the default provider is a deterministic *feature-hashing* embedding: tokens
and character n-grams are hashed (with a sign bit, the standard hashing trick) into
a fixed-dimensional vector and L2-normalised, so cosine similarity reflects lexical
overlap. It is not a neural embedding, but it is deterministic across processes
(hashing via :mod:`hashlib`, never Python's salted ``hash``), dependency-free, and
a faithful stand-in for "rank by surface resemblance" — which is exactly the
mechanism the similarity arms are meant to embody.

Real neural backends (OpenAI ``text-embedding-3-small`` to match Brief's own
stack, and sentence-transformers) are added alongside this one behind optional
extras; all conform to the same :class:`EmbeddingProvider` interface so an arm
never knows which backend it is using.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "EmbeddingProvider",
    "HashingEmbeddingProvider",
    "cosine_similarity",
    "cosine_similarity_matrix",
]

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


class EmbeddingProvider(ABC):
    """Maps text to dense vectors. All arms depend on this interface, not a backend."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier of the provider (recorded in run provenance)."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the produced vectors."""

    @abstractmethod
    def embed(self, texts: list[str]) -> NDArray[np.float64]:
        """Embed a batch of texts into an ``(n, dim)`` array."""

    def embed_one(self, text: str) -> NDArray[np.float64]:
        """Embed a single text into a ``(dim,)`` vector."""
        return np.asarray(self.embed([text])[0], dtype=np.float64)


class HashingEmbeddingProvider(EmbeddingProvider):
    r"""Deterministic, offline feature-hashing embedding.

    Each text is decomposed into word tokens and intra-token character n-grams.
    Every feature is hashed with BLAKE2b into a bucket index and a sign, its
    (signed) count accumulated, and the resulting vector L2-normalised. Two texts
    that share vocabulary or sub-word structure land close in cosine space.

    Parameters
    ----------
    dim
        Vector dimensionality (number of hash buckets).
    char_ngram_range
        Inclusive ``(min, max)`` character n-gram sizes added per token (set both
        to 0 to disable and use word tokens only).
    """

    def __init__(self, dim: int = 256, char_ngram_range: tuple[int, int] = (3, 5)) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        lo, hi = char_ngram_range
        if lo < 0 or hi < lo:
            raise ValueError(f"invalid char_ngram_range {char_ngram_range!r}")
        self._dim = dim
        self._char_ngram_range = char_ngram_range

    @property
    def name(self) -> str:
        """Provider identifier, e.g. ``hashing-256``."""
        return f"hashing-{self._dim}"

    @property
    def dim(self) -> int:
        """Vector dimensionality (number of hash buckets)."""
        return self._dim

    def _features(self, text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        features = list(tokens)
        lo, hi = self._char_ngram_range
        if hi > 0:
            for token in tokens:
                padded = f"#{token}#"
                for n in range(lo, hi + 1):
                    if n <= 0:
                        continue
                    features.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
        return features

    def _bucket_and_sign(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=9).digest()
        bucket = int.from_bytes(digest[:8], "big") % self._dim
        sign = 1.0 if (digest[8] & 1) else -1.0
        return bucket, sign

    def embed(self, texts: list[str]) -> NDArray[np.float64]:
        """Embed texts into an ``(n, dim)`` L2-normalised feature-hash matrix."""
        matrix = np.zeros((len(texts), self._dim), dtype=np.float64)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                bucket, sign = self._bucket_and_sign(feature)
                matrix[row, bucket] += sign
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0  # leave all-zero (empty) rows as zero vectors
        return np.asarray(matrix / norms, dtype=np.float64)


def cosine_similarity(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Cosine similarity between two vectors (0.0 if either is the zero vector)."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_similarity_matrix(
    queries: NDArray[np.float64],
    corpus: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the ``(n_queries, n_corpus)`` cosine-similarity matrix.

    Assumes inputs may be unnormalised; normalises defensively so zero vectors
    yield zero similarity rather than NaN.
    """
    q = np.atleast_2d(np.asarray(queries, dtype=np.float64))
    c = np.atleast_2d(np.asarray(corpus, dtype=np.float64))
    qn = np.linalg.norm(q, axis=1, keepdims=True)
    cn = np.linalg.norm(c, axis=1, keepdims=True)
    qn[qn == 0.0] = 1.0
    cn[cn == 0.0] = 1.0
    return np.asarray((q / qn) @ (c / cn).T, dtype=np.float64)
