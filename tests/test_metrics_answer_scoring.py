"""Tests for public-benchmark answer scoring (paper sec:pubbench / tab:stdbench).

Every golden here is hand-computed from the formula (exact fractions), never
produced by running the module under test. Paper numbers (LoCoMo 87.6, DMR
94.2) are provenance in the module docstrings only and are never asserted.
"""

from __future__ import annotations

import random

from membench.metrics.answer_scoring import (
    exact_match,
    normalize_answer,
    tokenize_answer,
)

# Deterministic vocabulary for seeded invariant checks (no benchmark data).
_VOCAB = ("paris", "france", "The", "Eiffel", "tower", "in", "2024", "an", "answer")


def _random_texts(seed: int, n: int) -> list[str]:
    """Deterministic sample of messy strings drawn from a small vocabulary."""
    rng = random.Random(seed)
    texts = []
    for _ in range(n):
        words = rng.choices(_VOCAB, k=rng.randint(0, 6))
        joiner = rng.choice([" ", "  ", " \t ", "\n"])
        punct = rng.choice(["", "!", "...", ",", "?!"])
        texts.append(joiner.join(words) + punct)
    return texts


class TestNormalizeAnswer:
    def test_lowercase_punctuation_articles_whitespace(self) -> None:
        # SQuAD pipeline: lowercase -> strip punctuation -> drop a/an/the -> collapse.
        assert normalize_answer("The  Eiffel Tower, in Paris!") == "eiffel tower in paris"

    def test_article_removed_only_as_whole_word(self) -> None:
        # "than" contains "an"; "theory" contains "the" -- neither is an article.
        assert normalize_answer("more than a theory") == "more than theory"

    def test_all_noise_normalizes_to_empty(self) -> None:
        # Nothing but punctuation, articles, and whitespace -> "no answer".
        assert normalize_answer("  The ... an!  a  ") == ""

    def test_idempotent_seeded_sample(self) -> None:
        # normalize(normalize(t)) == normalize(t) over a deterministic sample.
        for text in _random_texts(seed=0, n=200):
            once = normalize_answer(text)
            assert normalize_answer(once) == once

    def test_case_insensitive_seeded_sample(self) -> None:
        for text in _random_texts(seed=1, n=200):
            assert normalize_answer(text.upper()) == normalize_answer(text.lower())


class TestTokenizeAnswer:
    def test_tokenizes_normalized_form(self) -> None:
        assert tokenize_answer("The Eiffel Tower!") == ["eiffel", "tower"]

    def test_empty_answer_gives_no_tokens(self) -> None:
        assert tokenize_answer("the ... an") == []


class TestExactMatch:
    def test_match_up_to_normalization(self) -> None:
        assert exact_match("The Eiffel Tower!", "eiffel   tower")

    def test_mismatch(self) -> None:
        assert not exact_match("Eiffel Tower", "Louvre")

    def test_both_no_answer_match(self) -> None:
        # Two strings that both normalize to "" agree (SQuAD no-answer rule).
        assert exact_match("", "the an a ...")

    def test_symmetry_seeded_sample(self) -> None:
        texts = _random_texts(seed=2, n=40)
        for a in texts[:20]:
            for b in texts[20:]:
                assert exact_match(a, b) == exact_match(b, a)
