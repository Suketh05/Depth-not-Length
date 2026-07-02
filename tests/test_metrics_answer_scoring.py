"""Tests for the public-benchmark answer scorers (LoCoMo acc, DMR fact F1).

Every golden here is hand-computed in the comment and asserted verbatim; none is
produced by running the module under test. Paper anchors: tab:stdbench (LoCoMo
LLM-judge acc, DMR fact F1), sec:pubbench, sec:grader (arm-blind judge protocol).
"""

from __future__ import annotations

from membench.metrics.answer_scoring import answer_tokens, normalize_answer


class TestNormalizeAnswer:
    def test_lowercases_strips_punctuation_and_articles(self) -> None:
        # "The Cat, sat!" -> lower "the cat, sat!" -> depunct "the cat sat"
        # -> drop article "the" -> collapse ws -> "cat sat".
        assert normalize_answer("The Cat, sat!") == "cat sat"

    def test_drops_all_three_articles_as_whole_words(self) -> None:
        # Articles a/an/the are removed only as whole words: "a" and "an" and "the"
        # go, but "anaconda" (contains "an") and "theory" (contains "the") stay.
        assert normalize_answer("a an the anaconda theory") == "anaconda theory"

    def test_collapses_internal_and_edge_whitespace(self) -> None:
        assert normalize_answer("  hello    world  ") == "hello world"

    def test_all_noise_normalizes_to_empty_string(self) -> None:
        # Pure punctuation + articles + whitespace has no content tokens.
        assert normalize_answer("the a an ... , !") == ""

    def test_idempotent(self) -> None:
        once = normalize_answer("The Answer, is: 42!")
        assert normalize_answer(once) == once

    def test_unicode_and_digits_preserved(self) -> None:
        assert normalize_answer("Room 101 (Café)") == "room 101 café"


class TestAnswerTokens:
    def test_tokenizes_normalized_form(self) -> None:
        assert answer_tokens("The quick, brown fox!") == ["quick", "brown", "fox"]

    def test_empty_answer_is_empty_token_list(self) -> None:
        assert answer_tokens("the a an") == []
        assert answer_tokens("") == []
