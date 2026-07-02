"""Tests for the public-benchmark answer scorers (LoCoMo acc, DMR fact F1).

Every golden here is hand-computed in the comment and asserted verbatim; none is
produced by running the module under test. Paper anchors: tab:stdbench (LoCoMo
LLM-judge acc, DMR fact F1), sec:pubbench, sec:grader (arm-blind judge protocol).
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from membench.metrics.answer_scoring import (
    AnswerJudge,
    AnswerRecord,
    JudgeVerdict,
    TokenOverlapJudge,
    answer_exact_match,
    answer_tokens,
    fact_f1,
    llm_judge_accuracy,
    normalize_answer,
    token_f1,
)


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


class TestTokenF1:
    def test_two_of_three_shared_tokens_golden(self) -> None:
        # pred tokens = [red, green, blue]; gold tokens = [green, blue, yellow].
        # num_same = 2 (green, blue). P = 2/3, R = 2/3, F1 = 2PR/(P+R) = 2/3.
        assert token_f1("red green blue", "green blue yellow") == pytest.approx(
            float(Fraction(2, 3))
        )

    def test_asymmetric_lengths_golden(self) -> None:
        # pred [cat] vs gold [cat sat mat]: num_same=1, P=1/1=1, R=1/3,
        # F1 = 2*1*(1/3)/(1 + 1/3) = (2/3)/(4/3) = 1/2.
        assert token_f1("cat", "the cat sat mat") == pytest.approx(0.5)

    def test_exact_match_is_one(self) -> None:
        assert token_f1("The Answer", "answer!") == pytest.approx(1.0)

    def test_disjoint_tokens_is_zero(self) -> None:
        assert token_f1("alpha beta", "gamma delta") == 0.0

    def test_both_empty_is_one_but_one_empty_is_zero(self) -> None:
        # SQuAD boundary convention: both empty -> vacuously perfect; exactly one
        # empty -> impossible overlap.
        assert token_f1("the a an", "") == pytest.approx(1.0)
        assert token_f1("something", "the a an") == 0.0

    def test_multiset_semantics_penalise_repetition(self) -> None:
        # pred [ha, ha, ha] vs gold [ha]: num_same = 1 (multiset min).
        # P = 1/3, R = 1/1, F1 = 2*(1/3)*1/(1/3 + 1) = (2/3)/(4/3) = 1/2.
        assert token_f1("ha ha ha", "ha") == pytest.approx(0.5)


class TestAnswerExactMatch:
    def test_match_up_to_normalization(self) -> None:
        assert answer_exact_match("The Cat.", "cat") is True

    def test_mismatch(self) -> None:
        assert answer_exact_match("cat", "dog") is False


class TestFactF1:
    def test_two_of_three_precision_two_of_four_recall_golden(self) -> None:
        # Golden pack: predicted 3 facts, 2 in gold => precision = 2/3.
        # gold 4 facts, 2 recovered => recall = 2/4 = 1/2.
        # F1 = 2*P*R/(P+R) = 2*(2/3)*(1/2)/((2/3)+(1/2)) = (2/3)/(7/6) = 4/7.
        result = fact_f1(
            predicted_facts=["Alpha", "beta", "gamma"],
            gold_facts=["alpha", "BETA", "delta", "epsilon"],
        )
        assert result.true_positives == 2
        assert result.n_predicted == 3
        assert result.n_gold == 4
        assert result.precision == pytest.approx(float(Fraction(2, 3)))
        assert result.recall == pytest.approx(0.5)
        assert result.f1 == pytest.approx(float(Fraction(4, 7)))

    def test_perfect_recovery_is_one(self) -> None:
        result = fact_f1(["a1", "b2", "c3"], ["a1", "b2", "c3"])
        assert result.f1 == pytest.approx(1.0)
        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(1.0)

    def test_f1_percent_is_tab_stdbench_unit(self) -> None:
        # percent == 100 * f1: the DMR "fact F1 (%)" column unit (arithmetic
        # identity, not a reproduction of the paper's 94.2 headline).
        result = fact_f1(["Alpha", "beta", "gamma"], ["alpha", "BETA", "delta", "epsilon"])
        assert result.f1_percent == pytest.approx(100.0 * result.f1)
        assert result.f1_percent == pytest.approx(100.0 * float(Fraction(4, 7)))

    def test_both_empty_is_vacuously_perfect(self) -> None:
        result = fact_f1([], [])
        assert (result.precision, result.recall, result.f1) == (1.0, 1.0, 1.0)
        assert result.true_positives == 0

    def test_one_side_empty_is_zero(self) -> None:
        # Non-empty prediction, empty gold: precision 0/1, recall vacuous -> F1 0.
        assert fact_f1(["x1"], []).f1 == 0.0
        # Empty prediction, non-empty gold: recall 0/1 -> F1 0.
        assert fact_f1([], ["y2"]).f1 == 0.0

    def test_disjoint_sets_is_zero(self) -> None:
        result = fact_f1(["p1", "q2"], ["r3", "s4"])
        assert result.true_positives == 0
        assert result.f1 == 0.0

    def test_normalization_and_dedup_collapse_facts(self) -> None:
        # "The Widget!" and "widget" normalize to the same fact; punctuation-only
        # facts drop out. So predicted collapses to {"widget"}, gold to {"widget"}.
        result = fact_f1(["The Widget!", "widget", "!!!"], ["Widget", "..."])
        assert result.n_predicted == 1
        assert result.n_gold == 1
        assert result.f1 == pytest.approx(1.0)


class TestTokenOverlapJudge:
    def test_exact_match_is_correct_with_full_score(self) -> None:
        verdict = TokenOverlapJudge()("q", "Paris", "paris.")
        assert verdict.correct is True
        assert verdict.score == pytest.approx(1.0)

    def test_partial_overlap_above_threshold_is_correct(self) -> None:
        # gold [paris france], pred [paris]: F1 = 2*(1)*(1/2)/(1+1/2) = 2/3 >= 0.5.
        verdict = TokenOverlapJudge(threshold=0.5)("q", "Paris France", "Paris")
        assert verdict.score == pytest.approx(float(Fraction(2, 3)))
        assert verdict.correct is True

    def test_overlap_below_threshold_is_incorrect(self) -> None:
        verdict = TokenOverlapJudge(threshold=0.9)("q", "Paris France", "Paris")
        assert verdict.correct is False

    def test_disjoint_answer_is_incorrect(self) -> None:
        verdict = TokenOverlapJudge()("q", "Paris", "London")
        assert verdict.correct is False
        assert verdict.score == 0.0

    def test_satisfies_answer_judge_protocol(self) -> None:
        assert isinstance(TokenOverlapJudge(), AnswerJudge)

    def test_threshold_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            TokenOverlapJudge(threshold=1.5)


class TestLlmJudgeAccuracy:
    def test_three_of_four_correct_is_075(self) -> None:
        # Deterministic default judge; 3 exact/near matches, 1 disjoint -> 3/4.
        records = [
            AnswerRecord("q1", "Paris", "paris"),
            AnswerRecord("q2", "1999", "1999."),
            AnswerRecord("q3", "blue", "blue"),
            AnswerRecord("q4", "Monday", "Friday"),
        ]
        result = llm_judge_accuracy(records, TokenOverlapJudge())
        assert result.n_correct == 3
        assert result.n_total == 4
        assert result.accuracy == pytest.approx(0.75)
        assert len(result.verdicts) == 4

    def test_all_correct_saturates_at_one(self) -> None:
        records = [AnswerRecord("q", "a1 b2", "a1 b2"), AnswerRecord("q", "c3", "c3")]
        result = llm_judge_accuracy(records, TokenOverlapJudge())
        assert result.accuracy == pytest.approx(1.0)
        assert result.percent == pytest.approx(100.0)  # tab:stdbench LoCoMo unit (%)

    def test_all_wrong_floors_at_zero(self) -> None:
        records = [AnswerRecord("q", "alpha", "zeta"), AnswerRecord("q", "beta", "omega")]
        assert llm_judge_accuracy(records, TokenOverlapJudge()).accuracy == 0.0

    def test_empty_records_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            llm_judge_accuracy([], TokenOverlapJudge())

    def test_custom_judge_hook_is_used_and_arm_blind(self) -> None:
        # A stub live judge: correct iff the gold appears verbatim in the prediction.
        # Its signature has no arm parameter, so it cannot condition on the arm.
        def substring_judge(question: str, gold_answer: str, predicted_answer: str) -> JudgeVerdict:
            hit = gold_answer.lower() in predicted_answer.lower()
            return JudgeVerdict(correct=hit, score=1.0 if hit else 0.0, rationale="substring")

        records = [
            AnswerRecord("q1", "widget", "the widget is used"),
            AnswerRecord("q2", "gadget", "no match here"),
        ]
        result = llm_judge_accuracy(records, substring_judge)
        assert result.accuracy == pytest.approx(0.5)
