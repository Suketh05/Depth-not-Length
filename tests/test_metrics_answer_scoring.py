"""Tests for public-benchmark answer scoring (paper sec:pubbench / tab:stdbench).

Every golden here is hand-computed from the formula (exact fractions), never
produced by running the module under test. Paper numbers (LoCoMo 87.6, DMR
94.2) are provenance in the module docstrings only and are never asserted.
"""

from __future__ import annotations

import random

import pytest

from membench.metrics.answer_scoring import (
    QARecord,
    exact_match,
    fact_f1,
    llm_judge_accuracy,
    mean_fact_f1,
    normalize_answer,
    split_facts,
    token_f1,
    token_overlap_judge,
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


class TestTokenF1:
    def test_golden_two_of_three_vs_four(self) -> None:
        # Hand-computed: pred tokens {paris, france, europe} (3), gold tokens
        # {paris, france, in, winter} (4), overlap 2.
        # P = 2/3, R = 2/4 = 1/2, F1 = 2PR/(P+R) = 2*overlap/(3+4) = 4/7.
        result = token_f1("Paris, France, Europe", "Paris France in winter")
        assert result.overlap == 2
        assert result.n_predicted == 3
        assert result.n_gold == 4
        assert result.precision == 2 / 3
        assert result.recall == 1 / 2
        assert result.f1 == 4 / 7  # exact: one integer division, no rounding

    def test_golden_multiset_overlap(self) -> None:
        # Bag semantics: pred [yes yes yes] vs gold [yes yes no] ->
        # overlap = min(3, 2) = 2; P = 2/3, R = 2/3, F1 = 2*2/(3+3) = 2/3.
        result = token_f1("yes yes yes", "yes yes no")
        assert result.overlap == 2
        assert result.precision == 2 / 3
        assert result.recall == 2 / 3
        assert result.f1 == 2 / 3

    def test_exact_match_scores_one(self) -> None:
        result = token_f1("The Eiffel Tower", "eiffel tower!")
        assert result.f1 == 1.0
        assert result.precision == 1.0
        assert result.recall == 1.0

    def test_disjoint_scores_zero(self) -> None:
        result = token_f1("red", "blue")
        assert result.f1 == 0.0
        assert result.overlap == 0

    def test_both_empty_is_perfect(self) -> None:
        # SQuAD degenerate rule: both normalize to "" -> 1.0.
        assert token_f1("", "the an").f1 == 1.0

    def test_one_empty_is_zero(self) -> None:
        assert token_f1("", "paris").f1 == 0.0
        assert token_f1("paris", "the").f1 == 0.0

    def test_symmetric_f1_seeded_sample(self) -> None:
        # F1 is symmetric (P and R swap); range stays in [0, 1].
        texts = _random_texts(seed=3, n=30)
        for a in texts[:15]:
            for b in texts[15:]:
                fwd, rev = token_f1(a, b), token_f1(b, a)
                assert fwd.f1 == pytest.approx(rev.f1)
                assert fwd.precision == pytest.approx(rev.recall)
                assert 0.0 <= fwd.f1 <= 1.0

    def test_harmonic_mean_sandwich(self) -> None:
        # The harmonic mean lies between its arguments: min(P,R) <= F1 <= max(P,R).
        # On the 4/7 golden: 1/2 <= 4/7 <= 2/3.
        result = token_f1("Paris, France, Europe", "Paris France in winter")
        assert min(result.precision, result.recall) <= result.f1
        assert result.f1 <= max(result.precision, result.recall)


class TestSplitFacts:
    def test_splits_on_newlines_semicolons_sentences(self) -> None:
        text = "Lives in Lyon; works at Acme.\nHas two cats. Plays chess!"
        assert split_facts(text) == (
            "Lives in Lyon",
            "works at Acme.",
            "Has two cats.",
            "Plays chess!",
        )

    def test_drops_empty_fragments(self) -> None:
        assert split_facts(";;\n\n ; ") == ()

    def test_single_fact_untouched(self) -> None:
        assert split_facts("moved to Berlin in 2021") == ("moved to Berlin in 2021",)


class TestFactF1:
    def test_golden_two_of_three_predicted_two_of_four_gold(self) -> None:
        # Hand-computed (the tab:stdbench DMR metric on a tiny set):
        # predicted facts 3, gold facts 4, matched (normalized-identical) 2.
        # P = 2/3, R = 2/4 = 1/2, F1 = 2PR/(P+R) = 2*2/(3+4) = 4/7.
        predicted = ["Lives in Lyon", "Works at Acme", "Owns a dog"]
        gold = ["lives in lyon!", "works at Acme.", "has two cats", "plays chess"]
        result = fact_f1(predicted, gold)
        assert result.overlap == 2
        assert result.n_predicted == 3
        assert result.n_gold == 4
        assert result.precision == 2 / 3
        assert result.recall == 1 / 2
        assert result.f1 == 4 / 7  # exact fraction

    def test_duplicate_facts_deduplicate(self) -> None:
        # Restating the same fact (any surface form) neither helps nor hurts:
        # both predictions normalize to one fact -> P = R = F1 = 1.
        result = fact_f1(["Lives in Lyon.", "lives in LYON"], ["lives in lyon"])
        assert result.n_predicted == 1
        assert result.f1 == 1.0

    def test_hallucinated_facts_cost_precision_only(self) -> None:
        # All 2 gold facts recovered plus 2 inventions: R = 1, P = 2/4 = 1/2,
        # F1 = 2*2/(4+2) = 2/3.
        predicted = ["fact one", "fact two", "invented three", "invented four"]
        result = fact_f1(predicted, ["fact one", "fact two"])
        assert result.recall == 1.0
        assert result.precision == 1 / 2
        assert result.f1 == 2 / 3

    def test_matching_is_normalized_not_fuzzy(self) -> None:
        # "The same fact" up to normalization matches; a paraphrase does not
        # (semantic credit is the LLM judge's job, not the deterministic one).
        assert fact_f1(["The cat is black!"], ["cat is black"]).f1 == 1.0
        assert fact_f1(["the cat is dark-colored"], ["cat is black"]).f1 == 0.0

    def test_both_empty_perfect_one_empty_zero(self) -> None:
        assert fact_f1([], []).f1 == 1.0
        assert fact_f1(["the", "an"], []).f1 == 1.0  # all noise == no facts
        assert fact_f1([], ["real fact"]).f1 == 0.0
        assert fact_f1(["real fact"], []).f1 == 0.0

    def test_order_invariant_seeded_shuffle(self) -> None:
        # Set semantics: shuffling fact order never changes the score.
        rng = random.Random(4)
        predicted = ["a b", "c d", "e f", "g h"]
        gold = ["c d", "e f", "x y"]
        baseline = fact_f1(predicted, gold)
        for _ in range(20):
            shuffled_pred = rng.sample(predicted, k=len(predicted))
            shuffled_gold = rng.sample(gold, k=len(gold))
            assert fact_f1(shuffled_pred, shuffled_gold) == baseline


class TestMeanFactF1:
    def test_macro_average(self) -> None:
        # Hand-computed: mean of per-question F1 {1.0, 4/7} = (1 + 4/7)/2 = 11/14.
        perfect = fact_f1(["a"], ["a"])
        partial = fact_f1(["m one", "m two", "m three"], ["m one", "m two", "g", "h"])
        assert partial.f1 == 4 / 7
        assert mean_fact_f1([perfect, partial]) == pytest.approx(11 / 14)

    def test_single_question(self) -> None:
        only = fact_f1(["a"], ["b"])
        assert mean_fact_f1([only]) == 0.0

    def test_empty_run_raises(self) -> None:
        # No questions -> no benchmark score; a silent 0 would fabricate one.
        with pytest.raises(ValueError, match="at least one"):
            mean_fact_f1([])


def _record(task_id: str, prediction: str, gold: str) -> QARecord:
    return QARecord(
        task_id=task_id,
        question="Where does Ann live?",
        prediction=prediction,
        gold=gold,
    )


class TestTokenOverlapJudge:
    def test_exact_match_passes(self) -> None:
        assert token_overlap_judge("q", "The Eiffel Tower", "eiffel tower")

    def test_disjoint_fails(self) -> None:
        assert not token_overlap_judge("q", "blue bike", "red car")

    def test_boundary_is_inclusive(self) -> None:
        # F1 = 2*1/(2+2) = 1/2 sits exactly on the default 0.5 threshold -> pass.
        assert token_overlap_judge("q", "blue car", "red car")

    def test_threshold_is_tunable(self) -> None:
        # Same 1/2-F1 pair fails a stricter threshold.
        assert not token_overlap_judge("q", "blue car", "red car", threshold=0.6)

    def test_question_does_not_change_verdict(self) -> None:
        # Deterministic fallback ignores the question by construction.
        assert token_overlap_judge("q1", "lyon", "Lyon") == token_overlap_judge(
            "q2", "lyon", "Lyon"
        )


class TestLLMJudgeAccuracy:
    def test_golden_three_of_four(self) -> None:
        # Default fallback judge: verdicts (T, T, T, F) -> accuracy = 3/4 exactly.
        records = [
            _record("t1", "Lyon", "lyon"),  # exact match -> T
            _record("t2", "in Lyon, France", "Lyon France"),  # overlap 2: F1 = 2*2/(3+2) = 4/5 -> T
            _record("t3", "blue car", "red car"),  # F1 = 1/2 boundary -> T
            _record("t4", "no idea", "Lyon"),  # overlap 0 -> F
        ]
        result = llm_judge_accuracy(records)
        assert result.verdicts == (True, True, True, False)
        assert result.correct == 3
        assert result.total == 4
        assert result.accuracy == 3 / 4  # exact fraction

    def test_custom_judge_hook_receives_armblind_triple(self) -> None:
        # The hook sees exactly (question, prediction, gold) -- nothing else
        # exists to leak arm identity (paper sec:grader arm-blind protocol).
        seen: list[tuple[str, str, str]] = []

        def spy_judge(question: str, prediction: str, gold: str) -> bool:
            seen.append((question, prediction, gold))
            return prediction == gold

        records = [_record("t1", "a", "a"), _record("t2", "b", "c")]
        result = llm_judge_accuracy(records, judge=spy_judge)
        assert seen == [("Where does Ann live?", "a", "a"), ("Where does Ann live?", "b", "c")]
        assert result.verdicts == (True, False)
        assert result.accuracy == 1 / 2

    def test_all_correct_and_all_wrong_extremes(self) -> None:
        perfect = llm_judge_accuracy([_record("t", "x", "x")])
        assert perfect.accuracy == 1.0
        hopeless = llm_judge_accuracy([_record("t", "x", "y")], judge=lambda q, p, g: False)
        assert hopeless.accuracy == 0.0

    def test_empty_run_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            llm_judge_accuracy([])
