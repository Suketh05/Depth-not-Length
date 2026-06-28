"""Tests for the recency-correct supersession-rate metric."""

from __future__ import annotations

import pytest

from membench.metrics.supersession import (
    SupersessionCase,
    case_from_items,
    current_decision,
    score_supersession,
    supersession_rate,
)
from membench.types import MemoryItem


def _case(
    chain: tuple[str, ...],
    supersedes: tuple[tuple[str, str], ...],
    applied: str | None,
) -> SupersessionCase:
    return SupersessionCase(chain=chain, supersedes=supersedes, applied=applied)


# Hand-built golden set of 4 cases (the prompt's golden):
#   1. A<-B<-C (C current), applied C   -> correct
#   2. D<-E    (E current), applied E   -> correct
#   3. F<-G    (G current), applied F   -> STALE (wrong)
#   4. H<-I<-J (J current), applied J   -> correct
# Recency-correct = 3 of 4  =>  rate = 3/4 = 0.75 (computed by hand).
GOLDEN_CASES = (
    _case(("A", "B", "C"), (("B", "A"), ("C", "B")), "C"),
    _case(("D", "E"), (("E", "D"),), "E"),
    _case(("F", "G"), (("G", "F"),), "F"),
    _case(("H", "I", "J"), (("I", "H"), ("J", "I")), "J"),
)


class TestCurrentDecision:
    def test_linear_chain_head_is_current(self) -> None:
        # C supersedes B supersedes A -> only C is non-superseded.
        assert current_decision(_case(("A", "B", "C"), (("B", "A"), ("C", "B")), "C")) == "C"

    def test_single_decision_is_its_own_current(self) -> None:
        assert current_decision(_case(("A",), (), "A")) == "A"

    def test_cycle_has_no_current(self) -> None:
        with pytest.raises(ValueError, match="non-superseded"):
            current_decision(_case(("A", "B"), (("A", "B"), ("B", "A")), "A"))

    def test_ambiguous_branch_rejected(self) -> None:
        # Two independent live heads (B and C) -> no single latest decision.
        with pytest.raises(ValueError, match="non-superseded"):
            current_decision(_case(("A", "B", "C"), (("B", "A"),), "B"))


class TestSupersessionRate:
    def test_golden_rate_is_three_quarters(self) -> None:
        # Independently hand-computed: 3 of 4 cases recency-correct.
        assert supersession_rate(GOLDEN_CASES) == pytest.approx(0.75)

    def test_stale_application_is_incorrect(self) -> None:
        assert supersession_rate([_case(("F", "G"), (("G", "F"),), "F")]) == 0.0

    def test_applying_none_is_incorrect(self) -> None:
        assert supersession_rate([_case(("F", "G"), (("G", "F"),), None)]) == 0.0

    def test_perfect_when_always_current(self) -> None:
        assert supersession_rate([_case(("F", "G"), (("G", "F"),), "G")]) == 1.0

    def test_empty_input_handled(self) -> None:
        assert supersession_rate([]) == 0.0

    def test_rate_in_unit_interval(self) -> None:
        rate = supersession_rate(GOLDEN_CASES)
        assert 0.0 <= rate <= 1.0


class TestScoreSupersession:
    def test_golden_report(self) -> None:
        report = score_supersession(GOLDEN_CASES)
        assert report.total == 4
        assert report.recency_correct == 3
        assert report.rate == pytest.approx(0.75)
        # Chance baseline = mean(1/len(chain)) over (3,2,2,3)-long chains:
        #   (1/3 + 1/2 + 1/2 + 1/3) / 4 = (5/3) / 4 = 5/12 ~= 0.416667
        assert report.chance_baseline == pytest.approx(5.0 / 12.0)
        # The system must beat chance on this golden set.
        assert report.rate > report.chance_baseline

    def test_empty_report_is_zeroed(self) -> None:
        report = score_supersession([])
        assert report.total == 0
        assert report.recency_correct == 0
        assert report.rate == 0.0
        assert report.chance_baseline == 0.0


class TestCaseFromItems:
    def test_builds_case_from_edge_metadata(self) -> None:
        items = [
            MemoryItem(item_id="A", text="original policy"),
            MemoryItem(
                item_id="B", text="revised policy", metadata={"edges": [("A", "supersedes")]}
            ),
        ]
        case = case_from_items(items, applied="B")
        assert case.chain == ("A", "B")
        assert case.supersedes == (("B", "A"),)
        assert current_decision(case) == "B"

    def test_ignores_non_supersession_edges(self) -> None:
        items = [
            MemoryItem(item_id="A", text="constraint"),
            MemoryItem(
                item_id="B",
                text="justification",
                metadata={"edges": [("A", "derived_from")]},
            ),
        ]
        case = case_from_items(items, applied="B")
        assert case.supersedes == ()


class TestValidation:
    def test_empty_chain_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _case((), (), None)

    def test_duplicate_chain_ids_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            _case(("A", "A"), (), "A")

    def test_self_supersession_rejected(self) -> None:
        with pytest.raises(ValueError, match="itself"):
            _case(("A",), (("A", "A"),), "A")

    def test_edge_outside_chain_rejected(self) -> None:
        with pytest.raises(ValueError, match="outside the chain"):
            _case(("A",), (("A", "Z"),), "A")
