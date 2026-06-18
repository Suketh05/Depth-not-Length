"""Trivial tests for the scoring layer against the Task contract."""

from benchmarks.task import MemoryItem, Task
from scoring.compliance import score_compliance
from scoring.correctness import score_correctness
from scoring.cost import AttemptCost, score_cost_to_correct, score_retrieval


def _task(governing_decisions, memory_corpus, repo_ref="repo@main"):
    return Task(
        task_id="t1",
        dataset="dcbench",
        query="q",
        repo_ref=repo_ref,
        memory_corpus=memory_corpus,
        governing_decisions=governing_decisions,
        depth=1,
        spec_variant="full",
        scorer="compliance",
    )


def test_compliance_honors_decision_via_extracted_identifier():
    corpus = [MemoryItem(item_id="D-002", text="Must call withAuditLog() before streaming.")]
    task = _task(["D-002"], corpus)

    result = score_compliance("I wrapped the export with withAuditLog as required.", task)

    assert result.honored == 1
    assert result.honored_ids == ["D-002"]
    assert result.rate == 1.0


def test_compliance_falls_back_to_bare_id_when_no_keyword_extractable():
    corpus = [MemoryItem(item_id="D-003", text="secondary button variant for read only actions")]
    task = _task(["D-003"], corpus)

    result = score_compliance("Mentions D-003 explicitly.", task)

    assert result.honored_ids == ["D-003"]


def test_compliance_with_no_governing_decisions_is_trivially_satisfied():
    task = _task([], [])

    result = score_compliance("anything", task)

    assert result.total == 0
    assert result.rate == 1.0


def test_correctness_requires_full_compliance():
    corpus = [MemoryItem(item_id="D-001", text="Use DateRangePicker.")]
    task = _task(["D-001"], corpus)
    compliance = score_compliance("no relevant identifiers here", task)

    result = score_correctness("no relevant identifiers here", task, compliance)

    assert result.merge_ready is False
    assert "unmet" in result.reason


def test_correctness_merge_ready_when_compliant_and_non_refusal():
    corpus = [MemoryItem(item_id="D-001", text="Use DateRangePicker.")]
    task = _task(["D-001"], corpus)
    compliance = score_compliance("Added DateRangePicker to the form.", task)

    result = score_correctness("Added DateRangePicker to the form.", task, compliance)

    assert result.merge_ready is True


def test_retrieval_precision_recall_against_gold_ids():
    result = score_retrieval(retrieved_ids=["D-001", "D-002", "D-099"], governing_decisions=["D-001", "D-002"])

    assert result.true_positives == ["D-001", "D-002"]
    assert result.recall == 1.0
    assert result.precision == 2 / 3


def test_retrieval_with_no_gold_decisions_is_trivially_full_recall():
    result = score_retrieval(retrieved_ids=[], governing_decisions=[])

    assert result.recall == 1.0
    assert result.precision == 1.0


def test_cost_to_correct_sums_tokens_and_stops_at_first_correct_attempt():
    attempts = [
        AttemptCost(input_tokens=100, output_tokens=10, dollars=0.01, correct=False),
        AttemptCost(input_tokens=120, output_tokens=12, dollars=0.012, correct=True),
    ]

    result = score_cost_to_correct("t1", attempts)

    assert result.resolved is True
    assert result.iterations_to_correct == 2
    assert result.total_input_tokens == 220
    assert result.total_dollars == 0.022


def test_cost_to_correct_never_resolved_when_no_attempt_is_correct():
    attempts = [AttemptCost(input_tokens=50, output_tokens=5, dollars=0.005, correct=False)]

    result = score_cost_to_correct("t1", attempts)

    assert result.resolved is False
    assert result.iterations_to_correct is None
