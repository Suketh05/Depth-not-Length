"""Cost-to-correct (tokens + dollars, summed over attempts), retrieval
precision/recall against the gold governing_decisions, and
iterations-to-correct.

Per-call tokens are a reported field only -- this module measures cost,
it never decides whether to retry. Whether an attempt counts as "correct"
is decided upstream (by compliance/correctness scoring) and passed in;
cost.py just sums and reports.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalScore:
    precision: float
    recall: float
    retrieved: int
    gold: int
    true_positives: list[str]


def score_retrieval(
    retrieved_ids: list[str], governing_decisions: list[str]
) -> RetrievalScore:
    gold = set(governing_decisions)
    retrieved = set(retrieved_ids)
    true_positives = sorted(gold & retrieved)
    # Convention: precision on an empty retrieval is 1.0 only if there was
    # nothing to retrieve (vacuously correct); recall on no gold IDs is
    # trivially 1.0 since there's nothing to miss.
    precision = len(true_positives) / len(retrieved) if retrieved else (1.0 if not gold else 0.0)
    recall = len(true_positives) / len(gold) if gold else 1.0
    return RetrievalScore(
        precision=precision,
        recall=recall,
        retrieved=len(retrieved_ids),
        gold=len(governing_decisions),
        true_positives=true_positives,
    )


@dataclass
class AttemptCost:
    input_tokens: int
    output_tokens: int
    dollars: float
    correct: bool


@dataclass
class CostToCorrect:
    task_id: str
    attempts: int
    resolved: bool
    iterations_to_correct: int | None
    total_input_tokens: int
    total_output_tokens: int
    total_dollars: float


def score_cost_to_correct(task_id: str, attempts: list[AttemptCost]) -> CostToCorrect:
    iterations_to_correct = None
    resolved = False
    for index, attempt in enumerate(attempts, start=1):
        if attempt.correct:
            iterations_to_correct = index
            resolved = True
            break
    return CostToCorrect(
        task_id=task_id,
        attempts=len(attempts),
        resolved=resolved,
        iterations_to_correct=iterations_to_correct,
        total_input_tokens=sum(a.input_tokens for a in attempts),
        total_output_tokens=sum(a.output_tokens for a in attempts),
        total_dollars=sum(a.dollars for a in attempts),
    )
