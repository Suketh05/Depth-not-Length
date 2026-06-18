"""The skeleton loop: build an arm, write() its corpus, retrieve(), call the
model, score, log one JSONL row per attempt.

Scoring is the real scoring/ package: compliance always runs (it's defined
whenever a task has governing_decisions); correctness runs additionally for
coding tasks (repo_ref is not None) since "merge-ready" only means something
when there's a repo to merge into. cost.py's retrieval precision/recall and
cost-to-correct are task-level, so they're computed once per task and
denormalized onto every attempt row for that task -- this keeps "one JSONL
row per attempt" the only row shape, so the four output tables stay plain
groupbys with no special-casing.

retry_policy is a config flag, default single_shot, per CLAUDE.md's standing
instruction not to optimize per-call tokens. Per-call tokens (input_tokens,
output_tokens, dollars) are logged as plain reported fields on every row;
the only thing that gates a retry is the `correct` flag, never token counts.
"""

import dataclasses
import json
import time
from pathlib import Path
from typing import Callable

from adapters.base import MemorySystem
from agent.llm_client import LLMClient
from benchmarks.task import Task
from scoring.compliance import score_compliance
from scoring.correctness import score_correctness
from scoring.cost import AttemptCost, score_cost_to_correct, score_retrieval

RETRY_POLICIES = ("single_shot", "retry_until_correct")


@dataclasses.dataclass
class RunConfig:
    arm_name: str
    budget_tokens: int
    retry_policy: str = "single_shot"
    max_attempts: int = 3  # only consulted when retry_policy == retry_until_correct


def run_task(
    task: Task, memory: MemorySystem, llm: LLMClient, config: RunConfig
) -> list[dict]:
    """Run one task through one arm. Returns the JSONL rows for this task."""
    if config.retry_policy not in RETRY_POLICIES:
        raise ValueError(f"unknown retry_policy {config.retry_policy!r}")

    memory.write(task.memory_corpus)
    retrieved = memory.retrieve(task.query, config.budget_tokens)
    retrieval = score_retrieval(retrieved.ids, task.governing_decisions)

    system = "You are a coding agent. Honor the team decisions in the provided context."
    user = f"{retrieved.text}\n\n{task.query}" if retrieved.text else task.query

    max_attempts = (
        config.max_attempts if config.retry_policy == "retry_until_correct" else 1
    )
    rows = []
    attempt_costs = []
    for attempt in range(1, max_attempts + 1):
        response = llm.complete(system=system, user=user, max_tokens=1024)
        compliance = score_compliance(response.text, task)
        correctness = (
            score_correctness(response.text, task, compliance)
            if task.repo_ref is not None
            else None
        )
        is_correct = correctness.merge_ready if correctness else compliance.rate == 1.0
        attempt_costs.append(
            AttemptCost(
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                dollars=response.dollars,
                correct=is_correct,
            )
        )
        rows.append(
            {
                "task_id": task.task_id,
                "dataset": task.dataset,
                "arm": config.arm_name,
                "model": response.model,
                "attempt": attempt,
                "depth": task.depth,
                "spec_variant": task.spec_variant,
                "retrieved_ids": retrieved.ids,
                "governing_decisions": task.governing_decisions,
                "compliance_rate": compliance.rate,
                "compliance_honored": compliance.honored,
                "compliance_total": compliance.total,
                "compliance_honored_ids": compliance.honored_ids,
                "correctness_merge_ready": correctness.merge_ready if correctness else None,
                "correctness_reason": correctness.reason if correctness else None,
                "correct": is_correct,
                # Per-call tokens: reported only, never an optimization target.
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "dollars": response.dollars,
                "response_text": response.text,
                "timestamp": time.time(),
            }
        )
        if config.retry_policy == "single_shot" or is_correct:
            break

    cost_to_correct = score_cost_to_correct(task.task_id, attempt_costs)
    for row in rows:
        row.update(
            {
                "retrieval_precision": retrieval.precision,
                "retrieval_recall": retrieval.recall,
                "retrieval_true_positives": retrieval.true_positives,
                "retrieval_gold_count": retrieval.gold,
                "retrieval_retrieved_count": retrieval.retrieved,
                "cost_attempts": cost_to_correct.attempts,
                "cost_resolved": cost_to_correct.resolved,
                "cost_iterations_to_correct": cost_to_correct.iterations_to_correct,
                "cost_total_input_tokens": cost_to_correct.total_input_tokens,
                "cost_total_output_tokens": cost_to_correct.total_output_tokens,
                "cost_total_dollars": cost_to_correct.total_dollars,
            }
        )
    return rows


def run_and_log(
    tasks: list[Task],
    memory_factory: Callable[[], MemorySystem],
    llm: LLMClient,
    config: RunConfig,
    output_path: Path,
) -> None:
    """memory_factory must return a fresh MemorySystem per task, so one
    task's corpus never leaks into another's retrieval."""
    with open(output_path, "a") as f:
        for task in tasks:
            memory = memory_factory()
            for row in run_task(task, memory, llm, config):
                f.write(json.dumps(row) + "\n")
