"""The task runner: wire a memory arm to a model and emit one record per attempt.

The runner is deliberately metrics-free. It writes the corpus, retrieves under the
fixed budget, prompts the model, and records the raw outcome — including the
retrieved ids and the gold governing decisions — as an :class:`AttemptRecord`.
Scoring (compliance, retrieval quality, cost) is computed *afterwards* over these
records, so the runner has no dependency on the metrics layer and the record is the
single canonical row every table groups over.

Retry policy is a flag (default ``single_shot``, per the standing instruction not
to optimise per-call tokens). ``retry_until_correct`` consults a caller-supplied
correctness predicate; per-call tokens are reported, never a retry gate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from membench.agents.llm.base import LLMClient
from membench.retrieval.base import MemorySystem
from membench.types import Task

__all__ = ["AttemptRecord", "RunConfig", "run_task"]

_SYSTEM_PROMPT = "You are a coding agent. Honour the team decisions in the provided context."
RETRY_POLICIES = ("single_shot", "retry_until_correct")


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Per-run knobs shared identically across arms (the fairness lock at runtime)."""

    budget_tokens: int
    retry_policy: str = "single_shot"
    max_attempts: int = 3
    max_output_tokens: int = 1024

    def __post_init__(self) -> None:
        if self.retry_policy not in RETRY_POLICIES:
            raise ValueError(f"unknown retry_policy {self.retry_policy!r}")
        if self.budget_tokens < 0:
            raise ValueError("budget_tokens must be non-negative")


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """One attempt's raw outcome — the canonical result row scoring reads."""

    task_id: str
    dataset: str
    arm: str
    model: str
    attempt: int
    depth: int
    spec_variant: str
    scorer: str
    retrieved_ids: tuple[str, ...]
    governing_decisions: tuple[str, ...]
    response_text: str
    input_tokens: int
    output_tokens: int
    dollars: float
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable row (tuples flattened to lists)."""
        row = asdict(self)
        row["retrieved_ids"] = list(self.retrieved_ids)
        row["governing_decisions"] = list(self.governing_decisions)
        return row


def run_task(
    task: Task,
    memory: MemorySystem,
    llm: LLMClient,
    config: RunConfig,
    *,
    arm_name: str,
    is_correct: Callable[[str, Task], bool] | None = None,
) -> list[AttemptRecord]:
    """Run one task through one (arm, model) pair and return its attempt records.

    Parameters
    ----------
    task
        The evaluation unit.
    memory
        A fresh memory arm (its corpus must not be shared across tasks).
    llm
        The language-model client.
    config
        Runtime configuration (budget, retry policy).
    arm_name
        Registry name of the arm, recorded on each row.
    is_correct
        Correctness predicate consulted only under ``retry_until_correct``. If
        omitted there, the runner makes a single attempt.
    """
    memory.write(task.memory_corpus)
    retrieved = memory.retrieve(task.query, config.budget_tokens)
    user = f"{retrieved.text}\n\n{task.query}" if retrieved.text else task.query

    retrying = config.retry_policy == "retry_until_correct" and is_correct is not None
    max_attempts = config.max_attempts if retrying else 1

    records: list[AttemptRecord] = []
    for attempt in range(1, max_attempts + 1):
        response = llm.complete(_SYSTEM_PROMPT, user, config.max_output_tokens)
        records.append(
            AttemptRecord(
                task_id=task.task_id,
                dataset=task.dataset,
                arm=arm_name,
                model=response.model,
                attempt=attempt,
                depth=task.depth,
                spec_variant=task.spec_variant.value,
                scorer=task.scorer.value,
                retrieved_ids=retrieved.ids,
                governing_decisions=task.governing_decisions,
                response_text=response.text,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                dollars=response.dollars,
            )
        )
        if not retrying or (is_correct is not None and is_correct(response.text, task)):
            break
    return records
