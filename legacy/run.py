"""The loop over tasks x arms x models, driven entirely by configs/.

configs/ is the fairness lock: budget_tokens, code_search_tool, and
retry_policy are read from configs/common.json identically for every cell,
never chosen here. This file only expands configs/loader.py's RunCells into
task lists, memory arms, and LLM clients, and calls the existing
agent/runner.py loop -- it makes no scoring or table decisions; those live in
scoring/tables.py as groupbys over the JSONL this writes.

Supersedes the earlier one-off run_ablation.py: the 4-arm ablation is no
longer a separate script. It's now just one of scoring/tables.py's groupbys
over the same results this loop produces (see scoring/tables.py's
ablation_table).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from adapters.brief import BriefMemorySystem
from adapters.fullcontext import FullContextMemorySystem
from adapters.mem0 import Mem0MemorySystem
from adapters.none import NoneMemorySystem
from adapters.random_context import RandomContextMemorySystem
from agent.llm_client import AnthropicLLMClient, LLMClient
from agent.runner import RunConfig, run_and_log
from benchmarks.dcbench import load_dcbench_tasks
from benchmarks.longmemeval import load_longmemeval_tasks
from benchmarks.swebench import load_swebench_tasks
from benchmarks.task import Task
from configs.loader import RunCell, load_run_cells

ARM_FACTORIES = {
    "none": NoneMemorySystem,
    "fullcontext": FullContextMemorySystem,
    "mem0": Mem0MemorySystem,
    "brief": BriefMemorySystem,
    "random_context": RandomContextMemorySystem,
}


def _tasks_for(cell: RunCell) -> list[Task]:
    if cell.dataset == "dcbench":
        return load_dcbench_tasks(spec_variant=cell.spec_variant, depth=cell.depth)
    if cell.dataset == "swebench":
        return load_swebench_tasks(spec_variant=cell.spec_variant, depth=cell.depth)
    if cell.dataset == "longmemeval":
        # Not config-dialed: depth comes from the dataset's own multi-hop
        # structure per question, so cell.spec_variant/cell.depth are inert.
        return load_longmemeval_tasks()
    raise ValueError(f"unknown dataset {cell.dataset!r}")


def _llm_for(cell: RunCell, llm_overrides: dict) -> LLMClient:
    if cell.model_name in llm_overrides:
        return llm_overrides[cell.model_name]
    if cell.model_backend == "anthropic":
        return AnthropicLLMClient(model=cell.model_id)
    raise NotImplementedError(
        f"backend {cell.model_backend!r} ({cell.model_name}) has no LLMClient "
        "implementation yet -- see configs/models.json"
    )


def main(
    output_path: Optional[Path] = None, llm_overrides: Optional[dict] = None
) -> Path:
    """llm_overrides lets a caller inject a test LLMClient keyed by
    configs/models.json model name (e.g. {"claude": StubLLMClient()})
    without touching real backend selection above -- for tests/demos only;
    a real run passes nothing and uses AnthropicLLMClient."""
    llm_overrides = llm_overrides or {}
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = output_path or (results_dir / "all_runs.jsonl")
    output_path.write_text("")

    for cell in load_run_cells():
        if not cell.model_implemented and cell.model_name not in llm_overrides:
            print(
                f"skip {cell.dataset}/{cell.arm}/{cell.model_name}: backend "
                f"{cell.model_backend!r} not implemented (configs/models.json)"
            )
            continue

        memory_factory = ARM_FACTORIES[cell.arm]
        try:
            memory_factory()  # probe construction: e.g. brief needs BRIEF_OAUTH_TOKEN
        except RuntimeError as e:
            print(f"skip {cell.dataset}/{cell.arm}/{cell.model_name}: {e}")
            continue

        tasks = _tasks_for(cell)
        llm = _llm_for(cell, llm_overrides)
        config = RunConfig(
            arm_name=cell.arm, budget_tokens=cell.budget_tokens, retry_policy=cell.retry_policy
        )
        run_and_log(tasks, memory_factory, llm, config, output_path)

    return output_path


if __name__ == "__main__":
    path = main()
    print(f"wrote results to {path}")
