"""Model x memory combinations: the systems the benchmark actually compares.

A benchmarked *system* is a base model paired with a memory arm. Holding the model
fixed and sweeping the arm gives the headline table; holding the arm fixed at Brief
and sweeping the model gives the robustness table; and specific named pairings give
the marketing-facing comparisons buyers ask for ("ChatGPT alone" vs "ChatGPT +
Brief", "Claude + Brief" vs "Claude + Mem0", ...).

The same combo runs offline (deterministic stub model, priced as the real model so
token-cost differences are still meaningful) or live (real Anthropic/OpenAI
backends), selected by one flag.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from membench.agents.llm.anthropic_client import AnthropicLLMClient
from membench.agents.llm.base import LLMClient
from membench.agents.llm.openai_client import OpenAILLMClient
from membench.agents.llm.stub import StubLLMClient
from membench.retrieval.base import MemorySystem, build_arm

__all__ = [
    "MODELS",
    "NAMED_COMBOS",
    "Combo",
    "ModelSpec",
    "build_system",
    "combo_grid",
    "make_llm_client",
]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A base model: its grid key, concrete id, backend, and implementation status."""

    name: str
    model_id: str
    backend: str
    implemented: bool


MODELS: dict[str, ModelSpec] = {
    "claude": ModelSpec("claude", "claude-sonnet-4-6", "anthropic", implemented=True),
    "gpt": ModelSpec("gpt", "gpt-5.1", "openai", implemented=True),
    "open_weight": ModelSpec("open_weight", "llama-4-maverick", "together", implemented=False),
}


def make_llm_client(model_name: str, *, offline: bool = True) -> LLMClient:
    """Build an LLM client for a model grid key.

    ``offline=True`` (default) returns the deterministic stub priced as the real
    model, so the pipeline runs free and reproducibly; ``offline=False`` returns the
    real backend.
    """
    spec = MODELS[model_name]
    if offline:
        return StubLLMClient(model=spec.model_id)
    if spec.backend == "anthropic":
        return AnthropicLLMClient(model=spec.model_id)
    if spec.backend == "openai":
        return OpenAILLMClient(model=spec.model_id)
    raise NotImplementedError(
        f"no live backend for model {model_name!r} (backend {spec.backend!r}); "
        "run offline or add the backend"
    )


@dataclass(frozen=True, slots=True)
class Combo:
    """A (model, memory arm) pairing with a human-facing label."""

    label: str
    model: str
    arm: str
    arm_kwargs: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))


# Curated, marketing-facing comparisons (a subset of the full grid).
NAMED_COMBOS: list[Combo] = [
    Combo("Claude alone", "claude", "none"),
    Combo("Claude + Brief", "claude", "brief_graph"),
    Combo("Claude + Mem0", "claude", "mem0"),
    Combo("Claude + plain RAG", "claude", "dense"),
    Combo("Claude + hybrid RAG", "claude", "hybrid_rrf"),
    Combo("Claude + Zep", "claude", "zep"),
    Combo("Claude + GraphRAG", "claude", "graphrag"),
    Combo("ChatGPT alone", "gpt", "none"),
    Combo("ChatGPT + Brief", "gpt", "brief_graph"),
    Combo("ChatGPT + Mem0", "gpt", "mem0"),
]


def combo_grid(models: list[str], arms: list[str]) -> list[Combo]:
    """Return the full model x arm grid as combos with generated labels."""
    return [Combo(f"{m}+{a}", m, a) for m in models for a in arms]


def build_system(combo: Combo, *, offline: bool = True) -> tuple[LLMClient, MemorySystem]:
    """Instantiate the (LLM client, memory arm) pair for a combo."""
    llm = make_llm_client(combo.model, offline=offline)
    arm = build_arm(combo.arm, **dict(combo.arm_kwargs))
    return llm, arm
