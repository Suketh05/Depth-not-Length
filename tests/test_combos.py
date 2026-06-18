"""Tests for the model x memory combo matrix."""

from __future__ import annotations

import pytest

# Import arm modules so the registry is populated for build_system.
from membench.agents.combos import (
    NAMED_COMBOS,
    Combo,
    build_system,
    combo_grid,
    make_llm_client,
)
from membench.agents.llm.anthropic_client import AnthropicLLMClient
from membench.agents.llm.base import LLMClient
from membench.agents.llm.stub import StubLLMClient
from membench.retrieval import controls  # noqa: F401
from membench.retrieval.base import MemorySystem
from membench.retrieval.external import mem0_adapter  # noqa: F401
from membench.retrieval.graph import arm  # noqa: F401


class TestMakeLLMClient:
    def test_offline_returns_stub_priced_as_model(self) -> None:
        client = make_llm_client("claude", offline=True)
        assert isinstance(client, StubLLMClient)
        assert client.model_name == "claude-sonnet-4-6"

    def test_offline_gpt(self) -> None:
        assert make_llm_client("gpt", offline=True).model_name == "gpt-5.1"

    def test_live_anthropic(self) -> None:
        assert isinstance(make_llm_client("claude", offline=False), AnthropicLLMClient)

    def test_live_open_weight_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="no live backend"):
            make_llm_client("open_weight", offline=False)


class TestNamedCombos:
    def test_chatgpt_plus_brief_present(self) -> None:
        by_label = {c.label: c for c in NAMED_COMBOS}
        assert by_label["ChatGPT + Brief"].model == "gpt"
        assert by_label["ChatGPT + Brief"].arm == "brief_graph"
        assert by_label["ChatGPT alone"].arm == "none"

    def test_labels_unique(self) -> None:
        labels = [c.label for c in NAMED_COMBOS]
        assert len(labels) == len(set(labels))


class TestBuildSystem:
    def test_builds_llm_and_arm(self) -> None:
        llm, memory = build_system(Combo("Claude + Brief", "claude", "brief_graph"), offline=True)
        assert isinstance(llm, LLMClient)
        assert isinstance(memory, MemorySystem)

    def test_grid_is_product(self) -> None:
        grid = combo_grid(["claude", "gpt"], ["none", "brief_graph", "mem0"])
        assert len(grid) == 6
        assert grid[0].label == "claude+none"
