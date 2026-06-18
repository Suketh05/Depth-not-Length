"""Tests for the LLM client layer (stub determinism, pricing, accounting)."""

from __future__ import annotations

import pytest

from membench.agents.llm.anthropic_client import AnthropicLLMClient
from membench.agents.llm.openai_client import OpenAILLMClient
from membench.agents.llm.pricing import price_dollars
from membench.agents.llm.stub import StubLLMClient


class TestPricing:
    def test_known_model(self) -> None:
        # 1M input @ $3 + 1M output @ $15 = $18 for claude-sonnet-4-6
        assert price_dollars("claude-sonnet-4-6", 1_000_000, 1_000_000) == pytest.approx(18.0)

    def test_unknown_model_fails_loud(self) -> None:
        with pytest.raises(KeyError, match="no pricing entry"):
            price_dollars("not-a-model", 10, 10)


class TestStub:
    def test_honours_identifiers_in_context(self) -> None:
        client = StubLLMClient()
        resp = client.complete(
            system="You are a coding agent.",
            user="Add a CSV export. Context: must call withAuditLog and use DateRangePicker.",
        )
        assert "withAuditLog" in resp.text
        assert "DateRangePicker" in resp.text

    def test_no_identifiers(self) -> None:
        resp = StubLLMClient().complete(system="s", user="please add a thing")
        assert "No governing decisions" in resp.text

    def test_is_deterministic(self) -> None:
        a = StubLLMClient().complete("s", "use withAuditLog now")
        b = StubLLMClient().complete("s", "use withAuditLog now")
        assert (a.text, a.input_tokens, a.output_tokens, a.dollars) == (
            b.text,
            b.input_tokens,
            b.output_tokens,
            b.dollars,
        )

    def test_token_accounting_and_cost(self) -> None:
        resp = StubLLMClient().complete("system", "use withAuditLog")
        assert resp.input_tokens > 0 and resp.output_tokens > 0
        assert resp.dollars == pytest.approx(
            price_dollars("stub", resp.input_tokens, resp.output_tokens)
        )

    def test_more_context_costs_more(self) -> None:
        # The token-economics story in miniature: a bigger prompt costs more.
        small = StubLLMClient().complete("s", "use withAuditLog")
        big = StubLLMClient().complete("s", "use withAuditLog " + "filler text " * 200)
        assert big.input_tokens > small.input_tokens
        assert big.dollars > small.dollars


class TestRealClientConstruction:
    def test_anthropic_rejects_unknown_model(self) -> None:
        with pytest.raises(ValueError, match="no pricing entry"):
            AnthropicLLMClient(model="nope")

    def test_openai_rejects_unknown_model(self) -> None:
        with pytest.raises(ValueError, match="no pricing entry"):
            OpenAILLMClient(model="nope")

    def test_model_name_exposed(self) -> None:
        assert AnthropicLLMClient().model_name == "claude-sonnet-4-6"
        assert OpenAILLMClient().model_name == "gpt-5.1"
