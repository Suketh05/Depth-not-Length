"""Tests for the optional live-Brief MCP arm.

Only offline behaviour is tested here (registration, credential handling, response
parsing). The networked path is exercised separately under the `live` marker with
a real token; it is not part of the reproducible suite.
"""

from __future__ import annotations

import pytest

from membench.retrieval.base import ArmFamily, build_arm, get_spec
from membench.retrieval.graph import live  # noqa: F401  (registration side effect)
from membench.retrieval.graph.live import BriefLiveMemory, _extract_matches


class TestRegistration:
    def test_registered_as_network_nondeterministic_graph(self) -> None:
        spec = get_spec("brief_live")
        assert spec.capabilities.family is ArmFamily.GRAPH
        assert spec.capabilities.requires_network is True
        assert spec.capabilities.deterministic is False
        assert spec.capabilities.follows_links is True


class TestCredentialHandling:
    def test_raises_without_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRIEF_OAUTH_TOKEN", raising=False)
        with pytest.raises(RuntimeError, match="OAuth token"):
            build_arm("brief_live")

    def test_reads_token_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BRIEF_OAUTH_TOKEN", "tok-123")
        arm = build_arm("brief_live")
        assert isinstance(arm, BriefLiveMemory)

    def test_write_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BRIEF_OAUTH_TOKEN", "tok-123")
        arm = build_arm("brief_live")
        arm.write([])  # type: ignore[attr-defined]  # must not raise


class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


class _Result:
    def __init__(self, structured: object = None, content: list[_Block] | None = None) -> None:
        self.structuredContent = structured
        self.content = content or []


class TestExtractMatches:
    def test_structured_results_field(self) -> None:
        res = _Result(structured={"results": [{"id": "a"}, {"id": "b"}]})
        assert [m["id"] for m in _extract_matches(res)] == ["a", "b"]

    def test_structured_list(self) -> None:
        assert _extract_matches(_Result(structured=[{"id": "x"}])) == [{"id": "x"}]

    def test_json_text_block(self) -> None:
        res = _Result(content=[_Block('{"results": [{"id": "z"}]}')])
        assert _extract_matches(res)[0]["id"] == "z"

    def test_garbage_text_block_ignored(self) -> None:
        assert _extract_matches(_Result(content=[_Block("not json")])) == []

    def test_empty(self) -> None:
        assert _extract_matches(_Result()) == []
