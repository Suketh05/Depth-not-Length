from unittest.mock import AsyncMock, patch

import pytest

from adapters.brief import BriefMemorySystem
from benchmarks.task import MemoryItem


def test_requires_oauth_token_from_env(monkeypatch):
    monkeypatch.delenv("BRIEF_OAUTH_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        BriefMemorySystem()


def test_write_is_a_documented_noop(monkeypatch):
    monkeypatch.setenv("BRIEF_OAUTH_TOKEN", "test-token")
    mem = BriefMemorySystem()
    mem.write([MemoryItem(item_id="D-001", text="anything")])  # must not raise


def test_retrieve_returns_ids_of_walked_nodes_and_respects_budget(monkeypatch):
    monkeypatch.setenv("BRIEF_OAUTH_TOKEN", "test-token")
    mem = BriefMemorySystem()

    fake_matches = [
        {"id": "D-002", "snippet": "audit log decision"},
        {"id": "D-099", "snippet": "x" * 4000},  # too big to fit the budget
    ]
    with patch("adapters.brief._search_via_mcp", new=AsyncMock(return_value=fake_matches)):
        result = mem.retrieve("export endpoint", budget_tokens=50)

    assert result.ids == ["D-002"]
