"""Shared fakes for external-adapter tests (no real third-party packages).

These implement the small internal client contract the adapters depend on
(``add(item_id, text)`` / ``search(query, k) -> list[item_id]``), ranking by
lexical overlap so the adapter's id-mapping and budget logic can be exercised
deterministically without a heavyweight, networked backend.
"""

from __future__ import annotations

import re

_TOK = re.compile(r"[a-z0-9]+")


class LexicalIdSearchClient:
    """A fake add/search store that returns our item_ids ranked by token overlap."""

    def __init__(self) -> None:
        self._docs: list[tuple[str, str]] = []

    def add(self, item_id: str, text: str) -> None:
        self._docs.append((item_id, text))

    def search(self, query: str, k: int) -> list[str]:
        q = set(_TOK.findall(query.lower()))

        def overlap(doc: tuple[str, str]) -> int:
            return len(q & set(_TOK.findall(doc[1].lower())))

        return [iid for iid, _ in sorted(self._docs, key=overlap, reverse=True)[:k]]
