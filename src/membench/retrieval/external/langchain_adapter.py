"""Adapter for a LangChain FAISS vector retriever.

The canonical "roll your own RAG" stack: a LangChain ``FAISS`` vector store over
the corpus with similarity search. It is deterministic and local (given a local
embedding), representing the do-it-yourself baseline a team reaches for before
buying a memory product. We store each chunk's ``item_id`` in document metadata.

Requires the isolated ``langchain``/``faiss-cpu`` env
(``competitors/envs/langchain.txt``); the injected-client path is unit-tested
without it.
"""

from __future__ import annotations

from typing import Any

from membench.retrieval.base import ArmFamily, register_arm
from membench.retrieval.external._base import KeyedExternalAdapter

__all__ = ["LangChainVectorMemory"]


class _RealLangChainFAISS:
    """Wrap a LangChain FAISS store to the internal add/search contract."""

    def __init__(self, embeddings: Any | None = None) -> None:
        try:
            from langchain_community.vectorstores import FAISS  # noqa: F401
            from langchain_core.documents import Document  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "LangChain adapter needs langchain-community + faiss-cpu in an isolated env: "
                "uv pip install -r competitors/envs/langchain.txt"
            ) from exc
        if embeddings is None:
            from langchain_community.embeddings import FakeEmbeddings

            embeddings = FakeEmbeddings(size=256)
        self._embeddings = embeddings
        self._docs: list[Any] = []
        self._store: Any | None = None

    def add(self, item_id: str, text: str) -> None:
        from langchain_core.documents import Document

        self._docs.append(Document(page_content=text, metadata={"item_id": item_id}))
        self._store = None  # invalidate; rebuilt at search

    def search(self, query: str, k: int) -> list[str]:
        from langchain_community.vectorstores import FAISS

        if self._store is None:
            self._store = FAISS.from_documents(self._docs, self._embeddings)
        hits = self._store.similarity_search(query, k=k)
        return [str(hit.metadata.get("item_id")) for hit in hits if hit.metadata.get("item_id")]


@register_arm(
    "langchain_vec",
    family=ArmFamily.EXTERNAL,
    follows_links=False,
    uses_embeddings=True,
    requires_network=False,
    deterministic=True,
)
class LangChainVectorMemory(KeyedExternalAdapter):
    """LangChain FAISS vector-retriever memory arm.

    Parameters
    ----------
    client
        Object exposing ``add(item_id, text)`` and ``search(query, k) -> list[str]``;
        defaults to a lazily-constructed real FAISS wrapper.
    """

    def __init__(self, client: Any | None = None) -> None:
        super().__init__()
        self._client = client

    def _ensure(self) -> Any:
        if self._client is None:
            self._client = _RealLangChainFAISS()
        return self._client

    def _backend_add(self, item_id: str, text: str) -> str:
        self._ensure().add(item_id, text)
        return item_id

    def _backend_search(self, query: str, k: int) -> list[str]:
        results: list[str] = self._ensure().search(query, k)
        return results
