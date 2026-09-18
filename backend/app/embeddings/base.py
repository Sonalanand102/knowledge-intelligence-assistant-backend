from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Common contract for retrieval-oriented embedding providers.
    """

    def embed_document(self, text: str) -> list[float]:
        """
        Generate an embedding for a single document/chunk.
        """
        ...

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple documents/chunks.
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """
        Generate an embedding for a single search query.
        """
        ...

    def embed_queries(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple search queries.
        """
        ...