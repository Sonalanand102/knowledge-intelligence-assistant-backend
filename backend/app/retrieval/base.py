from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from backend.app.embeddings.service import EmbeddedChunk


@dataclass(frozen=True)
class VectorSearchResult:
    """
    A single result returned by a vector store.

    Carries enough information for downstream retrieval,
    provenance, and citation handling.
    """

    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    async def ensure_collection(
        self,
        vector_size: int,
    ) -> None:
        ...

    async def upsert(
        self,
        embedded_chunks: list[EmbeddedChunk],
    ) -> None:
        ...

    async def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[VectorSearchResult]:
        ...


from backend.app.ingestion.models.chunk_document import ChunkDocument


class SparseVectorStore(Protocol):
    async def upsert_sparse(
        self,
        chunks: list[ChunkDocument],
    ) -> None:
        ...

    async def search_sparse(
        self,
        query: str,
        top_k: int,
    ) -> list[VectorSearchResult]:
        ...

class VectorIndexStore(VectorStore, SparseVectorStore, Protocol):
    pass