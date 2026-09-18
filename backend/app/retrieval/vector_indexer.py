from __future__ import annotations

import asyncio

from backend.app.embeddings.service import EmbeddingService
from backend.app.ingestion.models.chunk_document import ChunkDocument
from backend.app.retrieval.base import (
    SparseVectorStore,
    VectorStore,
)

from backend.app.retrieval.base import VectorIndexStore

class VectorIndexer:
    """
    Coordinates chunk embedding and vector-store indexing.

    Flow:

        ChunkDocument[]
              ↓
        EmbeddingService
              ↓
        EmbeddedChunk[]
              ↓
        VectorStore
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorIndexStore,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    async def index(
        self,
        chunks: list[ChunkDocument],
    ) -> None:
        if not chunks:
            return

        embedded_chunks = await asyncio.to_thread(
            self.embedding_service.embed_chunks,
            chunks,
        )

        await self.vector_store.upsert(
            embedded_chunks,
        )

        await self.vector_store.upsert_sparse(
            chunks,
        )
