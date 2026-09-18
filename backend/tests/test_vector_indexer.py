from __future__ import annotations

import pytest

from backend.app.embeddings.service import EmbeddedChunk
from backend.app.ingestion.models.chunk_document import ChunkDocument
from backend.app.retrieval.vector_indexer import VectorIndexer


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.received_chunks = None

    def embed_chunks(
        self,
        chunks: list[ChunkDocument],
    ) -> list[EmbeddedChunk]:
        self.received_chunks = chunks

        return [
            EmbeddedChunk(
                chunk=chunk,
                embedding=[1.0, 0.0, 0.0],
            )
            for chunk in chunks
        ]


class FakeVectorStore:
    def __init__(self) -> None:
        self.received_chunks = None
        self.received_sparse_chunks = None

    async def upsert(
        self,
        embedded_chunks: list[EmbeddedChunk],
    ) -> None:
        self.received_chunks = embedded_chunks

    async def upsert_sparse(
        self,
        chunks: list[ChunkDocument],
    ) -> None:
        self.received_sparse_chunks = chunks

@pytest.mark.asyncio
async def test_index_embeds_and_stores_chunks():
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()

    indexer = VectorIndexer(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    chunks = [
        ChunkDocument(
            content="What is RAG?",
            document_id="doc-001",
            chunk_index=0,
        ),
        ChunkDocument(
            content="What are embeddings?",
            document_id="doc-001",
            chunk_index=1,
        ),
    ]

    await indexer.index(chunks)

    assert embedding_service.received_chunks == chunks

    assert vector_store.received_chunks is not None

    assert len(
        vector_store.received_chunks
    ) == 2

    assert (
        vector_store.received_chunks[0].chunk
        == chunks[0]
    )

    assert (
        vector_store.received_chunks[0].embedding
        == [1.0, 0.0, 0.0]
    )

    assert (
        vector_store.received_sparse_chunks
        == chunks
    )


@pytest.mark.asyncio
async def test_index_empty_chunks_does_nothing():
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()

    indexer = VectorIndexer(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    await indexer.index([])

    assert (
        embedding_service.received_chunks is None
    )

    assert (
        vector_store.received_chunks is None
    )

    assert (
        vector_store.received_sparse_chunks is None
    )