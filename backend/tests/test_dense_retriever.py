from __future__ import annotations

from dataclasses import dataclass

import pytest

from backend.app.retrieval.base import VectorSearchResult
from backend.app.retrieval.dense_retriever import (
    DenseRetriever,
)


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.received_query = None

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        self.received_query = query
        return [1.0, 0.0, 0.0]


class FakeVectorStore:
    def __init__(self) -> None:
        self.received_query_embedding = None
        self.received_top_k = None

    async def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[VectorSearchResult]:
        self.received_query_embedding = query_embedding
        self.received_top_k = top_k

        return [
            VectorSearchResult(
                chunk_id="chunk_001",
                document_id="doc_001",
                chunk_index=0,
                content="...",
                score=0.98,
                metadata={},
            )
        ]


@pytest.mark.asyncio
async def test_dense_retriever_connects_embedding_and_vector_store():
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()

    retriever = DenseRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    results = await retriever.retrieve(
        query="What is RAG?",
        top_k=5,
    )

    assert embedding_service.received_query == (
        "What is RAG?"
    )

    assert vector_store.received_query_embedding == [
        1.0,
        0.0,
        0.0,
    ]

    assert vector_store.received_top_k == 5

    assert len(results) == 1

    assert results[0].chunk_id == "chunk_001"
    assert results[0].score == 0.98


@pytest.mark.asyncio
async def test_dense_retriever_rejects_empty_query():
    retriever = DenseRetriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(ValueError):
        await retriever.retrieve(
            query="",
            top_k=5,
        )


@pytest.mark.asyncio
async def test_dense_retriever_rejects_invalid_top_k():
    retriever = DenseRetriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(ValueError):
        await retriever.retrieve(
            query="What is RAG?",
            top_k=0,
        )