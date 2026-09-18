from __future__ import annotations

import uuid

import pytest

from backend.app.embeddings.local import LocalEmbeddingProvider
from backend.app.embeddings.service import EmbeddingService
from backend.app.ingestion.models.chunk_document import ChunkDocument
from backend.app.retrieval.dense_retriever import DenseRetriever
from backend.app.retrieval.qdrant import create_qdrant_client
from backend.app.retrieval.qdrant_store import QdrantVectorStore


@pytest.mark.asyncio
async def test_dense_retrieval_end_to_end():
    client = create_qdrant_client()

    collection_name = (
        f"dense_retrieval_test_{uuid.uuid4().hex}"
    )

    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )

    embedding_service = EmbeddingService(
        provider=LocalEmbeddingProvider(),
    )

    retriever = DenseRetriever(
        embedding_service=embedding_service,
        vector_store=store,
    )

    chunks = [
        ChunkDocument(
            content=(
                "Retrieval Augmented Generation combines "
                "information retrieval with language model generation."
            ),
            document_id="rag-doc",
            chunk_index=0,
            metadata={
                "source": "test",
            },
        ),
        ChunkDocument(
            content=(
                "Redis is an in-memory data store commonly "
                "used for caching and low-latency workloads."
            ),
            document_id="redis-doc",
            chunk_index=0,
            metadata={
                "source": "test",
            },
        ),
        ChunkDocument(
            content=(
                "PostgreSQL is a relational database system "
                "that supports SQL queries and transactions."
            ),
            document_id="postgres-doc",
            chunk_index=0,
            metadata={
                "source": "test",
            },
        ),
        ChunkDocument(
            content=(
                "Docker packages applications and their "
                "dependencies into containers."
            ),
            document_id="docker-doc",
            chunk_index=0,
            metadata={
                "source": "test",
            },
        ),
    ]

    try:
        embedded_chunks = (
            embedding_service.embed_chunks(chunks)
        )

        await store.upsert(
            embedded_chunks
        )

        results = await retriever.retrieve(
            query=(
                "What is Retrieval Augmented Generation "
                "and how does it work?"
            ),
            top_k=3,
        )

        assert len(results) == 3

        result_chunk_ids = {
            result.chunk_id
            for result in results
        }

        rag_chunk_id = chunks[0].chunk_id

        assert rag_chunk_id in result_chunk_ids

        assert all(
            result.score >= 0.0
            for result in results
        )

        assert all(
            results[index].score
            >= results[index + 1].score
            for index in range(len(results) - 1)
        )

    finally:
        exists = await client.collection_exists(
            collection_name=collection_name,
        )

        if exists:
            await client.delete_collection(
                collection_name=collection_name,
            )

        await client.close()