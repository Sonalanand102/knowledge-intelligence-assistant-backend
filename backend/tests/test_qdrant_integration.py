from __future__ import annotations

import uuid

import pytest

from backend.app.embeddings.service import EmbeddedChunk
from backend.app.ingestion.models.chunk_document import ChunkDocument
from backend.app.retrieval.qdrant import create_qdrant_client
from backend.app.retrieval.qdrant_store import QdrantVectorStore


def make_embedded_chunk(
    document_id: str,
    content: str,
    embedding: list[float],
) -> EmbeddedChunk:
    chunk = ChunkDocument(
        content=content,
        document_id=document_id,
        chunk_index=0,
        metadata={
            "source": "integration-test",
            "test": True,
        },
    )

    return EmbeddedChunk(
        chunk=chunk,
        embedding=embedding,
    )


@pytest.mark.asyncio
async def test_qdrant_upsert_and_search():
    client = create_qdrant_client()

    collection_name = (
        f"test_knowledge_chunks_{uuid.uuid4().hex}"
    )

    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )

    try:
        embedded_chunks = [
            make_embedded_chunk(
                document_id="doc-rag",
                content=(
                    "Retrieval Augmented Generation "
                    "retrieves external context."
                ),
                embedding=[1.0, 0.0, 0.0],
            ),
            make_embedded_chunk(
                document_id="doc-embeddings",
                content=(
                    "Embeddings represent text "
                    "as numerical vectors."
                ),
                embedding=[0.0, 1.0, 0.0],
            ),
            make_embedded_chunk(
                document_id="doc-postgres",
                content=(
                    "PostgreSQL is a relational database."
                ),
                embedding=[0.0, 0.0, 1.0],
            ),
        ]

        await store.upsert(
            embedded_chunks
        )

        results = await store.search(
            query_embedding=[1.0, 0.0, 0.0],
            top_k=3,
        )

        assert len(results) == 3

        assert results[0].content == (
            "Retrieval Augmented Generation "
            "retrieves external context."
        )

        assert results[0].score > results[1].score
        assert results[0].score > results[2].score

        assert results[0].metadata["source"] == (
            "integration-test"
        )

        assert results[0].metadata["test"] is True

    finally:
        exists = await client.collection_exists(
            collection_name=collection_name,
        )

        if exists:
            await client.delete_collection(
                collection_name=collection_name,
            )

        await client.close()