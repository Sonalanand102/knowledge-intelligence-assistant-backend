from __future__ import annotations

import uuid

import pytest
from qdrant_client import models

from backend.app.ingestion.models.chunk_document import ChunkDocument
from backend.app.retrieval.qdrant import create_qdrant_client
from backend.app.retrieval.qdrant_store import QdrantVectorStore


@pytest.mark.asyncio
async def test_upsert_sparse_preserves_dense_vector():
    client = create_qdrant_client()

    collection_name = f"test-sparse-{uuid.uuid4().hex}"

    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )

    chunk = ChunkDocument(
        content="Retrieval augmented generation uses retrieval and language models.",
        document_id="doc-1",
        chunk_index=0,
        metadata={},
    )

    point_id = store._point_id(chunk.chunk_id)

    try:
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=3,
                distance=models.Distance.COSINE,
            ),
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )

        await client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=[1.0, 0.0, 0.0],
                    payload={
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                    },
                )
            ],
        )

        await store.upsert_sparse([chunk])

        points = await client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_vectors=True,
        )

        assert len(points) == 1

        vectors = points[0].vector

        assert vectors is not None
        assert "bm25" in vectors
        assert vectors[""] == [1.0, 0.0, 0.0]

    finally:
        await client.delete_collection(
            collection_name=collection_name
        )
        await client.close()