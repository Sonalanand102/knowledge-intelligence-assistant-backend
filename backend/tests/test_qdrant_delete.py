from __future__ import annotations

import uuid

import pytest
from qdrant_client import models

from backend.app.core.config import settings
from backend.app.retrieval.qdrant import create_qdrant_client
from backend.app.retrieval.qdrant_store import QdrantVectorStore


@pytest.mark.asyncio
async def test_delete_by_document_id():
    client = create_qdrant_client()

    collection_name = f"test-delete-{uuid.uuid4().hex}"

    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )

    try:
        await store.ensure_collection(vector_size=3)

        await client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=[1.0, 0.0, 0.0],
                    payload={
                        "document_id": "doc-1",
                    },
                ),
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=[0.0, 1.0, 0.0],
                    payload={
                        "document_id": "doc-1",
                    },
                ),
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=[0.0, 0.0, 1.0],
                    payload={
                        "document_id": "doc-2",
                    },
                ),
            ],
        )

        await store.delete_by_document_id("doc-1")

        points, _ = await client.scroll(
            collection_name=collection_name,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )

        assert len(points) == 1
        assert points[0].payload["document_id"] == "doc-2"

    finally:
        await client.delete_collection(collection_name=collection_name)
        await client.close()