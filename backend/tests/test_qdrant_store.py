
from __future__ import annotations

from dataclasses import dataclass

import pytest
from qdrant_client import models

from backend.app.embeddings.service import EmbeddedChunk
from backend.app.ingestion.models.chunk_document import ChunkDocument
from backend.app.retrieval.qdrant_store import QdrantVectorStore


# ============================================================
# Fake Qdrant response objects
# ============================================================


@dataclass
class FakePoint:
    id: str
    payload: dict
    score: float = 0.0
    vector: object | None = None


@dataclass
class FakeQueryResponse:
    points: list[FakePoint]


# ============================================================
# Fake Qdrant client
# ============================================================


class FakeQdrantClient:
    def __init__(self) -> None:
        self.collections: dict[str, dict[str, object]] = {}
        self.upserted_points: list[models.PointStruct] = []
        self.updated_vectors: list[models.PointVectors] = []

    async def collection_exists(
        self,
        collection_name: str,
    ) -> bool:
        return collection_name in self.collections

    async def create_collection(
        self,
        collection_name: str,
        vectors_config,
        sparse_vectors_config=None,
    ) -> None:
        self.collections[collection_name] = {
            "size": vectors_config.size,
            "sparse_vectors": (
                sparse_vectors_config or {}
            ),
            "points": {},
        }

    async def get_collection(
        self,
        collection_name: str,
    ):
        collection = self.collections[collection_name]

        size = collection["size"]

        vectors = type(
            "Vectors",
            (),
            {
                "size": size,
            },
        )()

        params = type(
            "Params",
            (),
            {
                "vectors": vectors,
                "sparse_vectors": collection.get(
                    "sparse_vectors",
                    {},
                ),
            },
        )()

        config = type(
            "Config",
            (),
            {
                "params": params,
            },
        )()

        return type(
            "CollectionInfo",
            (),
            {
                "config": config,
            },
        )()

    async def upsert(
        self,
        collection_name: str,
        points,
        wait: bool = True,
    ) -> None:
        self.upserted_points.extend(points)

        collection = self.collections[
            collection_name
        ]

        stored_points = collection.setdefault(
            "points",
            {},
        )

        for point in points:
            stored_points[point.id] = {
                "vector": point.vector,
                "payload": point.payload,
            }

    async def scroll(
        self,
        collection_name: str,
        limit: int,
        offset=None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ):
        points = []

        for point_id, point in self.collections[
            collection_name
        ].get("points", {}).items():

            points.append(
                FakePoint(
                    id=point_id,
                    payload=point["payload"],
                )
            )

        return points[:limit], None

    async def update_vectors(
        self,
        collection_name: str,
        points,
        wait: bool = True,
    ) -> None:
        self.updated_vectors.extend(points)

        collection = self.collections[
            collection_name
        ]

        stored_points = collection.setdefault(
            "points",
            {},
        )

        for point in points:
            existing = stored_points.setdefault(
                point.id,
                {
                    "vector": None,
                    "payload": {},
                },
            )

            vector = existing["vector"]

            if not isinstance(vector, dict):
                vector = {
                    "": vector,
                }

            for name, value in point.vector.items():
                vector[name] = value

            existing["vector"] = vector

    async def query_points(
        self,
        collection_name: str,
        query,
        limit: int,
        with_payload: bool = True,
        query_filter = None
    ) -> FakeQueryResponse:
        return FakeQueryResponse(
            points=[
                FakePoint(
                    id="point-1",
                    payload={
                        "chunk_id": "chunk-1",
                        "document_id": "doc-1",
                        "chunk_index": 0,
                        "content": "What is RAG?",
                        "metadata": {
                            "page_number": 1,
                        },
                    },
                    score=0.92,
                )
            ][:limit]
        )


# ============================================================
# Helpers
# ============================================================


def make_chunk() -> ChunkDocument:
    return ChunkDocument(
        content="Retrieval augmented generation uses retrieval.",
        document_id="doc-001",
        chunk_index=0,
        metadata={
            "page_number": 1,
        },
    )


def make_embedded_chunk() -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=make_chunk(),
        embedding=[1.0, 0.0, 0.0],
    )


# ============================================================
# Tests
# ============================================================


@pytest.mark.asyncio
async def test_ensure_collection_creates_collection():
    client = FakeQdrantClient()

    store = QdrantVectorStore(
        client=client,
        collection_name="test_collection",
    )

    await store.ensure_collection(
        vector_size=3,
    )

    assert (
        "test_collection"
        in client.collections
    )

    collection = client.collections[
        "test_collection"
    ]

    assert collection["size"] == 3

    sparse_vectors = collection[
        "sparse_vectors"
    ]

    assert "bm25" in sparse_vectors


@pytest.mark.asyncio
async def test_existing_collection_dimension_must_match():
    client = FakeQdrantClient()

    store = QdrantVectorStore(
        client=client,
        collection_name="test_collection",
    )

    await store.ensure_collection(
        vector_size=3,
    )

    with pytest.raises(
        ValueError,
        match="different vector dimension",
    ):
        await store.ensure_collection(
            vector_size=4,
        )


@pytest.mark.asyncio
async def test_upsert_stores_embedded_chunk():
    client = FakeQdrantClient()

    store = QdrantVectorStore(
        client=client,
        collection_name="test_collection",
    )

    embedded_chunk = make_embedded_chunk()

    await store.upsert(
        [embedded_chunk]
    )

    assert len(
        client.upserted_points
    ) == 1

    point = client.upserted_points[0]

    assert point.vector == [
        1.0,
        0.0,
        0.0,
    ]

    assert (
        point.payload["chunk_id"]
        == embedded_chunk.chunk.chunk_id
    )

    assert (
        point.payload["document_id"]
        == "doc-001"
    )

    assert (
        point.payload["chunk_index"]
        == 0
    )

    assert (
        point.payload["content"]
        == embedded_chunk.chunk.content
    )


@pytest.mark.asyncio
async def test_upsert_rejects_inconsistent_dimensions():
    client = FakeQdrantClient()

    store = QdrantVectorStore(
        client=client,
        collection_name="test_collection",
    )

    chunk_one = EmbeddedChunk(
        chunk=make_chunk(),
        embedding=[1.0, 0.0, 0.0],
    )

    chunk_two = EmbeddedChunk(
        chunk=ChunkDocument(
            content="Another chunk",
            document_id="doc-001",
            chunk_index=1,
        ),
        embedding=[1.0, 0.0],
    )

    with pytest.raises(
        ValueError,
        match="inconsistent dimensions",
    ):
        await store.upsert(
            [
                chunk_one,
                chunk_two,
            ]
        )


@pytest.mark.asyncio
async def test_search_returns_results():
    client = FakeQdrantClient()

    store = QdrantVectorStore(
        client=client,
        collection_name="test_collection",
    )

    results = await store.search(
        query_embedding=[
            1.0,
            0.0,
            0.0,
        ],
        top_k=5,
    )

    assert len(results) == 1

    result = results[0]

    assert result.chunk_id == "chunk-1"
    assert result.document_id == "doc-1"
    assert result.chunk_index == 0
    assert result.content == "What is RAG?"
    assert result.score == 0.92

    assert result.metadata == {
        "page_number": 1,
    }


@pytest.mark.asyncio
async def test_search_rejects_empty_query():
    client = FakeQdrantClient()

    store = QdrantVectorStore(
        client=client,
        collection_name="test_collection",
    )

    with pytest.raises(
        ValueError,
        match="Query embedding cannot be empty",
    ):
        await store.search(
            query_embedding=[],
            top_k=5,
        )


@pytest.mark.asyncio
async def test_search_rejects_invalid_top_k():
    client = FakeQdrantClient()

    store = QdrantVectorStore(
        client=client,
        collection_name="test_collection",
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        await store.search(
            query_embedding=[
                1.0,
                0.0,
                0.0,
            ],
            top_k=0,
        )


@pytest.mark.asyncio
async def test_upsert_sparse_updates_bm25_vector():
    client = FakeQdrantClient()

    store = QdrantVectorStore(
        client=client,
        collection_name="test_collection",
    )

    chunk = make_chunk()

    # Create the collection and an existing dense point.
    await store.ensure_collection(
        vector_size=3,
    )

    await store.upsert(
        [
            EmbeddedChunk(
                chunk=chunk,
                embedding=[
                    1.0,
                    0.0,
                    0.0,
                ],
            )
        ]
    )

    await store.upsert_sparse(
        [chunk]
    )

    assert len(
        client.updated_vectors
    ) == 1

    point_id = store._point_id(
        chunk.chunk_id
    )

    stored_point = client.collections[
        "test_collection"
    ]["points"][point_id]

    vectors = stored_point["vector"]

    assert "bm25" in vectors


@pytest.mark.asyncio
async def test_backfill_bm25_updates_existing_points():
    client = FakeQdrantClient()

    store = QdrantVectorStore(
        client=client,
        collection_name="test_collection",
    )

    chunk = make_chunk()

    await store.ensure_collection(
        vector_size=3,
    )

    await store.upsert(
        [
            EmbeddedChunk(
                chunk=chunk,
                embedding=[
                    1.0,
                    0.0,
                    0.0,
                ],
            )
        ]
    )

    updated = await store.backfill_bm25(
        batch_size=100,
    )

    assert updated == 1

    point_id = store._point_id(
        chunk.chunk_id
    )

    stored_point = client.collections[
        "test_collection"
    ]["points"][point_id]

    vectors = stored_point["vector"]

    assert "" in vectors
    assert "bm25" in vectors
