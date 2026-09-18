from __future__ import annotations

import uuid

from qdrant_client import AsyncQdrantClient, models

from backend.app.embeddings.service import EmbeddedChunk
from backend.app.ingestion.models.chunk_document import ChunkDocument
from backend.app.retrieval.base import VectorSearchResult



class QdrantVectorStore:
    BM25_VECTOR_NAME = "bm25"
    DEFAULT_COLLECTION_NAME = "knowledge_chunks"

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.client = client
        self.collection_name = collection_name

    async def ensure_collection(
        self,
        vector_size: int,
    ) -> None:
        if vector_size <= 0:
            raise ValueError(
                "Vector size must be greater than zero"
            )

        exists = await self.client.collection_exists(
            collection_name=self.collection_name,
        )

        if not exists:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
                sparse_vectors_config={
                    self.BM25_VECTOR_NAME: models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    ),
                },
            )
            return

        collection_info = await self.client.get_collection(
            collection_name=self.collection_name,
        )

        assert (
            "bm25"
            in collection_info.config.params.sparse_vectors
        )

        configured_size = (
            collection_info.config.params.vectors.size
        )

        if configured_size != vector_size:
            raise ValueError(
                "Existing Qdrant collection has a different "
                f"vector dimension: expected {vector_size}, "
                f"found {configured_size}"
            )

    async def upsert(
        self,
        embedded_chunks: list[EmbeddedChunk],
    ) -> None:
        if not embedded_chunks:
            return

        vector_sizes = {
            len(chunk.embedding)
            for chunk in embedded_chunks
        }

        if 0 in vector_sizes:
            raise ValueError(
                "Cannot upsert an empty embedding vector"
            )

        if len(vector_sizes) != 1:
            raise ValueError(
                "Embedding vectors have inconsistent dimensions"
            )

        vector_size = len(
            embedded_chunks[0].embedding
        )

        await self.ensure_collection(
            vector_size=vector_size,
        )

        points = [
            models.PointStruct(
                id=self._point_id(
                    embedded_chunk.chunk.chunk_id
                ),
                vector=embedded_chunk.embedding,
                payload={
                    "chunk_id": (
                        embedded_chunk.chunk.chunk_id
                    ),
                    "content": (
                        embedded_chunk.chunk.content
                    ),
                    "document_id": (
                        embedded_chunk.chunk.document_id
                    ),
                    "chunk_index": (
                        embedded_chunk.chunk.chunk_index
                    ),
                    "metadata": (
                        embedded_chunk.chunk.metadata
                    ),
                },
            )
            for embedded_chunk in embedded_chunks
        ]

        await self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    async def backfill_bm25(
        self,
        batch_size: int = 100,
    ) -> int:
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero"
            )

        updated_count = 0
        offset = None

        while True:
            records, next_offset = await self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            if not records:
                break

            points = []

            for record in records:
                payload = record.payload or {}

                content = payload.get("content")

                if content is None:
                    raise ValueError(
                        f"Qdrant point {record.id} is missing content"
                    )

                points.append(
                    models.PointVectors(
                        id=record.id,
                        vector={
                            self.BM25_VECTOR_NAME: (
                                models.Document(
                                    text=str(content),
                                    model="qdrant/bm25",
                                )
                            )
                        },
                    )
                )

            if points:
                await self.client.update_vectors(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )

                updated_count += len(points)

            offset = next_offset

            if offset is None:
                break

        return updated_count

    async def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[VectorSearchResult]:
        if not query_embedding:
            raise ValueError(
                "Query embedding cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        response = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )

        results: list[VectorSearchResult] = []

        for point in response.points:
            payload = point.payload or {}

            chunk_id = payload.get("chunk_id")
            document_id = payload.get("document_id")
            chunk_index = payload.get("chunk_index")
            content = payload.get("content")

            if chunk_id is None:
                raise ValueError(
                    "Qdrant result is missing chunk_id"
                )

            if document_id is None:
                raise ValueError(
                    "Qdrant result is missing document_id"
                )

            if chunk_index is None:
                raise ValueError(
                    "Qdrant result is missing chunk_index"
                )

            if content is None:
                raise ValueError(
                    "Qdrant result is missing content"
                )

            metadata = payload.get(
                "metadata",
                {},
            )

            results.append(
                VectorSearchResult(
                    chunk_id=str(chunk_id),
                    document_id=str(document_id),
                    chunk_index=int(chunk_index),
                    content=str(content),
                    score=float(point.score),
                    metadata=dict(metadata),
                )
            )

        return results
    
    

    @staticmethod
    def _point_id(
        chunk_id: str,
    ) -> str:
        """
        Generate a deterministic UUID for the Qdrant point.

        The application-level chunk ID remains the canonical
        identity and is stored in the payload.
        """
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"knowledge-intelligence-assistant:{chunk_id}",
            )
        )

    async def delete_by_document_id(self, document_id: str) -> None:
        if not document_id or not document_id.strip():
            raise ValueError("document_id must be non-empty")

        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    async def upsert_sparse(
        self,
        chunks: list[ChunkDocument],
    ) -> None:
        if not chunks:
            return

        points = [
            models.PointVectors(
                id=self._point_id(chunk.chunk_id),
                vector={
                    "bm25": models.Document(
                        text=chunk.content,
                        model="qdrant/bm25",
                    )
                },
            )
            for chunk in chunks
        ]

        await self.client.update_vectors(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    async def search_sparse(
        self,
        query: str,
        top_k: int,
    ) -> list[VectorSearchResult]:
        if not query or not query.strip():
            raise ValueError(
                "Sparse query cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        response = await self.client.query_points(
            collection_name=self.collection_name,
            query=models.Document(
                text=query,
                model="qdrant/bm25",
            ),
            using=self.BM25_VECTOR_NAME,
            limit=top_k,
            with_payload=True,
        )

        results: list[VectorSearchResult] = []

        for point in response.points:
            payload = point.payload or {}

            chunk_id = payload.get("chunk_id")
            document_id = payload.get("document_id")
            chunk_index = payload.get("chunk_index")
            content = payload.get("content")

            if chunk_id is None:
                raise ValueError(
                    "Qdrant result is missing chunk_id"
                )

            if document_id is None:
                raise ValueError(
                    "Qdrant result is missing document_id"
                )

            if chunk_index is None:
                raise ValueError(
                    "Qdrant result is missing chunk_index"
                )

            if content is None:
                raise ValueError(
                    "Qdrant result is missing content"
                )

            metadata = payload.get(
                "metadata",
                {},
            )

            results.append(
                VectorSearchResult(
                    chunk_id=str(chunk_id),
                    document_id=str(document_id),
                    chunk_index=int(chunk_index),
                    content=str(content),
                    score=float(point.score),
                    metadata=dict(metadata),
                )
            )

        return results