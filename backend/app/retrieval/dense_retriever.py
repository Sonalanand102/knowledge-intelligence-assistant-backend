from __future__ import annotations

import asyncio

from backend.app.embeddings.service import EmbeddingService
from backend.app.retrieval.base import VectorSearchResult, VectorStore


class DenseRetriever:
    """
    Dense semantic retriever.

    Converts a natural-language query into an embedding
    and retrieves the nearest chunks from a vector store.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        if not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        query_embedding = await asyncio.to_thread(
            self.embedding_service.embed_query,
            query,
        )

        if document_ids:
            return await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                document_ids=document_ids,
            )

        return await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )