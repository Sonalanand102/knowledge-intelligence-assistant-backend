from __future__ import annotations

from backend.app.retrieval.base import SparseVectorStore
from backend.app.retrieval.base import VectorSearchResult


class SparseRetriever:
    def __init__(
        self,
        vector_store: SparseVectorStore,
    ) -> None:
        self.vector_store = vector_store

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        return await self.vector_store.search_sparse(
            query=query,
            top_k=top_k,
        )