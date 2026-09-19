from __future__ import annotations

import asyncio

from backend.app.retrieval.base import Retriever, VectorSearchResult
from backend.app.retrieval.reranker import BaseReranker


class RerankingRetriever:
    """
    Two-stage retrieval:

        first-stage retriever
            ↓
        candidate_k
            ↓
        reranker
            ↓
        top_k
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: BaseReranker,
        candidate_k: int = 10,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than zero"
            )

        self.retriever = retriever
        self.reranker = reranker
        self.candidate_k = candidate_k

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        if self.candidate_k < top_k:
            raise ValueError(
                "candidate_k must be greater than or equal to top_k"
            )

        candidates = await self.retriever.retrieve(
            query=query,
            top_k=self.candidate_k,
        )

        if not candidates:
            return []

        return await asyncio.to_thread(
            self.reranker.rerank,
            query,
            candidates,
            top_k,
        )