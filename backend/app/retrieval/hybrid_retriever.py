from __future__ import annotations

import asyncio

from backend.app.retrieval.base import VectorSearchResult
from backend.app.retrieval.rrf import reciprocal_rank_fusion


class HybridRetriever:
    def __init__(
        self,
        dense_retriever,
        sparse_retriever,
        rrf_k: int = 60,
    ) -> None:

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0")
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf_k = rrf_k

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10,
    ) -> list[VectorSearchResult]:
        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        if candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than zero"
            )

        if candidate_k < top_k:
            raise ValueError(
                "candidate_k must be greater than or equal to top_k"
            )

        dense_results, sparse_results = await asyncio.gather(
            self.dense_retriever.retrieve(
                query,
                top_k=candidate_k,
            ),
            self.sparse_retriever.retrieve(
                query,
                top_k=candidate_k,
            ),
        )

        return reciprocal_rank_fusion(
            [
                dense_results,
                sparse_results,
            ],
            top_k=top_k,
            rrf_k=self.rrf_k,
        )