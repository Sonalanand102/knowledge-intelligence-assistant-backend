from __future__ import annotations

import asyncio

from backend.app.retrieval.base import (
    Retriever,
    VectorSearchResult,
)
from backend.app.retrieval.query_rewriter import (
    BaseQueryRewriter,
)


class QueryRewritingRetriever:
    """
    Retrieval pipeline:

        original query
            ↓
        query rewriter
            ↓
        underlying retriever
            ↓
        search results
    """

    def __init__(
        self,
        retriever: Retriever,
        query_rewriter: BaseQueryRewriter,
    ) -> None:
        self.retriever = retriever
        self.query_rewriter = query_rewriter

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        rewritten_query = await asyncio.to_thread(
            self.query_rewriter.rewrite,
            query,
        )

        return await self.retriever.retrieve(
            query=rewritten_query,
            top_k=top_k,
        )