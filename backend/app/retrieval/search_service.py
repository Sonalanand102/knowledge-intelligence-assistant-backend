from __future__ import annotations

from dataclasses import dataclass

from backend.app.retrieval.base import VectorSearchResult
from backend.app.retrieval.dense_retriever import DenseRetriever
from backend.app.retrieval.hybrid_retriever import HybridRetriever

@dataclass(frozen=True)
class SearchResponse:
    query: str
    results: list[VectorSearchResult]


class SearchService:
    """
    Application-level search orchestration.

    Keeps API concerns separate from retrieval implementation.
    """

    def __init__(
        self,
        retriever: HybridRetriever | DenseRetriever,
    ) -> None:
        self.retriever = retriever

    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> SearchResponse:
        if not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        results = await self.retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        return SearchResponse(
            query=query,
            results=results,
        )