from __future__ import annotations

from abc import ABC, abstractmethod

from backend.app.retrieval.base import VectorSearchResult


class BaseReranker(ABC):
    """
    Base contract for rerankers.

    A reranker receives a query and a candidate set produced by
    an earlier retrieval stage, then returns the candidates in
    a new relevance order.
    """

    def rerank(
        self,
        query: str,
        results: list[VectorSearchResult],
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        if not results:
            return []

        ranked_results = self._rerank(
            query=query,
            results=results,
        )

        return ranked_results[:top_k]

    @abstractmethod
    def _rerank(
        self,
        query: str,
        results: list[VectorSearchResult],
    ) -> list[VectorSearchResult]:
        """
        Implement model-specific reranking here.
        """
        raise NotImplementedError