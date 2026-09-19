from __future__ import annotations

import pytest

from backend.app.retrieval.base import VectorSearchResult
from backend.app.retrieval.reranking_retriever import (
    RerankingRetriever,
)
from backend.app.retrieval.reranker import BaseReranker


def make_result(chunk_id: str) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        chunk_index=0,
        content=f"Content {chunk_id}",
        score=0.5,
        metadata={},
    )


class FakeRetriever:
    def __init__(self) -> None:
        self.requested_top_k: int | None = None

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        self.requested_top_k = top_k

        return [
            make_result("chunk-1"),
            make_result("chunk-2"),
            make_result("chunk-3"),
        ]


class FakeReranker(BaseReranker):
    def _rerank(
        self,
        query: str,
        results: list[VectorSearchResult],
    ) -> list[VectorSearchResult]:
        return list(reversed(results))


@pytest.mark.asyncio
async def test_uses_candidate_k_for_first_stage():
    retriever = FakeRetriever()

    pipeline = RerankingRetriever(
        retriever=retriever,
        reranker=FakeReranker(),
        candidate_k=10,
    )

    await pipeline.retrieve(
        query="test query",
        top_k=5,
    )

    assert retriever.requested_top_k == 10


@pytest.mark.asyncio
async def test_returns_reranked_results():
    pipeline = RerankingRetriever(
        retriever=FakeRetriever(),
        reranker=FakeReranker(),
        candidate_k=3,
    )

    results = await pipeline.retrieve(
        query="test query",
        top_k=3,
    )

    assert [
        result.chunk_id
        for result in results
    ] == [
        "chunk-3",
        "chunk-2",
        "chunk-1",
    ]


@pytest.mark.asyncio
async def test_top_k_limits_final_results():
    pipeline = RerankingRetriever(
        retriever=FakeRetriever(),
        reranker=FakeReranker(),
        candidate_k=3,
    )

    results = await pipeline.retrieve(
        query="test query",
        top_k=2,
    )

    assert len(results) == 2


@pytest.mark.asyncio
async def test_empty_results_return_empty_list():
    class EmptyRetriever:
        async def retrieve(
            self,
            query: str,
            top_k: int = 5,
        ) -> list[VectorSearchResult]:
            return []

    pipeline = RerankingRetriever(
        retriever=EmptyRetriever(),
        reranker=FakeReranker(),
        candidate_k=10,
    )

    results = await pipeline.retrieve(
        query="test query",
        top_k=5,
    )

    assert results == []


@pytest.mark.asyncio
async def test_candidate_k_cannot_be_less_than_top_k():
    pipeline = RerankingRetriever(
        retriever=FakeRetriever(),
        reranker=FakeReranker(),
        candidate_k=3,
    )

    with pytest.raises(
        ValueError,
        match="candidate_k must be greater than or equal to top_k",
    ):
        await pipeline.retrieve(
            query="test query",
            top_k=5,
        )


@pytest.mark.asyncio
async def test_empty_query_is_rejected():
    pipeline = RerankingRetriever(
        retriever=FakeRetriever(),
        reranker=FakeReranker(),
        candidate_k=10,
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        await pipeline.retrieve(
            query="   ",
            top_k=5,
        )