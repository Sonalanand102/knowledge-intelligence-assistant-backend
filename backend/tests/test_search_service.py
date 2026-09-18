from __future__ import annotations

import pytest

from backend.app.retrieval.base import VectorSearchResult
from backend.app.retrieval.search_service import (
    SearchService,
)


class FakeRetriever:
    async def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[VectorSearchResult]:
        return [
            VectorSearchResult(
                chunk_id="chunk_rag_001",
                document_id="doc_rag",
                chunk_index=0,
                content="RAG retrieves external context.",
                score=0.95,
                metadata={
                    "source": "test.pdf",
                },
            )
        ]


@pytest.mark.asyncio
async def test_search_service_returns_search_response():
    service = SearchService(
        retriever=FakeRetriever(),
    )

    response = await service.search(
        query="What is RAG?",
        top_k=5,
    )

    assert response.query == "What is RAG?"

    assert len(response.results) == 1

    result = response.results[0]

    assert result.chunk_id == "chunk_rag_001"
    assert result.document_id == "doc_rag"
    assert result.chunk_index == 0
    assert result.score == 0.95


@pytest.mark.asyncio
async def test_search_service_rejects_empty_query():
    service = SearchService(
        retriever=FakeRetriever(),
    )

    with pytest.raises(ValueError):
        await service.search(
            query="",
            top_k=5,
        )


@pytest.mark.asyncio
async def test_search_service_rejects_invalid_top_k():
    service = SearchService(
        retriever=FakeRetriever(),
    )

    with pytest.raises(ValueError):
        await service.search(
            query="What is RAG?",
            top_k=0,
        )