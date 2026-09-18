from __future__ import annotations

import pytest

from backend.app.retrieval.base import VectorSearchResult


class FakeRetriever:
    async def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[VectorSearchResult]:
        assert query == "What is RAG?"
        assert top_k == 3

        return [
            VectorSearchResult(
                chunk_id="chunk_rag_001",
                document_id="doc_rag",
                chunk_index=0,
                content="RAG retrieves external context.",
                score=0.97,
                metadata={
                    "source": "evaluation",
                },
            ),
            VectorSearchResult(
                chunk_id="chunk_rag_002",
                document_id="doc_rag",
                chunk_index=1,
                content="RAG combines retrieval and generation.",
                score=0.91,
                metadata={
                    "source": "evaluation",
                },
            ),
        ]


@pytest.mark.asyncio
async def test_search_retrieval_contract():
    retriever = FakeRetriever()

    results = await retriever.retrieve(
        query="What is RAG?",
        top_k=3,
    )

    assert len(results) == 2

    assert results[0].chunk_id == (
        "chunk_rag_001"
    )
    assert results[0].document_id == (
        "doc_rag"
    )
    assert results[0].chunk_index == 0
    assert results[0].score == 0.97

    assert results[1].chunk_id == (
        "chunk_rag_002"
    )
    assert results[1].document_id == (
        "doc_rag"
    )
    assert results[1].chunk_index == 1
    assert results[1].score == 0.91