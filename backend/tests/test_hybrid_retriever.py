from __future__ import annotations

import pytest

from backend.app.retrieval.base import VectorSearchResult
from backend.app.retrieval.hybrid_retriever import HybridRetriever


def make_result(
    chunk_id: str,
    score: float,
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        chunk_index=0,
        content=f"Content for {chunk_id}",
        score=score,
        metadata={},
    )


class FakeDenseRetriever:
    def __init__(self) -> None:
        self.received_query = None
        self.received_top_k = None

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        self.received_query = query
        self.received_top_k = top_k

        return [
            make_result("A", 0.95),
            make_result("B", 0.90),
            make_result("C", 0.85),
        ]


class FakeSparseRetriever:
    def __init__(self) -> None:
        self.received_query = None
        self.received_top_k = None

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        self.received_query = query
        self.received_top_k = top_k

        return [
            make_result("B", 4.2),
            make_result("A", 3.8),
            make_result("D", 2.1),
        ]


@pytest.mark.asyncio
async def test_hybrid_retriever_combines_dense_and_sparse():
    dense = FakeDenseRetriever()
    sparse = FakeSparseRetriever()

    retriever = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
    )

    results = await retriever.retrieve(
        "What is RAG?",
        top_k=3,
        candidate_k=10,
    )

    assert dense.received_query == "What is RAG?"
    assert sparse.received_query == "What is RAG?"

    assert dense.received_top_k == 10
    assert sparse.received_top_k == 10

    assert len(results) == 3

    assert [
        result.chunk_id
        for result in results
    ] == [
        "A",
        "B",
        "C",
    ]


@pytest.mark.asyncio
async def test_hybrid_retriever_rejects_empty_query():
    retriever = HybridRetriever(
        dense_retriever=FakeDenseRetriever(),
        sparse_retriever=FakeSparseRetriever(),
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        await retriever.retrieve("")


@pytest.mark.asyncio
async def test_hybrid_retriever_rejects_invalid_top_k():
    retriever = HybridRetriever(
        dense_retriever=FakeDenseRetriever(),
        sparse_retriever=FakeSparseRetriever(),
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        await retriever.retrieve(
            "RAG",
            top_k=0,
        )


@pytest.mark.asyncio
async def test_hybrid_retriever_rejects_invalid_candidate_k():
    retriever = HybridRetriever(
        dense_retriever=FakeDenseRetriever(),
        sparse_retriever=FakeSparseRetriever(),
    )

    with pytest.raises(
        ValueError,
        match="candidate_k must be greater than zero",
    ):
        await retriever.retrieve(
            "RAG",
            top_k=5,
            candidate_k=0,
        )


@pytest.mark.asyncio
async def test_hybrid_retriever_requires_candidate_k_to_cover_top_k():
    retriever = HybridRetriever(
        dense_retriever=FakeDenseRetriever(),
        sparse_retriever=FakeSparseRetriever(),
    )

    with pytest.raises(
        ValueError,
        match="candidate_k must be greater than or equal to top_k",
    ):
        await retriever.retrieve(
            "RAG",
            top_k=10,
            candidate_k=5,
        )

def test_rrf_k_must_be_positive():
    with pytest.raises(ValueError, match="rrf_k must be greater than 0"):
        HybridRetriever(
            dense_retriever=FakeDenseRetriever(),
            sparse_retriever=FakeSparseRetriever(),
            rrf_k=0,
        )