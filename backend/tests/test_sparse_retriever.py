from __future__ import annotations

import pytest

from backend.app.retrieval.base import VectorSearchResult
from backend.app.retrieval.sparse_retriever import SparseRetriever


class FakeSparseVectorStore:
    def __init__(self) -> None:
        self.query = None
        self.top_k = None

    async def search_sparse(
        self,
        query: str,
        top_k: int,
    ) -> list[VectorSearchResult]:
        self.query = query
        self.top_k = top_k

        return [
            VectorSearchResult(
                chunk_id="chunk-1",
                document_id="doc-1",
                chunk_index=0,
                content="Retrieval augmented generation",
                score=3.5,
                metadata={},
            )
        ]


@pytest.mark.asyncio
async def test_sparse_retriever_retrieves_results():
    store = FakeSparseVectorStore()

    retriever = SparseRetriever(
        vector_store=store,
    )

    results = await retriever.retrieve(
        "retrieval augmented generation",
        top_k=5,
    )

    assert store.query == (
        "retrieval augmented generation"
    )

    assert store.top_k == 5

    assert len(results) == 1

    assert results[0].chunk_id == "chunk-1"


@pytest.mark.asyncio
async def test_sparse_retriever_rejects_empty_query():
    store = FakeSparseVectorStore()

    retriever = SparseRetriever(
        vector_store=store,
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        await retriever.retrieve("")


@pytest.mark.asyncio
async def test_sparse_retriever_rejects_invalid_top_k():
    store = FakeSparseVectorStore()

    retriever = SparseRetriever(
        vector_store=store,
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        await retriever.retrieve(
            "RAG",
            top_k=0,
        )