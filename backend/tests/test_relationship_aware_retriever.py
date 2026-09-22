from __future__ import annotations

import pytest

from backend.app.retrieval.base import VectorSearchResult
from backend.app.retrieval.relationship_aware_retriever import (
    RelationshipAwareRetriever,
)


class FakeRetriever:
    def __init__(
        self,
        results: list[VectorSearchResult],
    ) -> None:
        self.results = results
        self.calls: list[dict] = []

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_ids: set[str] | None = None,
    ) -> list[VectorSearchResult]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "document_ids": document_ids,
            }
        )

        return self.results


class FakeExpander:
    def __init__(self) -> None:
        self.calls: list[list[VectorSearchResult]] = []

    async def expand(
        self,
        results: list[VectorSearchResult],
    ) -> list[VectorSearchResult]:
        self.calls.append(results)

        expanded_results = []

        for result in results:
            expanded_results.append(
                VectorSearchResult(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    chunk_index=result.chunk_index,
                    content=(
                        result.content
                        + "\n\n[Related context]\n"
                        + "Related information"
                    ),
                    score=result.score,
                    metadata={
                        **result.metadata,
                        "relationship_context": [
                            {
                                "element_id": "related-element",
                                "text": "Related information",
                            }
                        ],
                    },
                )
            )

        return expanded_results


def make_result(
    chunk_id: str = "chunk-1",
    document_id: str = "doc-1",
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        content="Original content",
        score=0.95,
        metadata={
            "element_ids": ["element-1"],
        },
    )


@pytest.mark.asyncio
async def test_relationship_aware_retriever_expands_results():

    retriever = FakeRetriever(
        results=[make_result()]
    )

    expander = FakeExpander()

    relationship_aware_retriever = (
        RelationshipAwareRetriever(
            retriever=retriever,
            context_expander=expander,
        )
    )

    results = await relationship_aware_retriever.retrieve(
        query="What is RAG?",
        top_k=5,
    )

    assert len(results) == 1

    assert (
        results[0].content
        == "Original content\n\n"
        "[Related context]\n"
        "Related information"
    )

    assert (
        results[0].metadata[
            "relationship_context"
        ][0]["element_id"]
        == "related-element"
    )

    assert len(expander.calls) == 1


@pytest.mark.asyncio
async def test_relationship_aware_retriever_forwards_retrieval_arguments():

    retriever = FakeRetriever(
        results=[make_result()]
    )

    expander = FakeExpander()

    service = RelationshipAwareRetriever(
        retriever=retriever,
        context_expander=expander,
    )

    document_ids = {
        "doc-1",
        "doc-2",
    }

    await service.retrieve(
        query="Explain embeddings",
        top_k=3,
        document_ids=document_ids,
    )

    assert retriever.calls == [
        {
            "query": "Explain embeddings",
            "top_k": 3,
            "document_ids": document_ids,
        }
    ]


@pytest.mark.asyncio
async def test_relationship_aware_retriever_preserves_empty_results():

    retriever = FakeRetriever(
        results=[]
    )

    expander = FakeExpander()

    service = RelationshipAwareRetriever(
        retriever=retriever,
        context_expander=expander,
    )

    results = await service.retrieve(
        query="Unknown question",
        top_k=5,
    )

    assert results == []

    assert len(expander.calls) == 1
    assert expander.calls[0] == []


@pytest.mark.asyncio
async def test_relationship_aware_retriever_rejects_empty_query():

    retriever = FakeRetriever([])

    expander = FakeExpander()

    service = RelationshipAwareRetriever(
        retriever=retriever,
        context_expander=expander,
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        await service.retrieve(
            query="   ",
            top_k=5,
        )


@pytest.mark.asyncio
async def test_relationship_aware_retriever_rejects_invalid_top_k():

    retriever = FakeRetriever([])

    expander = FakeExpander()

    service = RelationshipAwareRetriever(
        retriever=retriever,
        context_expander=expander,
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        await service.retrieve(
            query="What is RAG?",
            top_k=0,
        )