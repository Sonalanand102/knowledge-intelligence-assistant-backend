from dataclasses import dataclass

import pytest

from backend.app.retrieval.query_rewriter import (
    BaseQueryRewriter,
)
from backend.app.retrieval.query_rewriting_retriever import (
    QueryRewritingRetriever,
)
from backend.app.retrieval.base import (
    VectorSearchResult,
)


@dataclass
class FakeQueryRewriter(BaseQueryRewriter):
    rewritten_query: str
    received_query: str | None = None

    def _rewrite(self, query: str) -> str:
        self.received_query = query
        return self.rewritten_query


class FakeRetriever:
    def __init__(
        self,
        results: list[VectorSearchResult],
    ) -> None:
        self.results = results
        self.received_query: str | None = None
        self.received_top_k: int | None = None

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        self.received_query = query
        self.received_top_k = top_k
        return self.results[:top_k]


@pytest.fixture
def results() -> list[VectorSearchResult]:
    return [
        VectorSearchResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            chunk_index=0,
            content="First result",
            score=0.9,
        ),
        VectorSearchResult(
            chunk_id="chunk-2",
            document_id="doc-1",
            chunk_index=1,
            content="Second result",
            score=0.8,
        ),
    ]


@pytest.mark.asyncio
async def test_rewrites_query_before_retrieval(results):
    rewriter = FakeQueryRewriter(
        rewritten_query="retrieval augmented generation definition"
    )

    retriever = FakeRetriever(results)

    rewriting_retriever = QueryRewritingRetriever(
        retriever=retriever,
        query_rewriter=rewriter,
    )

    returned_results = await rewriting_retriever.retrieve(
        query="What is RAG?",
        top_k=5,
    )

    assert rewriter.received_query == "What is RAG?"

    assert (
        retriever.received_query
        == "retrieval augmented generation definition"
    )

    assert returned_results == results


@pytest.mark.asyncio
async def test_passes_top_k_to_underlying_retriever(results):
    rewriter = FakeQueryRewriter(
        rewritten_query="rewritten query"
    )

    retriever = FakeRetriever(results)

    rewriting_retriever = QueryRewritingRetriever(
        retriever=retriever,
        query_rewriter=rewriter,
    )

    await rewriting_retriever.retrieve(
        query="What is RAG?",
        top_k=1,
    )

    assert retriever.received_top_k == 1


@pytest.mark.asyncio
async def test_empty_results_are_returned(results):
    rewriter = FakeQueryRewriter(
        rewritten_query="rewritten query"
    )

    retriever = FakeRetriever([])

    rewriting_retriever = QueryRewritingRetriever(
        retriever=retriever,
        query_rewriter=rewriter,
    )

    returned_results = await rewriting_retriever.retrieve(
        query="What is RAG?",
        top_k=5,
    )

    assert returned_results == []


@pytest.mark.asyncio
async def test_empty_query_is_rejected(results):
    rewriter = FakeQueryRewriter(
        rewritten_query="rewritten query"
    )

    retriever = FakeRetriever(results)

    rewriting_retriever = QueryRewritingRetriever(
        retriever=retriever,
        query_rewriter=rewriter,
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        await rewriting_retriever.retrieve(
            query="   ",
            top_k=5,
        )


@pytest.mark.asyncio
async def test_invalid_top_k_is_rejected(results):
    rewriter = FakeQueryRewriter(
        rewritten_query="rewritten query"
    )

    retriever = FakeRetriever(results)

    rewriting_retriever = QueryRewritingRetriever(
        retriever=retriever,
        query_rewriter=rewriter,
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        await rewriting_retriever.retrieve(
            query="What is RAG?",
            top_k=0,
        )


class EmptyQueryRewriter(BaseQueryRewriter):
    def _rewrite(self, query: str) -> str:
        return "   "


@pytest.mark.asyncio
async def test_empty_rewrite_is_rejected(results):
    rewriter = EmptyQueryRewriter()
    retriever = FakeRetriever(results)

    rewriting_retriever = QueryRewritingRetriever(
        retriever=retriever,
        query_rewriter=rewriter,
    )

    with pytest.raises(
        ValueError,
        match="Rewriter returned an empty query",
    ):
        await rewriting_retriever.retrieve(
            query="What is RAG?",
            top_k=5,
        )