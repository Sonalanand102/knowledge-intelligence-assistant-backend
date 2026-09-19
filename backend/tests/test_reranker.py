from __future__ import annotations

import pytest

from backend.app.retrieval.base import VectorSearchResult
from backend.app.retrieval.reranker import BaseReranker


class FakeReranker(BaseReranker):
    """
    Deterministic test implementation.

    Reverses the incoming candidate order so that we can verify
    that reranking actually changes ordering.
    """

    def _rerank(
        self,
        query: str,
        results: list[VectorSearchResult],
    ) -> list[VectorSearchResult]:
        return list(reversed(results))


def make_result(chunk_id: str) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        chunk_index=0,
        content=f"Content for {chunk_id}",
        score=0.5,
        metadata={},
    )


@pytest.fixture
def reranker() -> FakeReranker:
    return FakeReranker()


def test_empty_results_returns_empty_list(
    reranker: FakeReranker,
):
    result = reranker.rerank(
        query="test query",
        results=[],
    )

    assert result == []


def test_empty_query_is_rejected(
    reranker: FakeReranker,
):
    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        reranker.rerank(
            query="   ",
            results=[make_result("chunk-1")],
        )


def test_invalid_top_k_is_rejected(
    reranker: FakeReranker,
):
    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        reranker.rerank(
            query="test query",
            results=[make_result("chunk-1")],
            top_k=0,
        )


def test_top_k_limits_results(
    reranker: FakeReranker,
):
    results = [
        make_result("chunk-1"),
        make_result("chunk-2"),
        make_result("chunk-3"),
    ]

    reranked = reranker.rerank(
        query="test query",
        results=results,
        top_k=2,
    )

    assert len(reranked) == 2


def test_reranking_can_change_order(
    reranker: FakeReranker,
):
    results = [
        make_result("chunk-1"),
        make_result("chunk-2"),
        make_result("chunk-3"),
    ]

    reranked = reranker.rerank(
        query="test query",
        results=results,
        top_k=3,
    )

    assert [result.chunk_id for result in reranked] == [
        "chunk-3",
        "chunk-2",
        "chunk-1",
    ]


def test_reranking_preserves_candidate_identity(
    reranker: FakeReranker,
):
    results = [
        make_result("chunk-1"),
        make_result("chunk-2"),
        make_result("chunk-3"),
    ]

    reranked = reranker.rerank(
        query="test query",
        results=results,
        top_k=3,
    )

    assert {
        result.chunk_id
        for result in reranked
    } == {
        "chunk-1",
        "chunk-2",
        "chunk-3",
    }