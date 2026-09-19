from __future__ import annotations

from backend.app.retrieval.base import VectorSearchResult
from backend.app.retrieval.cross_encoder_reranker import (
    CrossEncoderReranker,
)


class FakeCrossEncoder:
    def predict(self, pairs):
        assert pairs == [
            (
                "What is RAG?",
                "RAG combines retrieval with generation.",
            ),
            (
                "What is RAG?",
                "FastAPI is a Python web framework.",
            ),
            (
                "What is RAG?",
                "RAG retrieves relevant context for an LLM.",
            ),
        ]

        return [0.3, 0.1, 0.9]


def make_result(
    chunk_id: str,
    content: str,
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        chunk_index=0,
        content=content,
        score=0.5,
        metadata={},
    )


def test_cross_encoder_reranks_candidates():
    reranker = CrossEncoderReranker(
        model=FakeCrossEncoder(),
    )

    results = [
        make_result(
            "chunk-1",
            "RAG combines retrieval with generation.",
        ),
        make_result(
            "chunk-2",
            "FastAPI is a Python web framework.",
        ),
        make_result(
            "chunk-3",
            "RAG retrieves relevant context for an LLM.",
        ),
    ]

    reranked = reranker.rerank(
        query="What is RAG?",
        results=results,
        top_k=3,
    )

    assert [
        result.chunk_id
        for result in reranked
    ] == [
        "chunk-3",
        "chunk-1",
        "chunk-2",
    ]


def test_cross_encoder_updates_scores():
    reranker = CrossEncoderReranker(
        model=FakeCrossEncoder(),
    )

    results = [
        make_result(
            "chunk-1",
            "RAG combines retrieval with generation.",
        ),
        make_result(
            "chunk-2",
            "FastAPI is a Python web framework.",
        ),
        make_result(
            "chunk-3",
            "RAG retrieves relevant context for an LLM.",
        ),
    ]

    reranked = reranker.rerank(
        query="What is RAG?",
        results=results,
        top_k=3,
    )

    assert reranked[0].score == 0.9
    assert reranked[1].score == 0.3
    assert reranked[2].score == 0.1


def test_cross_encoder_respects_top_k():
    reranker = CrossEncoderReranker(
        model=FakeCrossEncoder(),
    )

    results = [
        make_result(
            "chunk-1",
            "RAG combines retrieval with generation.",
        ),
        make_result(
            "chunk-2",
            "FastAPI is a Python web framework.",
        ),
        make_result(
            "chunk-3",
            "RAG retrieves relevant context for an LLM.",
        ),
    ]

    reranked = reranker.rerank(
        query="What is RAG?",
        results=results,
        top_k=2,
    )

    assert len(reranked) == 2
    assert [
        result.chunk_id
        for result in reranked
    ] == [
        "chunk-3",
        "chunk-1",
    ]