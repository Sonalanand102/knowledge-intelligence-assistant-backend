from __future__ import annotations

import pytest

from backend.app.evaluation.retrieval_runner import (
    DenseRetrievalEvaluationRunner,
)
from backend.app.embeddings.evaluation import (
    EmbeddingEvaluationCase,
)
from backend.app.retrieval.base import VectorSearchResult


class FakeRetriever:
    async def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[VectorSearchResult]:
        return [
            VectorSearchResult(
                chunk_id="chunk_001",
                content="Relevant content.",
                score=0.99,
                metadata={},
                document_id="...",
                chunk_index=0,
            ),
            VectorSearchResult(
                chunk_id="chunk_002",
                content="Distractor content.",
                score=0.80,
                metadata={},
                document_id="...",
                chunk_index=0,
            ),
        ]


@pytest.mark.asyncio
async def test_evaluate_cases():
    runner = DenseRetrievalEvaluationRunner(
        retriever=FakeRetriever(),
    )

    cases = [
        EmbeddingEvaluationCase(
            query="What is RAG?",
            relevant_chunk_ids={
                "chunk_001",
            },
        )
    ]

    results = await runner.evaluate_cases(
        cases=cases,
        k=2,
    )

    assert len(results) == 1

    assert results[0].ranked_chunk_ids == [
        "chunk_001",
        "chunk_002",
    ]

    assert results[0].recall_at_k == 1.0
    assert results[0].precision_at_k == 0.5
    assert results[0].mrr == 1.0


@pytest.mark.asyncio
async def test_empty_cases():
    runner = DenseRetrievalEvaluationRunner(
        retriever=FakeRetriever(),
    )

    results = await runner.evaluate_cases(
        cases=[],
        k=5,
    )

    assert results == []


@pytest.mark.asyncio
async def test_invalid_k():
    runner = DenseRetrievalEvaluationRunner(
        retriever=FakeRetriever(),
    )

    with pytest.raises(ValueError):
        await runner.evaluate_cases(
            cases=[
                EmbeddingEvaluationCase(
                    query="What is RAG?",
                    relevant_chunk_ids={
                        "chunk_001",
                    },
                )
            ],
            k=0,
        )


def test_aggregate_results():
    runner = DenseRetrievalEvaluationRunner(
        retriever=FakeRetriever(),
    )

    results = [
        type(
            "Result",
            (),
            {
                "recall_at_k": 1.0,
                "precision_at_k": 0.5,
                "mrr": 1.0,
            },
        )(),
        type(
            "Result",
            (),
            {
                "recall_at_k": 0.5,
                "precision_at_k": 0.25,
                "mrr": 0.5,
            },
        )(),
    ]

    aggregate = runner.aggregate_results(
        results
    )

    assert aggregate.recall_at_k == 0.75
    assert aggregate.precision_at_k == 0.375
    assert aggregate.mrr == 0.75
    assert aggregate.evaluated_queries == 2


def test_aggregate_empty_results():
    runner = DenseRetrievalEvaluationRunner(
        retriever=FakeRetriever(),
    )

    aggregate = runner.aggregate_results([])

    assert aggregate.recall_at_k == 0.0
    assert aggregate.precision_at_k == 0.0
    assert aggregate.mrr == 0.0
    assert aggregate.evaluated_queries == 0