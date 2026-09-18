from backend.app.evaluation.runner import (
    EmbeddingEvaluationRunner,
)

from backend.app.embeddings.evaluation import (
    EmbeddingEvaluationCase,
    EmbeddingEvaluationResult,
)
from backend.app.embeddings.service import EmbeddedChunk
from backend.app.ingestion.models.chunk_document import ChunkDocument

class FakeEmbeddingService:
    def embed_chunks(self, chunks):
        return []

    def embed_queries(self, queries):
        return []


def test_runner_requires_embedding_service():
    runner = EmbeddingEvaluationRunner(
        embedding_service=FakeEmbeddingService()
    )

    assert runner.embedding_service is not None


def test_runner_returns_aggregate_metrics():
    runner = EmbeddingEvaluationRunner(
        embedding_service=FakeEmbeddingService()
    )

    result = runner.aggregate_metrics(
        [
            {
                "recall_at_k": 1.0,
                "precision_at_k": 0.5,
                "mrr": 1.0,
            },
            {
                "recall_at_k": 0.5,
                "precision_at_k": 0.5,
                "mrr": 0.5,
            },
        ]
    )

    assert result["recall_at_k"] == 0.75
    assert result["precision_at_k"] == 0.5
    assert result["mrr"] == 0.75

def test_runner_evaluates_cases():
    chunk_1 = ChunkDocument(
        content="Unrelated content",
        document_id="doc_1",
        chunk_index=0,
    )

    chunk_2 = ChunkDocument(
        content="FastAPI is a Python web framework.",
        document_id="doc_1",
        chunk_index=1,
    )

    embedded_chunks = [
        EmbeddedChunk(
            chunk=chunk_1,
            embedding=[0.0, 1.0],
        ),
        EmbeddedChunk(
            chunk=chunk_2,
            embedding=[1.0, 0.0],
        ),
    ]

    case = EmbeddingEvaluationCase(
        query="What is FastAPI?",
        relevant_chunk_ids={chunk_2.chunk_id},
    )

    class FakeEmbeddingService:
        def embed_chunks(self, chunks):
            return embedded_chunks

        def embed_queries(self, queries):
            return [[1.0, 0.0] for _ in queries]

    runner = EmbeddingEvaluationRunner(
        embedding_service=FakeEmbeddingService()
    )

    results = runner.evaluate_cases(
        cases=[case],
        chunks=[chunk_1, chunk_2],
        k=1,
    )

    assert len(results) == 1
    assert isinstance(results[0], EmbeddingEvaluationResult)

    assert results[0].query == "What is FastAPI?"
    assert results[0].recall_at_k == 1.0
    assert results[0].precision_at_k == 1.0
    assert results[0].mrr == 1.0

def test_runner_aggregates_evaluation_results():
    runner = EmbeddingEvaluationRunner(
        embedding_service=FakeEmbeddingService()
    )

    results = [
        EmbeddingEvaluationResult(
            query="query 1",
            ranked_chunk_ids=["chunk_1"],
            recall_at_k=1.0,
            precision_at_k=1.0,
            mrr=1.0,
        ),
        EmbeddingEvaluationResult(
            query="query 2",
            ranked_chunk_ids=["chunk_2"],
            recall_at_k=0.5,
            precision_at_k=0.5,
            mrr=0.5,
        ),
    ]

    aggregated = runner.aggregate_results(results)

    assert aggregated.recall_at_k == 0.75
    assert aggregated.precision_at_k == 0.75
    assert aggregated.mrr == 0.75
    assert aggregated.evaluated_queries == 2