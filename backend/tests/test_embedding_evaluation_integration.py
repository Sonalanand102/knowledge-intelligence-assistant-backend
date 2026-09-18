from backend.app.embeddings.service import EmbeddedChunk
from backend.app.evaluation.datasets.embedding_dataset import (
    get_embedding_evaluation_cases,
    get_embedding_evaluation_corpus,
)
from backend.app.evaluation.runner import EmbeddingEvaluationRunner
from backend.app.ingestion.models.chunk_document import ChunkDocument


class FakeEmbeddingService:
    def embed_chunks(self, chunks):
        return [
            EmbeddedChunk(
                chunk=chunk,
                embedding=[1.0, 0.0],
            )
            for chunk in chunks
        ]

    def embed_queries(self, queries):
        return [
            [1.0, 0.0]
            for _ in queries
        ]


def test_evaluation_pipeline_works_with_embedding_service():
    corpus = get_embedding_evaluation_corpus()
    cases = get_embedding_evaluation_cases()

    chunks = [
        ChunkDocument(
            content=content,
            document_id=chunk_id,
            chunk_index=0,
        )
        for chunk_id, content in corpus.items()
    ]

    runner = EmbeddingEvaluationRunner(
        embedding_service=FakeEmbeddingService()
    )

    results = runner.evaluate_cases(
        cases=cases,
        chunks=chunks,
        k=5,
    )

    assert len(results) == len(cases)

    aggregated = runner.aggregate_results(results)

    assert aggregated.evaluated_queries == len(cases)