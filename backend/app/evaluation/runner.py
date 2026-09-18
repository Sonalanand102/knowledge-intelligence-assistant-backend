from __future__ import annotations

from typing import Any

from backend.app.embeddings.evaluation import (
    AggregateEmbeddingEvaluationResult,
    EmbeddingEvaluationCase,
    EmbeddingEvaluationResult,
    EmbeddingEvaluator,
)
from backend.app.ingestion.models.chunk_document import ChunkDocument


class EmbeddingEvaluationRunner:
    def __init__(self, embedding_service: Any) -> None:
        self.embedding_service = embedding_service
        self.evaluator = EmbeddingEvaluator()

    def evaluate_cases(
        self,
        cases: list[EmbeddingEvaluationCase],
        chunks: list[ChunkDocument],
        k: int,
    ) -> list[EmbeddingEvaluationResult]:
        if not cases:
            return []

        if not chunks:
            raise ValueError(
                "Cannot evaluate embeddings without chunks"
            )

        # Generate document/chunk embeddings once.
        embedded_chunks = self.embedding_service.embed_chunks(chunks)

        # Generate all query embeddings in one batch.
        query_embeddings = self.embedding_service.embed_queries(
            [case.query for case in cases]
        )

        if len(query_embeddings) != len(cases):
            raise ValueError(
                "Embedding service returned a different number "
                "of query embeddings than evaluation cases"
            )

        results: list[EmbeddingEvaluationResult] = []

        for case, query_embedding in zip(
            cases,
            query_embeddings,
        ):
            result = self.evaluator.evaluate_case(
                case=case,
                query_embedding=query_embedding,
                embedded_chunks=embedded_chunks,
                k=k,
            )

            results.append(result)

        return results

    @staticmethod
    def aggregate_metrics(
        results: list[dict[str, float]],
    ) -> dict[str, float]:
        if not results:
            return {
                "recall_at_k": 0.0,
                "precision_at_k": 0.0,
                "mrr": 0.0,
            }

        return {
            "recall_at_k": sum(
                result["recall_at_k"]
                for result in results
            ) / len(results),
            "precision_at_k": sum(
                result["precision_at_k"]
                for result in results
            ) / len(results),
            "mrr": sum(
                result["mrr"]
                for result in results
            ) / len(results),
        }

    def aggregate_results(
        self,
        results: list[EmbeddingEvaluationResult],
    ) -> AggregateEmbeddingEvaluationResult:
        if not results:
            return AggregateEmbeddingEvaluationResult(
                recall_at_k=0.0,
                precision_at_k=0.0,
                mrr=0.0,
                evaluated_queries=0,
            )

        return AggregateEmbeddingEvaluationResult(
            recall_at_k=sum(
                result.recall_at_k
                for result in results
            ) / len(results),
            precision_at_k=sum(
                result.precision_at_k
                for result in results
            ) / len(results),
            mrr=sum(
                result.mrr
                for result in results
            ) / len(results),
            evaluated_queries=len(results),
        )