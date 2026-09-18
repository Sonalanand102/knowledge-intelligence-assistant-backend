from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from backend.app.embeddings.evaluation import (
    EmbeddingEvaluationCase,
    EmbeddingEvaluationResult,
    EmbeddingEvaluator,
)
from backend.app.retrieval.base import VectorSearchResult


class Retriever(Protocol):
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        ...


@dataclass(frozen=True)
class AggregateRetrievalEvaluationResult:
    recall_at_k: float
    precision_at_k: float
    mrr: float
    evaluated_queries: int


class DenseRetrievalEvaluationRunner:
    """
    Evaluates any retriever that follows the Retriever protocol.

    Current consumers:
        - DenseRetriever
        - SparseRetriever
        - HybridRetriever
    """

    def __init__(
        self,
        retriever: Retriever,
    ) -> None:
        self.retriever = retriever
        self.evaluator = EmbeddingEvaluator()

    async def evaluate_cases(
        self,
        cases: list[EmbeddingEvaluationCase],
        k: int,
    ) -> list[EmbeddingEvaluationResult]:
        if not cases:
            return []

        if k <= 0:
            raise ValueError(
                "k must be greater than zero"
            )

        results: list[EmbeddingEvaluationResult] = []

        for case in cases:
            retrieved_results = await self.retriever.retrieve(
                query=case.query,
                top_k=k,
            )

            ranked_chunk_ids = [
                result.chunk_id
                for result in retrieved_results
            ]

            results.append(
                EmbeddingEvaluationResult(
                    query=case.query,
                    ranked_chunk_ids=ranked_chunk_ids,
                    recall_at_k=self.evaluator.recall_at_k(
                        ranked_chunk_ids,
                        case.relevant_chunk_ids,
                        k,
                    ),
                    precision_at_k=self.evaluator.precision_at_k(
                        ranked_chunk_ids,
                        case.relevant_chunk_ids,
                        k,
                    ),
                    mrr=self.evaluator.mrr(
                        ranked_chunk_ids,
                        case.relevant_chunk_ids,
                    ),
                )
            )

        return results

    def aggregate_results(
        self,
        results: list[EmbeddingEvaluationResult],
    ) -> AggregateRetrievalEvaluationResult:
        if not results:
            return AggregateRetrievalEvaluationResult(
                recall_at_k=0.0,
                precision_at_k=0.0,
                mrr=0.0,
                evaluated_queries=0,
            )

        return AggregateRetrievalEvaluationResult(
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