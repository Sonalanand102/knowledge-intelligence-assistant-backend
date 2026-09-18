from __future__ import annotations

from dataclasses import dataclass
import math

from backend.app.embeddings.service import EmbeddedChunk

@dataclass(frozen=True)
class AggregateEmbeddingEvaluationResult:
    recall_at_k: float
    precision_at_k: float
    mrr: float
    evaluated_queries: int
    
@dataclass(frozen=True)
class EmbeddingEvaluationResult:
    query: str
    ranked_chunk_ids: list[str]
    recall_at_k: float
    precision_at_k: float
    mrr: float

@dataclass(frozen=True)
class EmbeddingEvaluationCase:
    query: str
    relevant_chunk_ids: set[str]


class EmbeddingEvaluator:

    @staticmethod
    def cosine_similarity(
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:
        if len(vector_a) != len(vector_b):
            raise ValueError(
                "Vectors must have the same dimensions"
            )

        dot_product = sum(
            a * b
            for a, b in zip(vector_a, vector_b)
        )

        magnitude_a = math.sqrt(
            sum(a * a for a in vector_a)
        )

        magnitude_b = math.sqrt(
            sum(b * b for b in vector_b)
        )

        if magnitude_a == 0 or magnitude_b == 0:
            raise ValueError(
                "Cosine similarity is undefined for zero vectors"
            )

        return dot_product / (magnitude_a * magnitude_b)

    def rank_chunks(
        self,
        query_embedding: list[float],
        chunk_embeddings: dict[str, list[float]],
    ) -> list[str]:

        scored_chunks = []

        for chunk_id, embedding in chunk_embeddings.items():
            similarity = self.cosine_similarity(
                query_embedding,
                embedding,
            )

            scored_chunks.append(
                (chunk_id, similarity)
            )

        scored_chunks.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            chunk_id
            for chunk_id, _ in scored_chunks
        ]

    @staticmethod
    def recall_at_k(
        ranked_chunk_ids: list[str],
        relevant_chunk_ids: set[str],
        k: int,
    ) -> float:

        if not relevant_chunk_ids:
            return 0.0

        retrieved = set(ranked_chunk_ids[:k])

        relevant_retrieved = (
            retrieved & relevant_chunk_ids
        )

        return (
            len(relevant_retrieved)
            / len(relevant_chunk_ids)
        )

    @staticmethod
    def precision_at_k(
        ranked_chunk_ids: list[str],
        relevant_chunk_ids: set[str],
        k: int,
    ) -> float:

        if k <= 0:
            return 0.0

        retrieved = ranked_chunk_ids[:k]

        relevant_retrieved = sum(
            chunk_id in relevant_chunk_ids
            for chunk_id in retrieved
        )

        return relevant_retrieved / len(retrieved)

    @staticmethod
    def mrr(
        ranked_chunk_ids: list[str],
        relevant_chunk_ids: set[str],
    ) -> float:

        for rank, chunk_id in enumerate(
            ranked_chunk_ids,
            start=1,
        ):
            if chunk_id in relevant_chunk_ids:
                return 1.0 / rank

        return 0.0

    def rank_embedded_chunks(
        self,
        query_embedding: list[float],
        embedded_chunks: list[EmbeddedChunk],
    ) -> list[str]:
        chunk_embeddings = {
            embedded_chunk.chunk.chunk_id: embedded_chunk.embedding
            for embedded_chunk in embedded_chunks
        }

        return self.rank_chunks(
            query_embedding=query_embedding,
            chunk_embeddings=chunk_embeddings,
        )

    def evaluate_case(
        self,
        case: EmbeddingEvaluationCase,
        query_embedding: list[float],
        embedded_chunks: list[EmbeddedChunk],
        k: int,
    ) -> EmbeddingEvaluationResult:

        ranked_chunk_ids = self.rank_embedded_chunks(
            query_embedding=query_embedding,
            embedded_chunks=embedded_chunks,
        )

        return EmbeddingEvaluationResult(
            query=case.query,
            ranked_chunk_ids=ranked_chunk_ids,
            recall_at_k=self.recall_at_k(
                ranked_chunk_ids=ranked_chunk_ids,
                relevant_chunk_ids=case.relevant_chunk_ids,
                k=k,
            ),
            precision_at_k=self.precision_at_k(
                ranked_chunk_ids=ranked_chunk_ids,
                relevant_chunk_ids=case.relevant_chunk_ids,
                k=k,
            ),
            mrr=self.mrr(
                ranked_chunk_ids=ranked_chunk_ids,
                relevant_chunk_ids=case.relevant_chunk_ids,
            ),
        )
