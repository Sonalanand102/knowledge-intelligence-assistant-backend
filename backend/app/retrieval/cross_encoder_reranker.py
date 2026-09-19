from __future__ import annotations

from dataclasses import replace
from typing import Any

from sentence_transformers import CrossEncoder

from backend.app.retrieval.base import VectorSearchResult
from backend.app.retrieval.reranker import BaseReranker


class CrossEncoderReranker(BaseReranker):
    """
    Cross-Encoder based reranker.

    Receives query + retrieved candidates and assigns a
    query-document relevance score to each candidate.
    """

    DEFAULT_MODEL = (
        "cross-encoder/ms-marco-MiniLM-L6-v2"
    )

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        model: Any | None = None,
    ) -> None:
        self.model = model or CrossEncoder(model_name)

    def _rerank(
        self,
        query: str,
        results: list[VectorSearchResult],
    ) -> list[VectorSearchResult]:
        pairs = [
            (query, result.content)
            for result in results
        ]

        scores = self.model.predict(pairs)

        if len(scores) != len(results):
            raise ValueError(
                "Reranker returned a different number "
                "of scores than candidates"
            )

        reranked_results = [
            replace(
                result,
                score=float(score),
            )
            for result, score in zip(results, scores)
        ]

        reranked_results.sort(
            key=lambda result: (
                -result.score,
                result.chunk_id,
            )
        )

        return reranked_results