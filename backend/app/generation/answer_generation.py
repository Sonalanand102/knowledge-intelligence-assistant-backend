from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from backend.app.retrieval.base import VectorSearchResult


@dataclass(frozen=True)
class Citation:
    citation_id: int
    chunk_id: str
    document_id: str
    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    citations: list[Citation]


class BaseAnswerGenerator(ABC):
    def generate(
        self,
        query: str,
        results: list[VectorSearchResult],
    ) -> GeneratedAnswer:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not results:
            return GeneratedAnswer(
                answer=(
                    "I could not find enough relevant information "
                    "to answer the question."
                ),
                citations=[],
            )

        return self._generate(
            query=query.strip(),
            results=results,
        )

    @abstractmethod
    def _generate(
        self,
        query: str,
        results: list[VectorSearchResult],
    ) -> GeneratedAnswer:
        raise NotImplementedError


class GroundedAnswerGenerator(BaseAnswerGenerator):
    """
    Deterministic baseline answer generator.

    This class does not call an LLM. It provides a simple
    context-backed representation that we can use to validate
    the generation/citation contract before adding Gemini.
    """

    def _generate(
        self,
        query: str,
        results: list[VectorSearchResult],
    ) -> GeneratedAnswer:
        citations: list[Citation] = []

        for index, result in enumerate(
            results,
            start=1,
        ):
            citations.append(
                Citation(
                    citation_id=index,
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    metadata=dict(result.metadata),
                )
            )

        answer_parts = [
            f"[{index}] {result.content}"
            for index, result in enumerate(
                results,
                start=1,
            )
        ]

        answer = "\n\n".join(answer_parts)

        return GeneratedAnswer(
            answer=answer,
            citations=citations,
        )