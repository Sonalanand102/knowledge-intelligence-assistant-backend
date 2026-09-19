from __future__ import annotations

from dataclasses import dataclass

from backend.app.generation.answer_generation import (
    Citation,
)
from backend.app.retrieval.base import (
    VectorSearchResult,
)


@dataclass(frozen=True)
class CitationValidationResult:
    valid: bool
    validated_citations: list[Citation]
    invalid_citation_ids: list[int]
    coverage: float


class CitationValidator:
    def validate(
        self,
        citations: list[Citation],
        results: list[VectorSearchResult],
    ) -> CitationValidationResult:
        if not results:
            return CitationValidationResult(
                valid=not citations,
                validated_citations=[],
                invalid_citation_ids=[
                    citation.citation_id
                    for citation in citations
                ],
                coverage=0.0,
            )

        if not citations:
            return CitationValidationResult(
                valid=False,
                validated_citations=[],
                invalid_citation_ids=[],
                coverage=0.0,
            )

        validated_citations: list[Citation] = []
        invalid_citation_ids: list[int] = []

        seen_citation_ids: set[int] = set()

        for citation in citations:
            citation_id = citation.citation_id

            if citation_id in seen_citation_ids:
                invalid_citation_ids.append(
                    citation_id
                )
                continue

            seen_citation_ids.add(citation_id)

            if citation_id < 1 or citation_id > len(results):
                invalid_citation_ids.append(
                    citation_id
                )
                continue

            result = results[citation_id - 1]

            if citation.chunk_id != result.chunk_id:
                invalid_citation_ids.append(
                    citation_id
                )
                continue

            if citation.document_id != result.document_id:
                invalid_citation_ids.append(
                    citation_id
                )
                continue

            validated_citations.append(citation)

        valid = not invalid_citation_ids

        coverage = (
            len(validated_citations)
            / len(citations)
            if citations
            else 0.0
        )

        return CitationValidationResult(
            valid=valid,
            validated_citations=validated_citations,
            invalid_citation_ids=invalid_citation_ids,
            coverage=coverage,
        )