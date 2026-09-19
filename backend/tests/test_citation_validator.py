from backend.app.evaluation.citation_validator import (
    CitationValidator,
)
from backend.app.generation.answer_generation import (
    Citation,
)
from backend.app.retrieval.base import (
    VectorSearchResult,
)


def make_result(
    chunk_id: str,
    document_id: str,
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        content=f"Content for {chunk_id}",
        score=0.9,
        metadata={},
    )


def make_citation(
    citation_id: int,
    chunk_id: str,
    document_id: str,
) -> Citation:
    return Citation(
        citation_id=citation_id,
        chunk_id=chunk_id,
        document_id=document_id,
        metadata={},
    )


def test_valid_citations_pass():
    results = [
        make_result("chunk-1", "doc-1"),
        make_result("chunk-2", "doc-1"),
    ]

    citations = [
        make_citation(1, "chunk-1", "doc-1"),
        make_citation(2, "chunk-2", "doc-1"),
    ]

    validator = CitationValidator()

    result = validator.validate(
        citations=citations,
        results=results,
    )

    assert result.valid is True
    assert result.invalid_citation_ids == []
    assert result.coverage == 1.0
    assert result.validated_citations == citations


def test_invalid_citation_id_fails():
    results = [
        make_result("chunk-1", "doc-1"),
    ]

    citations = [
        make_citation(2, "chunk-1", "doc-1"),
    ]

    validator = CitationValidator()

    result = validator.validate(
        citations=citations,
        results=results,
    )

    assert result.valid is False
    assert result.invalid_citation_ids == [2]
    assert result.coverage == 0.0


def test_chunk_mismatch_fails():
    results = [
        make_result("chunk-1", "doc-1"),
    ]

    citations = [
        make_citation(1, "chunk-999", "doc-1"),
    ]

    validator = CitationValidator()

    result = validator.validate(
        citations=citations,
        results=results,
    )

    assert result.valid is False
    assert result.invalid_citation_ids == [1]
    assert result.coverage == 0.0


def test_document_mismatch_fails():
    results = [
        make_result("chunk-1", "doc-1"),
    ]

    citations = [
        make_citation(1, "chunk-1", "doc-999"),
    ]

    validator = CitationValidator()

    result = validator.validate(
        citations=citations,
        results=results,
    )

    assert result.valid is False
    assert result.invalid_citation_ids == [1]
    assert result.coverage == 0.0


def test_duplicate_citation_ids_are_invalid():
    results = [
        make_result("chunk-1", "doc-1"),
    ]

    citations = [
        make_citation(1, "chunk-1", "doc-1"),
        make_citation(1, "chunk-1", "doc-1"),
    ]

    validator = CitationValidator()

    result = validator.validate(
        citations=citations,
        results=results,
    )

    assert result.valid is False
    assert result.invalid_citation_ids == [1]
    assert result.coverage == 0.5


def test_empty_citations_are_invalid_for_non_empty_results():
    results = [
        make_result("chunk-1", "doc-1"),
    ]

    validator = CitationValidator()

    result = validator.validate(
        citations=[],
        results=results,
    )

    assert result.valid is False
    assert result.invalid_citation_ids == []
    assert result.coverage == 0.0


def test_empty_results_with_empty_citations_are_valid():
    validator = CitationValidator()

    result = validator.validate(
        citations=[],
        results=[],
    )

    assert result.valid is True
    assert result.invalid_citation_ids == []
    assert result.coverage == 0.0


def test_empty_results_with_citations_are_invalid():
    citations = [
        make_citation(1, "chunk-1", "doc-1"),
    ]

    validator = CitationValidator()

    result = validator.validate(
        citations=citations,
        results=[],
    )

    assert result.valid is False
    assert result.invalid_citation_ids == [1]
    assert result.coverage == 0.0


def test_partial_invalid_citations_report_coverage():
    results = [
        make_result("chunk-1", "doc-1"),
        make_result("chunk-2", "doc-1"),
    ]

    citations = [
        make_citation(1, "chunk-1", "doc-1"),
        make_citation(3, "chunk-3", "doc-1"),
    ]

    validator = CitationValidator()

    result = validator.validate(
        citations=citations,
        results=results,
    )

    assert result.valid is False
    assert result.invalid_citation_ids == [3]
    assert result.coverage == 0.5
    assert result.validated_citations == [
        citations[0],
    ]