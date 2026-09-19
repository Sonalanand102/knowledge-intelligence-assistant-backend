import pytest

from backend.app.generation.answer_generation import (
    BaseAnswerGenerator,
    GroundedAnswerGenerator,
)
from backend.app.retrieval.base import (
    VectorSearchResult,
)


def make_result(
    chunk_id: str,
    document_id: str,
    content: str,
    metadata: dict | None = None,
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        content=content,
        score=0.9,
        metadata=metadata or {},
    )


def test_answer_generator_rejects_empty_query():
    generator = GroundedAnswerGenerator()

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        generator.generate(
            query="   ",
            results=[],
        )


def test_answer_generator_handles_empty_results():
    generator = GroundedAnswerGenerator()

    result = generator.generate(
        query="What is RAG?",
        results=[],
    )

    assert (
        result.answer
        == "I could not find enough relevant information "
        "to answer the question."
    )

    assert result.citations == []


def test_grounded_generator_creates_citations():
    generator = GroundedAnswerGenerator()

    results = [
        make_result(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="RAG combines retrieval with generation.",
            metadata={
                "file_name": "rag.pdf",
                "page_number": 1,
            },
        ),
        make_result(
            chunk_id="chunk-2",
            document_id="doc-1",
            content="The retriever finds relevant context.",
            metadata={
                "file_name": "rag.pdf",
                "page_number": 2,
            },
        ),
    ]

    result = generator.generate(
        query="What is RAG?",
        results=results,
    )

    assert len(result.citations) == 2

    assert result.citations[0].citation_id == 1
    assert result.citations[0].chunk_id == "chunk-1"
    assert result.citations[0].document_id == "doc-1"

    assert result.citations[1].citation_id == 2
    assert result.citations[1].chunk_id == "chunk-2"


def test_grounded_generator_preserves_metadata():
    generator = GroundedAnswerGenerator()

    results = [
        make_result(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="RAG content",
            metadata={
                "file_name": "rag.pdf",
                "page_number": 4,
                "source_type": "pdf",
            },
        ),
    ]

    result = generator.generate(
        query="What is RAG?",
        results=results,
    )

    assert result.citations[0].metadata == {
        "file_name": "rag.pdf",
        "page_number": 4,
        "source_type": "pdf",
    }


def test_grounded_generator_includes_context_in_answer():
    generator = GroundedAnswerGenerator()

    results = [
        make_result(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="RAG combines retrieval with generation.",
        ),
        make_result(
            chunk_id="chunk-2",
            document_id="doc-1",
            content="The retriever finds relevant context.",
        ),
    ]

    result = generator.generate(
        query="What is RAG?",
        results=results,
    )

    assert (
        "[1] RAG combines retrieval with generation."
        in result.answer
    )

    assert (
        "[2] The retriever finds relevant context."
        in result.answer
    )


class FakeAnswerGenerator(BaseAnswerGenerator):
    def _generate(
        self,
        query: str,
        results: list[VectorSearchResult],
    ):
        return GroundedAnswerGenerator().generate(
            query=query,
            results=results,
        )


def test_base_answer_generator_contract():
    generator = FakeAnswerGenerator()

    result = generator.generate(
        query="What is RAG?",
        results=[
            make_result(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="RAG content.",
            ),
        ],
    )

    assert result.answer
    assert len(result.citations) == 1