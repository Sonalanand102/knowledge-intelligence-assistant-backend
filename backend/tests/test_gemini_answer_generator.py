from types import SimpleNamespace

import pytest

from backend.app.generation.gemini_answer_generator import (
    AnswerGenerationResponse,
    GeminiAnswerGenerator,
)
from backend.app.retrieval.base import VectorSearchResult


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


class FakeGeminiModels:
    def __init__(
        self,
        parsed: AnswerGenerationResponse | None,
    ) -> None:
        self.parsed = parsed
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)

        return SimpleNamespace(
            parsed=self.parsed,
        )


class FakeGeminiClient:
    def __init__(
        self,
        parsed: AnswerGenerationResponse | None,
    ) -> None:
        self.models = FakeGeminiModels(parsed)


def test_gemini_answer_generator_returns_answer():
    client = FakeGeminiClient(
        AnswerGenerationResponse(
            answer="RAG combines retrieval with generation.",
            citation_ids=[1],
        )
    )

    generator = GeminiAnswerGenerator(
        client=client,
    )

    result = generator.generate(
        query="What is RAG?",
        results=[
            make_result(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="RAG combines retrieval with generation.",
            ),
        ],
    )

    assert result.answer == (
        "RAG combines retrieval with generation."
    )

    assert len(result.citations) == 1
    assert result.citations[0].citation_id == 1
    assert result.citations[0].chunk_id == "chunk-1"
    assert result.citations[0].document_id == "doc-1"


def test_gemini_answer_generator_maps_citations_to_results():
    client = FakeGeminiClient(
        AnswerGenerationResponse(
            answer="RAG retrieves context before generation.",
            citation_ids=[2, 1],
        )
    )

    generator = GeminiAnswerGenerator(
        client=client,
    )

    result = generator.generate(
        query="How does RAG work?",
        results=[
            make_result(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="First context.",
            ),
            make_result(
                chunk_id="chunk-2",
                document_id="doc-2",
                content="Second context.",
            ),
        ],
    )

    assert [
        citation.chunk_id
        for citation in result.citations
    ] == [
        "chunk-2",
        "chunk-1",
    ]


def test_gemini_answer_generator_preserves_metadata():
    client = FakeGeminiClient(
        AnswerGenerationResponse(
            answer="RAG answer.",
            citation_ids=[1],
        )
    )

    generator = GeminiAnswerGenerator(
        client=client,
    )

    result = generator.generate(
        query="What is RAG?",
        results=[
            make_result(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="RAG answer.",
                metadata={
                    "file_name": "rag.pdf",
                    "page_number": 3,
                },
            ),
        ],
    )

    assert result.citations[0].metadata == {
        "file_name": "rag.pdf",
        "page_number": 3,
    }


def test_gemini_answer_generator_calls_structured_output():
    client = FakeGeminiClient(
        AnswerGenerationResponse(
            answer="RAG answer.",
            citation_ids=[1],
        )
    )

    generator = GeminiAnswerGenerator(
        client=client,
        model_name="fake-model",
    )

    generator.generate(
        query="What is RAG?",
        results=[
            make_result(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="RAG answer.",
            ),
        ],
    )

    assert len(client.models.calls) == 1

    call = client.models.calls[0]

    assert call["model"] == "fake-model"

    config = call["config"]

    assert config.response_mime_type == "application/json"
    assert (
        config.response_schema
        is AnswerGenerationResponse
    )


def test_gemini_answer_generator_rejects_empty_structured_response():
    client = FakeGeminiClient(
        parsed=None,
    )

    generator = GeminiAnswerGenerator(
        client=client,
    )

    with pytest.raises(
        ValueError,
        match="Gemini returned no structured answer response",
    ):
        generator.generate(
            query="What is RAG?",
            results=[
                make_result(
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    content="RAG answer.",
                ),
            ],
        )


def test_gemini_answer_generator_rejects_empty_answer():
    client = FakeGeminiClient(
        AnswerGenerationResponse(
            answer="   ",
            citation_ids=[1],
        )
    )

    generator = GeminiAnswerGenerator(
        client=client,
    )

    with pytest.raises(
        ValueError,
        match="Gemini returned an empty answer",
    ):
        generator.generate(
            query="What is RAG?",
            results=[
                make_result(
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    content="RAG answer.",
                ),
            ],
        )


def test_gemini_answer_generator_requires_citations():
    client = FakeGeminiClient(
        AnswerGenerationResponse(
            answer="RAG answer.",
            citation_ids=[],
        )
    )

    generator = GeminiAnswerGenerator(
        client=client,
    )

    with pytest.raises(
        ValueError,
        match="Gemini returned no citation IDs",
    ):
        generator.generate(
            query="What is RAG?",
            results=[
                make_result(
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    content="RAG answer.",
                ),
            ],
        )


def test_gemini_answer_generator_rejects_invalid_citation_ids():
    client = FakeGeminiClient(
        AnswerGenerationResponse(
            answer="RAG answer.",
            citation_ids=[3],
        )
    )

    generator = GeminiAnswerGenerator(
        client=client,
    )

    with pytest.raises(
        ValueError,
        match="invalid citation IDs",
    ):
        generator.generate(
            query="What is RAG?",
            results=[
                make_result(
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    content="RAG answer.",
                ),
                make_result(
                    chunk_id="chunk-2",
                    document_id="doc-1",
                    content="More RAG context.",
                ),
            ],
        )


def test_gemini_answer_generator_deduplicates_citations():
    client = FakeGeminiClient(
        AnswerGenerationResponse(
            answer="RAG answer.",
            citation_ids=[1, 1, 2, 2],
        )
    )

    generator = GeminiAnswerGenerator(
        client=client,
    )

    result = generator.generate(
        query="What is RAG?",
        results=[
            make_result(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="First context.",
            ),
            make_result(
                chunk_id="chunk-2",
                document_id="doc-1",
                content="Second context.",
            ),
        ],
    )

    assert [
        citation.citation_id
        for citation in result.citations
    ] == [1, 2]