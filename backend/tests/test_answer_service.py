import pytest

from backend.app.generation.answer_generation import (
    Citation,
    GeneratedAnswer,
)
from backend.app.generation.answer_service import (
    AnswerService,
)
from backend.app.retrieval.base import (
    VectorSearchResult,
)


def make_result(
    chunk_id: str,
    content: str,
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        chunk_index=0,
        content=content,
        score=0.9,
        metadata={
            "source": "test",
        },
    )


class FakeRetriever:
    def __init__(
        self,
        results: list[VectorSearchResult],
    ) -> None:
        self.results = results
        self.received_query: str | None = None
        self.received_top_k: int | None = None

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        self.received_query = query
        self.received_top_k = top_k

        return self.results


class FakeAnswerGenerator:
    def __init__(self) -> None:
        self.received_query: str | None = None
        self.received_results: list[
            VectorSearchResult
        ] | None = None

    def generate(
        self,
        query: str,
        results: list[VectorSearchResult],
    ) -> GeneratedAnswer:
        self.received_query = query
        self.received_results = results

        return GeneratedAnswer(
            answer="Test answer.",
            citations=[
                Citation(
                    citation_id=1,
                    chunk_id=results[0].chunk_id,
                    document_id=results[0].document_id,
                    metadata=dict(results[0].metadata),
                )
            ],
        )


@pytest.mark.asyncio
async def test_answer_service_retrieves_then_generates():
    results = [
        make_result(
            "chunk-1",
            "RAG combines retrieval and generation.",
        ),
        make_result(
            "chunk-2",
            "The retriever finds relevant context.",
        ),
    ]

    retriever = FakeRetriever(results)
    generator = FakeAnswerGenerator()

    service = AnswerService(
        retriever=retriever,
        answer_generator=generator,
    )

    result = await service.answer(
        query="What is RAG?",
        top_k=5,
    )

    assert result.answer == "Test answer."

    assert retriever.received_query == "What is RAG?"
    assert retriever.received_top_k == 5

    assert generator.received_query == "What is RAG?"
    assert generator.received_results == results


@pytest.mark.asyncio
async def test_answer_service_strips_query():
    results = [
        make_result(
            "chunk-1",
            "RAG combines retrieval and generation.",
        )
    ]

    retriever = FakeRetriever(results)
    generator = FakeAnswerGenerator()

    service = AnswerService(
        retriever=retriever,
        answer_generator=generator,
    )

    await service.answer(
        query="  What is RAG?  ",
        top_k=5,
    )

    assert retriever.received_query == "What is RAG?"
    assert generator.received_query == "What is RAG?"

@pytest.mark.asyncio
async def test_answer_service_rejects_empty_query():
    retriever = FakeRetriever([])
    generator = FakeAnswerGenerator()

    service = AnswerService(
        retriever=retriever,
        answer_generator=generator,
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        await service.answer(
            query="   ",
            top_k=5,
        )


@pytest.mark.asyncio
async def test_answer_service_rejects_invalid_top_k():
    retriever = FakeRetriever([])
    generator = FakeAnswerGenerator()

    service = AnswerService(
        retriever=retriever,
        answer_generator=generator,
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        await service.answer(
            query="What is RAG?",
            top_k=0,
        )

