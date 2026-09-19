from dataclasses import dataclass

import pytest

from backend.app.retrieval.base import (
    VectorSearchResult,
)
from backend.app.retrieval.multi_query_retriever import (
    BaseMultiQueryGenerator,
    MultiQueryRetriever,
    GeminiMultiQueryGenerator,
    MultiQueryResponse,
)

from types import SimpleNamespace


class FakeMultiQueryGenerator(BaseMultiQueryGenerator):
    def __init__(
        self,
        queries: list[str],
    ) -> None:
        self.queries = queries
        self.received_query: str | None = None

    def _generate(
        self,
        query: str,
    ) -> list[str]:
        self.received_query = query
        return self.queries


class EmptyMultiQueryGenerator(BaseMultiQueryGenerator):
    def _generate(
        self,
        query: str,
    ) -> list[str]:
        return []


class FakeRetriever:
    def __init__(
        self,
        results_by_query: dict[str, list[VectorSearchResult]],
    ) -> None:
        self.results_by_query = results_by_query
        self.received_queries: list[str] = []
        self.received_top_k: list[int] = []

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        self.received_queries.append(query)
        self.received_top_k.append(top_k)

        return self.results_by_query.get(
            query,
            [],
        )[:top_k]


def make_result(
    chunk_id: str,
    score: float,
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        chunk_index=0,
        content=f"Content for {chunk_id}",
        score=score,
    )


def test_multi_query_generator_rejects_empty_query():
    generator = FakeMultiQueryGenerator(
        queries=["query one"],
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        generator.generate("   ")


def test_multi_query_generator_removes_duplicates():
    generator = FakeMultiQueryGenerator(
        queries=[
            "RAG retrieval",
            "rag retrieval",
            "RAG retrieval ",
            "vector search",
        ],
    )

    result = generator.generate(
        "What is RAG?"
    )

    assert result == [
        "RAG retrieval",
        "vector search",
    ]


def test_multi_query_generator_rejects_no_valid_queries():
    generator = EmptyMultiQueryGenerator()

    with pytest.raises(
        ValueError,
        match="Multi-query generator returned no valid queries",
    ):
        generator.generate("What is RAG?")


@pytest.mark.asyncio
async def test_multi_query_retriever_includes_original_query():
    generator = FakeMultiQueryGenerator(
        queries=[
            "retrieval augmented generation",
            "RAG retrieval",
        ],
    )

    retriever = FakeRetriever(
        results_by_query={
            "What is RAG?": [
                make_result("chunk-1", 0.9),
            ],
            "retrieval augmented generation": [
                make_result("chunk-2", 0.9),
            ],
            "RAG retrieval": [
                make_result("chunk-3", 0.9),
            ],
        },
    )

    multi_query_retriever = MultiQueryRetriever(
        retriever=retriever,
        query_generator=generator,
    )

    await multi_query_retriever.retrieve(
        query="What is RAG?",
        top_k=5,
    )

    assert retriever.received_queries == [
        "What is RAG?",
        "retrieval augmented generation",
        "RAG retrieval",
    ]


@pytest.mark.asyncio
async def test_multi_query_retriever_can_exclude_original_query():
    generator = FakeMultiQueryGenerator(
        queries=[
            "retrieval augmented generation",
            "RAG retrieval",
        ],
    )

    retriever = FakeRetriever(
        results_by_query={
            "retrieval augmented generation": [],
            "RAG retrieval": [],
        },
    )

    multi_query_retriever = MultiQueryRetriever(
        retriever=retriever,
        query_generator=generator,
        include_original=False,
    )

    await multi_query_retriever.retrieve(
        query="What is RAG?",
        top_k=5,
    )

    assert retriever.received_queries == [
        "retrieval augmented generation",
        "RAG retrieval",
    ]


@pytest.mark.asyncio
async def test_multi_query_retriever_passes_top_k():
    generator = FakeMultiQueryGenerator(
        queries=[
            "query one",
            "query two",
        ],
    )

    retriever = FakeRetriever(
        results_by_query={
            "What is RAG?": [],
            "query one": [],
            "query two": [],
        },
    )

    multi_query_retriever = MultiQueryRetriever(
        retriever=retriever,
        query_generator=generator,
    )

    await multi_query_retriever.retrieve(
        query="What is RAG?",
        top_k=3,
    )

    assert retriever.received_top_k == [
        3,
        3,
        3,
    ]


@pytest.mark.asyncio
async def test_multi_query_retriever_fuses_results():
    generator = FakeMultiQueryGenerator(
        queries=[
            "RAG retrieval",
            "RAG generation",
        ],
    )

    retriever = FakeRetriever(
        results_by_query={
            "What is RAG?": [
                make_result("chunk-1", 0.9),
                make_result("chunk-2", 0.8),
            ],
            "RAG retrieval": [
                make_result("chunk-1", 0.9),
                make_result("chunk-3", 0.7),
            ],
            "RAG generation": [
                make_result("chunk-1", 0.9),
                make_result("chunk-4", 0.6),
            ],
        },
    )

    multi_query_retriever = MultiQueryRetriever(
        retriever=retriever,
        query_generator=generator,
    )

    results = await multi_query_retriever.retrieve(
        query="What is RAG?",
        top_k=3,
    )

    assert len(results) == 3

    assert results[0].chunk_id == "chunk-1"


@pytest.mark.asyncio
async def test_multi_query_retriever_empty_results():
    generator = FakeMultiQueryGenerator(
        queries=[
            "query one",
            "query two",
        ],
    )

    retriever = FakeRetriever(
        results_by_query={},
    )

    multi_query_retriever = MultiQueryRetriever(
        retriever=retriever,
        query_generator=generator,
    )

    results = await multi_query_retriever.retrieve(
        query="What is RAG?",
        top_k=5,
    )

    assert results == []


@pytest.mark.asyncio
async def test_multi_query_retriever_rejects_empty_query():
    generator = FakeMultiQueryGenerator(
        queries=["query"],
    )

    retriever = FakeRetriever(
        results_by_query={},
    )

    multi_query_retriever = MultiQueryRetriever(
        retriever=retriever,
        query_generator=generator,
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        await multi_query_retriever.retrieve(
            query="   ",
            top_k=5,
        )


@pytest.mark.asyncio
async def test_multi_query_retriever_rejects_invalid_top_k():
    generator = FakeMultiQueryGenerator(
        queries=["query"],
    )

    retriever = FakeRetriever(
        results_by_query={},
    )

    multi_query_retriever = MultiQueryRetriever(
        retriever=retriever,
        query_generator=generator,
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        await multi_query_retriever.retrieve(
            query="What is RAG?",
            top_k=0,
        )


class FakeGeminiModels:
    def __init__(
        self,
        parsed: MultiQueryResponse | None,
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
        parsed: MultiQueryResponse | None,
    ) -> None:
        self.models = FakeGeminiModels(parsed)



def test_gemini_multi_query_generator_parses_structured_response():
    client = FakeGeminiClient(
        MultiQueryResponse(
            queries=[
                "RAG retrieval definition",
                "how does retrieval augmented generation work",
                "RAG architecture and retrieval pipeline",
            ]
        )
    )

    generator = GeminiMultiQueryGenerator(
        client=client,
    )

    result = generator.generate(
        "What is RAG?"
    )

    assert result == [
        "RAG retrieval definition",
        "how does retrieval augmented generation work",
        "RAG architecture and retrieval pipeline",
    ]


def test_gemini_multi_query_generator_removes_duplicate_queries():
    client = FakeGeminiClient(
        MultiQueryResponse(
            queries=[
                "RAG retrieval",
                "rag retrieval",
                "vector search",
            ]
        )
    )

    generator = GeminiMultiQueryGenerator(
        client=client,
    )

    result = generator.generate(
        "What is RAG?"
    )

    assert result == [
        "RAG retrieval",
        "vector search",
    ]


def test_gemini_multi_query_generator_calls_generate_content():
    client = FakeGeminiClient(
        MultiQueryResponse(
            queries=[
                "query one",
                "query two",
                "query three",
            ]
        )
    )

    generator = GeminiMultiQueryGenerator(
        client=client,
        model_name="fake-model",
    )

    result = generator.generate(
        "What is RAG?"
    )

    assert result == [
        "query one",
        "query two",
        "query three",
    ]

    assert len(client.models.calls) == 1

    call = client.models.calls[0]

    assert call["model"] == "fake-model"
    assert call["contents"] == "What is RAG?"

    config = call["config"]

    assert config.response_mime_type == "application/json"
    assert config.response_schema is MultiQueryResponse


def test_gemini_multi_query_generator_rejects_empty_response():
    client = FakeGeminiClient(
        parsed=None,
    )

    generator = GeminiMultiQueryGenerator(
        client=client,
    )

    with pytest.raises(
        ValueError,
        match="Gemini returned no structured multi-query response",
    ):
        generator.generate(
            "What is RAG?"
        )