from dataclasses import dataclass

from backend.app.embeddings.gemini import GeminiEmbeddingProvider


@dataclass
class FakeEmbedding:
    values: list[float]


@dataclass
class FakeResponse:
    embeddings: list[FakeEmbedding]


class FakeModels:
    def __init__(self):
        self.calls = []

    def embed_content(
        self,
        *,
        model,
        contents,
        config=None,
    ):
        self.calls.append(
            {
                "model": model,
                "contents": contents,
                "config": config,
            }
        )

        if isinstance(contents, list):
            return FakeResponse(
                embeddings=[
                    FakeEmbedding(
                        values=[
                            float(index + 1),
                            2.0,
                            3.0,
                        ]
                    )
                    for index, _ in enumerate(contents)
                ]
            )

        return FakeResponse(
            embeddings=[
                FakeEmbedding(
                    values=[1.0, 2.0, 3.0]
                )
            ]
        )


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


def test_embed_document_returns_vector():
    client = FakeClient()

    provider = GeminiEmbeddingProvider(
        client=client,
        model="gemini-embedding-001",
    )

    result = provider.embed_document(
        "Hello world"
    )

    assert result == [1.0, 2.0, 3.0]


def test_embed_document_uses_retrieval_document_task():
    client = FakeClient()

    provider = GeminiEmbeddingProvider(
        client=client,
        model="gemini-embedding-001",
    )

    provider.embed_document("Hello world")

    call = client.models.calls[0]

    assert call["model"] == "gemini-embedding-001"
    assert call["contents"] == "Hello world"
    assert getattr(
        call["config"],
        "task_type",
        None,
    ) == "RETRIEVAL_DOCUMENT"


def test_embed_documents_returns_vectors_in_same_order():
    client = FakeClient()

    provider = GeminiEmbeddingProvider(
        client=client,
        model="gemini-embedding-001",
    )

    result = provider.embed_documents(
        [
            "First",
            "Second",
            "Third",
        ]
    )

    assert result == [
        [1.0, 2.0, 3.0],
        [2.0, 2.0, 3.0],
        [3.0, 2.0, 3.0],
    ]


def test_embed_documents_uses_retrieval_document_task():
    client = FakeClient()

    provider = GeminiEmbeddingProvider(
        client=client,
        model="gemini-embedding-001",
    )

    provider.embed_documents(
        ["First", "Second"]
    )

    call = client.models.calls[0]

    assert call["contents"] == [
        "First",
        "Second",
    ]

    assert getattr(
        call["config"],
        "task_type",
        None,
    ) == "RETRIEVAL_DOCUMENT"


def test_embed_query_returns_vector():
    client = FakeClient()

    provider = GeminiEmbeddingProvider(
        client=client,
        model="gemini-embedding-001",
    )

    result = provider.embed_query(
        "What is RAG?"
    )

    assert result == [1.0, 2.0, 3.0]


def test_embed_query_uses_retrieval_query_task():
    client = FakeClient()

    provider = GeminiEmbeddingProvider(
        client=client,
        model="gemini-embedding-001",
    )

    provider.embed_query("What is RAG?")

    call = client.models.calls[0]

    assert call["contents"] == "What is RAG?"

    assert getattr(
        call["config"],
        "task_type",
        None,
    ) == "RETRIEVAL_QUERY"


def test_embed_queries_returns_vectors_in_same_order():
    client = FakeClient()

    provider = GeminiEmbeddingProvider(
        client=client,
        model="gemini-embedding-001",
    )

    result = provider.embed_queries(
        [
            "What is RAG?",
            "How does retrieval work?",
        ]
    )

    assert result == [
        [1.0, 2.0, 3.0],
        [2.0, 2.0, 3.0],
    ]


def test_embed_queries_uses_retrieval_query_task():
    client = FakeClient()

    provider = GeminiEmbeddingProvider(
        client=client,
        model="gemini-embedding-001",
    )

    provider.embed_queries(
        [
            "What is RAG?",
            "How does retrieval work?",
        ]
    )

    call = client.models.calls[0]

    assert call["contents"] == [
        "What is RAG?",
        "How does retrieval work?",
    ]

    assert getattr(
        call["config"],
        "task_type",
        None,
    ) == "RETRIEVAL_QUERY"


def test_empty_document_batch_returns_empty_list():
    client = FakeClient()

    provider = GeminiEmbeddingProvider(
        client=client,
    )

    assert provider.embed_documents([]) == []
    assert client.models.calls == []


def test_empty_query_batch_returns_empty_list():
    client = FakeClient()

    provider = GeminiEmbeddingProvider(
        client=client,
    )

    assert provider.embed_queries([]) == []
    assert client.models.calls == []


def test_empty_document_is_rejected():
    client = FakeClient()

    provider = GeminiEmbeddingProvider(
        client=client,
    )

    try:
        provider.embed_document("")
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_empty_query_is_rejected():
    client = FakeClient()

    provider = GeminiEmbeddingProvider(
        client=client,
    )

    try:
        provider.embed_query("")
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected ValueError"
        )