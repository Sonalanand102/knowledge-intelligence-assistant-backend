from backend.app.embeddings.local import LocalEmbeddingProvider


class FakeSentenceTransformer:
    def __init__(self):
        self.calls = []

    def encode_document(self, texts, **kwargs):
        self.calls.append(
            {
                "method": "encode_document",
                "texts": texts,
                "kwargs": kwargs,
            }
        )

        if isinstance(texts, str):
            return [0.1, 0.2, 0.3]

        return [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]

    def encode_query(self, texts, **kwargs):
        self.calls.append(
            {
                "method": "encode_query",
                "texts": texts,
                "kwargs": kwargs,
            }
        )

        if isinstance(texts, str):
            return [0.7, 0.8, 0.9]

        return [
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
        ]


def test_embed_document_returns_vector():
    model = FakeSentenceTransformer()

    provider = LocalEmbeddingProvider(
        model=model,
    )

    result = provider.embed_document(
        "Hello world"
    )

    assert result == [0.1, 0.2, 0.3]


def test_embed_document_uses_encode_document():
    model = FakeSentenceTransformer()

    provider = LocalEmbeddingProvider(
        model=model,
    )

    provider.embed_document("Hello world")

    assert len(model.calls) == 1
    assert model.calls[0]["method"] == "encode_document"
    assert model.calls[0]["texts"] == "Hello world"


def test_embed_documents_returns_vectors_in_same_order():
    model = FakeSentenceTransformer()

    provider = LocalEmbeddingProvider(
        model=model,
    )

    result = provider.embed_documents(
        [
            "First",
            "Second",
        ]
    )

    assert result == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]


def test_embed_documents_uses_batch_encode_document():
    model = FakeSentenceTransformer()

    provider = LocalEmbeddingProvider(
        model=model,
    )

    provider.embed_documents(
        [
            "First",
            "Second",
        ]
    )

    assert len(model.calls) == 1
    assert model.calls[0]["method"] == "encode_document"
    assert model.calls[0]["texts"] == [
        "First",
        "Second",
    ]


def test_embed_query_returns_vector():
    model = FakeSentenceTransformer()

    provider = LocalEmbeddingProvider(
        model=model,
    )

    result = provider.embed_query(
        "What is RAG?"
    )

    assert result == [0.7, 0.8, 0.9]


def test_embed_query_uses_encode_query():
    model = FakeSentenceTransformer()

    provider = LocalEmbeddingProvider(
        model=model,
    )

    provider.embed_query("What is RAG?")

    assert len(model.calls) == 1
    assert model.calls[0]["method"] == "encode_query"
    assert model.calls[0]["texts"] == "What is RAG?"


def test_embed_queries_returns_vectors_in_same_order():
    model = FakeSentenceTransformer()

    provider = LocalEmbeddingProvider(
        model=model,
    )

    result = provider.embed_queries(
        [
            "What is RAG?",
            "How does retrieval work?",
        ]
    )

    assert result == [
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
    ]


def test_embed_queries_uses_batch_encode_query():
    model = FakeSentenceTransformer()

    provider = LocalEmbeddingProvider(
        model=model,
    )

    provider.embed_queries(
        [
            "What is RAG?",
            "How does retrieval work?",
        ]
    )

    assert len(model.calls) == 1
    assert model.calls[0]["method"] == "encode_query"
    assert model.calls[0]["texts"] == [
        "What is RAG?",
        "How does retrieval work?",
    ]


def test_empty_document_batch_returns_empty_list():
    model = FakeSentenceTransformer()

    provider = LocalEmbeddingProvider(
        model=model,
    )

    assert provider.embed_documents([]) == []
    assert model.calls == []


def test_empty_query_batch_returns_empty_list():
    model = FakeSentenceTransformer()

    provider = LocalEmbeddingProvider(
        model=model,
    )

    assert provider.embed_queries([]) == []
    assert model.calls == []


def test_empty_document_is_rejected():
    model = FakeSentenceTransformer()

    provider = LocalEmbeddingProvider(
        model=model,
    )

    try:
        provider.embed_document("")
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_whitespace_document_is_rejected():
    model = FakeSentenceTransformer()

    provider = LocalEmbeddingProvider(
        model=model,
    )

    try:
        provider.embed_document("   ")
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_empty_query_is_rejected():
    model = FakeSentenceTransformer()

    provider = LocalEmbeddingProvider(
        model=model,
    )

    try:
        provider.embed_query("")
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_whitespace_query_is_rejected():
    model = FakeSentenceTransformer()

    provider = LocalEmbeddingProvider(
        model=model,
    )

    try:
        provider.embed_query("   ")
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected ValueError"
        )