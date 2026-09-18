from backend.app.embeddings.base import EmbeddingProvider


class FakeEmbeddingProvider:
    def embed_document(self, text: str) -> list[float]:
        return [1.0, 2.0, 3.0]

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            [1.0, 2.0, 3.0]
            for _ in texts
        ]

    def embed_query(self, text: str) -> list[float]:
        return [4.0, 5.0, 6.0]

    def embed_queries(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            [4.0, 5.0, 6.0]
            for _ in texts
        ]


def test_embedding_provider_is_protocol():
    provider = FakeEmbeddingProvider()

    assert isinstance(provider, EmbeddingProvider)


def test_provider_returns_document_embedding():
    provider = FakeEmbeddingProvider()

    result = provider.embed_document("Hello world")

    assert isinstance(result, list)
    assert all(isinstance(value, float) for value in result)
    assert len(result) == 3


def test_provider_returns_document_batch_embeddings():
    provider = FakeEmbeddingProvider()

    result = provider.embed_documents(
        ["Hello", "World"]
    )

    assert len(result) == 2
    assert all(isinstance(vector, list) for vector in result)


def test_provider_returns_query_embedding():
    provider = FakeEmbeddingProvider()

    result = provider.embed_query(
        "What is this about?"
    )

    assert isinstance(result, list)
    assert all(isinstance(value, float) for value in result)
    assert len(result) == 3


def test_provider_returns_query_batch_embeddings():
    provider = FakeEmbeddingProvider()

    result = provider.embed_queries(
        [
            "What is this about?",
            "How does it work?",
        ]
    )

    assert len(result) == 2
    assert all(isinstance(vector, list) for vector in result)


def test_query_and_document_embeddings_are_separate_operations():
    provider = FakeEmbeddingProvider()

    document_embedding = provider.embed_document(
        "Python is a programming language."
    )

    query_embedding = provider.embed_query(
        "What is Python?"
    )

    assert document_embedding != query_embedding