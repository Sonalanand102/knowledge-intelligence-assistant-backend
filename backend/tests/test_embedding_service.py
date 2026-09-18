from backend.app.embeddings.service import (
    EmbeddedChunk,
    EmbeddingService,
)
from backend.app.ingestion.models.chunk_document import ChunkDocument

class FakeEmbeddingProvider:
    def __init__(self):
        self.received_texts: list[str] = []

    def embed_document(self, text: str) -> list[float]:
        self.received_texts.append(text)
        return [1.0, 2.0, 3.0]

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        self.received_texts.extend(texts)

        return [
            [float(index + 1), 2.0, 3.0]
            for index, _ in enumerate(texts)
        ]

    def embed_query(self, text: str) -> list[float]:
        self.received_texts.append(text)
        return [4.0, 5.0, 6.0]

    def embed_queries(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        self.received_texts.extend(texts)

        return [
            [4.0, 5.0, 6.0]
            for _ in texts
        ]

def make_chunk(
    content: str,
    chunk_index: int = 0,
    metadata: dict | None = None,
) -> ChunkDocument:
    return ChunkDocument(
        content=content,
        document_id="document-1",
        chunk_index=chunk_index,
        metadata=metadata or {},
    )


def test_embed_chunks_returns_embedded_chunks():
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider)

    chunks = [
        make_chunk("Hello"),
        make_chunk("World", chunk_index=1),
    ]

    result = service.embed_chunks(chunks)

    assert len(result) == 2
    assert all(
        isinstance(item, EmbeddedChunk)
        for item in result
    )


def test_embed_chunks_uses_batch_embedding():
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider)

    chunks = [
        make_chunk("Hello"),
        make_chunk("World", chunk_index=1),
    ]

    service.embed_chunks(chunks)

    assert provider.received_texts == [
        "Hello",
        "World",
    ]


def test_embedding_is_attached_to_correct_chunk():
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider)

    chunks = [
        make_chunk("Hello"),
        make_chunk("World", chunk_index=1),
    ]

    result = service.embed_chunks(chunks)

    assert result[0].chunk.content == "Hello"
    assert result[0].embedding == [1.0, 2.0, 3.0]

    assert result[1].chunk.content == "World"
    assert result[1].embedding == [2.0, 2.0, 3.0]


def test_chunk_metadata_is_preserved():
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider)

    chunks = [
        make_chunk(
            "Hello",
            metadata={
                "element_ids": ["element-1"],
                "page_number": 3,
                "relationships": [],
            },
        ),
    ]

    result = service.embed_chunks(chunks)

    assert result[0].chunk.metadata == {
        "element_ids": ["element-1"],
        "page_number": 3,
        "relationships": [],
    }


def test_empty_chunk_list_returns_empty_list():
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider)

    result = service.embed_chunks([])

    assert result == []
    assert provider.received_texts == []


def test_empty_chunk_content_is_rejected():
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider)

    chunks = [
        make_chunk(""),
    ]

    try:
        service.embed_chunks(chunks)
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError")


def test_whitespace_only_chunk_content_is_rejected():
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider)

    chunks = [
        make_chunk("   \n\t"),
    ]

    try:
        service.embed_chunks(chunks)
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError")


def test_original_chunks_are_not_modified():
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider)

    chunks = [
        make_chunk(
            "Hello",
            metadata={"page_number": 3},
        ),
    ]

    original_content = chunks[0].content
    original_metadata = chunks[0].metadata.copy()

    service.embed_chunks(chunks)

    assert chunks[0].content == original_content
    assert chunks[0].metadata == original_metadata


def test_embedding_dimension_is_consistent():
    class InconsistentProvider:
        def embed_document(
            self,
            text: str,
        ) -> list[float]:
            return [1.0, 2.0]

        def embed_documents(
            self,
            texts: list[str],
        ) -> list[list[float]]:
            return [
                [1.0, 2.0, 3.0],
                [4.0, 5.0],
            ]

        def embed_query(
            self,
            text: str,
        ) -> list[float]:
            return [1.0, 2.0, 3.0]

        def embed_queries(
            self,
            texts: list[str],
        ) -> list[list[float]]:
            return [
                [1.0, 2.0, 3.0]
                for _ in texts
            ]

    service = EmbeddingService(
        InconsistentProvider()
    )

    chunks = [
        make_chunk("Hello"),
        make_chunk("World", chunk_index=1),
    ]

    try:
        service.embed_chunks(chunks)
    except ValueError as exc:
        assert "dimension" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError")
    
def test_embed_query_returns_embedding():
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider)

    result = service.embed_query(
        "What is retrieval augmented generation?"
    )

    assert result == [4.0, 5.0, 6.0]


def test_embed_queries_returns_embeddings_in_same_order():
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider)

    result = service.embed_queries(
        [
            "What is RAG?",
            "How does retrieval work?",
        ]
    )

    assert result == [
        [4.0, 5.0, 6.0],
        [4.0, 5.0, 6.0],
    ]


def test_empty_query_is_rejected():
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider)

    try:
        service.embed_query("")
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError")


def test_whitespace_only_query_is_rejected():
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider)

    try:
        service.embed_query("   ")
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError")


def test_empty_query_batch_returns_empty_list():
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider)

    result = service.embed_queries([])

    assert result == []