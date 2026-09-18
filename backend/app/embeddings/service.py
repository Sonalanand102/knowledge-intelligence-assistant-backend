from __future__ import annotations

from dataclasses import dataclass

from backend.app.embeddings.base import EmbeddingProvider
from backend.app.ingestion.models.chunk_document import ChunkDocument


@dataclass(frozen=True)
class EmbeddedChunk:
    """
    A chunk together with its embedding vector.
    """

    chunk: ChunkDocument
    embedding: list[float]


class EmbeddingService:
    """
    Orchestrates document and query embedding generation.
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
    ) -> None:
        self.provider = provider

    def embed_chunks(
        self,
        chunks: list[ChunkDocument],
    ) -> list[EmbeddedChunk]:
        """
        Generate embeddings for document chunks.

        The original ChunkDocuments are not modified.
        """
        if not chunks:
            return []

        texts: list[str] = []

        for chunk in chunks:
            if not chunk.content.strip():
                raise ValueError(
                    "Cannot generate embedding for empty chunk content"
                )

            texts.append(chunk.content)

        embeddings = self.provider.embed_documents(texts)

        self._validate_embedding_count(
            embeddings,
            expected_count=len(chunks),
        )

        self._validate_embedding_dimensions(
            embeddings,
        )

        return [
            EmbeddedChunk(
                chunk=chunk,
                embedding=embedding,
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        Generate an embedding for a single user query.
        """
        if not query.strip():
            raise ValueError(
                "Cannot generate embedding for empty query"
            )

        embedding = self.provider.embed_query(query)

        self._validate_embedding_dimensions(
            [embedding],
        )

        return embedding

    def embed_queries(
        self,
        queries: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple user queries.
        """
        if not queries:
            return []

        for query in queries:
            if not query.strip():
                raise ValueError(
                    "Cannot generate embedding for empty query"
                )

        embeddings = self.provider.embed_queries(queries)

        self._validate_embedding_count(
            embeddings,
            expected_count=len(queries),
        )

        self._validate_embedding_dimensions(
            embeddings,
        )

        return embeddings

    @staticmethod
    def _validate_embedding_count(
        embeddings: list[list[float]],
        expected_count: int,
    ) -> None:
        if len(embeddings) != expected_count:
            raise ValueError(
                "Embedding provider returned a different number "
                "of embeddings than inputs"
            )

    @staticmethod
    def _validate_embedding_dimensions(
        embeddings: list[list[float]],
    ) -> None:
        if not embeddings:
            return

        expected_dimension = len(embeddings[0])

        if expected_dimension == 0:
            raise ValueError(
                "Embedding provider returned an empty vector"
            )

        for embedding in embeddings:
            if len(embedding) != expected_dimension:
                raise ValueError(
                    "Embedding vectors have inconsistent dimensions"
                )