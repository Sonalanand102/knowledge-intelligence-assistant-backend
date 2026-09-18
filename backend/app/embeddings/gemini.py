from __future__ import annotations

from typing import Any

from google.genai import types

from backend.app.embeddings.base import EmbeddingProvider


class GeminiEmbeddingProvider:
    """
    Gemini text embedding provider.

    Uses retrieval-specific task types:
    - RETRIEVAL_DOCUMENT for indexed chunks
    - RETRIEVAL_QUERY for user queries
    """

    DEFAULT_MODEL = "gemini-embedding-001"

    def __init__(
        self,
        client: Any,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.client = client
        self.model = model

    def embed_document(
        self,
        text: str,
    ) -> list[float]:
        """Generate an embedding for one document/chunk."""
        if not text.strip():
            raise ValueError(
                "Cannot generate embedding for empty document"
            )

        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
            ),
        )

        return self._extract_embedding(response)

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple documents/chunks."""
        if not texts:
            return []

        for text in texts:
            if not text.strip():
                raise ValueError(
                    "Cannot generate embedding for empty document"
                )

        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
            ),
        )

        return self._extract_embeddings(response)

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """Generate an embedding for one user query."""
        if not text.strip():
            raise ValueError(
                "Cannot generate embedding for empty query"
            )

        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
            ),
        )

        return self._extract_embedding(response)

    def embed_queries(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple user queries."""
        if not texts:
            return []

        for text in texts:
            if not text.strip():
                raise ValueError(
                    "Cannot generate embedding for empty query"
                )

        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
            ),
        )

        return self._extract_embeddings(response)

    @staticmethod
    def _extract_embeddings(
        response: Any,
    ) -> list[list[float]]:
        embeddings = getattr(
            response,
            "embeddings",
            None,
        )

        if not embeddings:
            raise ValueError(
                "Gemini returned no embeddings"
            )

        return [
            GeminiEmbeddingProvider._extract_embedding_object(
                embedding
            )
            for embedding in embeddings
        ]

    @staticmethod
    def _extract_embedding(
        response: Any,
    ) -> list[float]:
        embeddings = getattr(
            response,
            "embeddings",
            None,
        )

        if not embeddings:
            raise ValueError(
                "Gemini returned no embeddings"
            )

        return GeminiEmbeddingProvider._extract_embedding_object(
            embeddings[0]
        )

    @staticmethod
    def _extract_embedding_object(
        embedding: Any,
    ) -> list[float]:
        values = getattr(
            embedding,
            "values",
            None,
        )

        if values is None:
            raise ValueError(
                "Gemini returned an embedding without values"
            )

        vector = [
            float(value)
            for value in values
        ]

        if not vector:
            raise ValueError(
                "Gemini returned an empty embedding vector"
            )

        return vector