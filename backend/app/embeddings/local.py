from __future__ import annotations

from typing import Any

from backend.app.embeddings.base import EmbeddingProvider


class LocalEmbeddingProvider:
    """
    Local embedding provider using Sentence Transformers.

    The model can be injected for testing so tests do not need
    to download or load a real model.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model: Any | None = None,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        if model is not None:
            self.model = model
            return

        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    @staticmethod
    def _to_vector(
        values: Any,
    ) -> list[float]:
        """
        Convert NumPy-like or Python-list model output
        into a plain list of floats.
        """
        if hasattr(values, "tolist"):
            values = values.tolist()

        return [
            float(value)
            for value in values
        ]

    def embed_document(
        self,
        text: str,
    ) -> list[float]:
        """Generate an embedding for one document/chunk."""
        if not text.strip():
            raise ValueError(
                "Cannot generate embedding for empty document"
            )

        embedding = self.model.encode_document(
            text,
            convert_to_numpy=True,
        )

        return self._to_vector(embedding)

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

        embeddings = self.model.encode_document(
            texts,
            convert_to_numpy=True,
        )

        return [
            self._to_vector(embedding)
            for embedding in embeddings
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """Generate an embedding for one user query."""
        if not text.strip():
            raise ValueError(
                "Cannot generate embedding for empty query"
            )

        embedding = self.model.encode_query(
            text,
            convert_to_numpy=True,
        )

        return self._to_vector(embedding)

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

        embeddings = self.model.encode_query(
            texts,
            convert_to_numpy=True,
        )

        return [
            self._to_vector(embedding)
            for embedding in embeddings
        ]