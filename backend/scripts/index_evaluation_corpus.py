from __future__ import annotations

from google import genai
from backend.app.core.config import settings

from backend.app.embeddings.gemini import GeminiEmbeddingProvider
from backend.app.embeddings.local import LocalEmbeddingProvider
from backend.app.embeddings.service import EmbeddingService
from backend.app.evaluation.datasets.embedding_dataset import (
    get_embedding_evaluation_corpus,
)
from backend.app.ingestion.models.chunk_document import ChunkDocument
from backend.app.retrieval.qdrant import create_qdrant_client
from backend.app.retrieval.qdrant_store import QdrantVectorStore
from backend.app.retrieval.vector_indexer import VectorIndexer


COLLECTION_NAME = "knowledge_chunks"


def build_chunks() -> list[ChunkDocument]:
    corpus = get_embedding_evaluation_corpus()

    return [
        ChunkDocument(
            content=content,
            document_id=chunk_id,
            chunk_index=0,
            metadata={
                "source": "embedding_evaluation_corpus",
                "source_type": "evaluation",
            },
        )
        for chunk_id, content in corpus.items()
    ]

def create_embedding_service() -> EmbeddingService:
    if settings.embedding_provider == "gemini":
        client = genai.Client(
            api_key=settings.gemini_api_key,
        )

        provider = GeminiEmbeddingProvider(
            client=client,
        )

        return EmbeddingService(
            provider=provider,
        )

    if settings.embedding_provider == "local":
        return EmbeddingService(
            provider=LocalEmbeddingProvider(),
        )

    raise ValueError(
        "Unsupported embedding provider: "
        f"{settings.embedding_provider}"
    )

async def main() -> None:

    chunks = build_chunks()

    embedding_service = create_embedding_service()

    qdrant_client = create_qdrant_client()

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
    )

    indexer = VectorIndexer(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    try:
        print("\nKIA Vector Indexing")
        print("-" * 60)
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Chunks: {len(chunks)}")

        provider_name = os.getenv(
            "EMBEDDING_PROVIDER",
            "gemini",
        ).lower()

        print(f"Embedding provider: {provider_name}")

        await indexer.index(chunks)

        collection_info = (
            await qdrant_client.get_collection(
                collection_name=COLLECTION_NAME,
            )
        )

        print("\nIndexing complete")
        print(
            f"Points in collection: "
            f"{collection_info.points_count}"
        )

    finally:
        await qdrant_client.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())