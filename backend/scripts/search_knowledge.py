from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv
from google import genai

from backend.app.embeddings.gemini import GeminiEmbeddingProvider
from backend.app.embeddings.service import EmbeddingService
from backend.app.retrieval.dense_retriever import DenseRetriever
from backend.app.retrieval.qdrant import create_qdrant_client
from backend.app.retrieval.qdrant_store import QdrantVectorStore


DEFAULT_TOP_K = 5


def create_gemini_embedding_service() -> EmbeddingService:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set"
        )

    client = genai.Client(
        api_key=api_key,
    )

    provider = GeminiEmbeddingProvider(
        client=client,
    )

    return EmbeddingService(
        provider=provider,
    )


async def search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> None:
    if not query.strip():
        raise ValueError(
            "Query cannot be empty"
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero"
        )

    embedding_service = (
        create_gemini_embedding_service()
    )

    qdrant_client = create_qdrant_client()

    collection_name = os.getenv(
        "QDRANT_COLLECTION_NAME",
        "knowledge_chunks",
    )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
    )

    retriever = DenseRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    try:
        results = await retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        print("\nKnowledge Search")
        print("=" * 70)
        print(f"Query: {query}")
        print(f"Top K: {top_k}")
        print(f"Collection: {collection_name}")
        print("=" * 70)

        if not results:
            print("\nNo results found.")
            return

        for index, result in enumerate(
            results,
            start=1,
        ):
            print(f"\n[{index}]")
            print(f"Score: {result.score:.6f}")
            print(f"Chunk ID: {result.chunk_id}")

            if result.metadata:
                print(
                    f"Metadata: {result.metadata}"
                )

            print("Content:")
            print(result.content)

    finally:
        await qdrant_client.close()


async def main() -> None:
    load_dotenv()

    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: "
            "uv run python -m backend.scripts.search_knowledge "
            "\"your question\""
        )

    query = sys.argv[1]

    top_k = DEFAULT_TOP_K

    if len(sys.argv) >= 3:
        try:
            top_k = int(sys.argv[2])
        except ValueError as exc:
            raise SystemExit(
                "top_k must be an integer"
            ) from exc

    await search(
        query=query,
        top_k=top_k,
    )


if __name__ == "__main__":
    asyncio.run(main())