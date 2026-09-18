from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from google import genai

from backend.app.api.v1.search import (
    router as search_router,
)
from backend.app.cache.redis import create_redis_client
from backend.app.core.config import settings
from backend.app.db.session import engine
from backend.app.embeddings.gemini import (
    GeminiEmbeddingProvider,
)
from backend.app.embeddings.service import (
    EmbeddingService,
)
from backend.app.retrieval.dense_retriever import (
    DenseRetriever,
)
from backend.app.retrieval.sparse_retriever import (
    SparseRetriever,
)
from backend.app.retrieval.hybrid_retriever import (
    HybridRetriever,
)
from backend.app.retrieval.qdrant import (
    create_qdrant_client,
)
from backend.app.retrieval.qdrant_store import (
    QdrantVectorStore,
)
from backend.app.retrieval.search_service import (
    SearchService,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_client = create_redis_client()
    qdrant_client = create_qdrant_client()

    gemini_client = genai.Client(
        api_key=settings.gemini_api_key,
    )

    try:
        # Verify infrastructure connectivity.
        await redis_client.ping()
        await qdrant_client.get_collections()

        # Build application services.
        embedding_provider = GeminiEmbeddingProvider(
            client=gemini_client,
        )

        embedding_service = EmbeddingService(
            provider=embedding_provider,
        )

        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=settings.qdrant_collection_name,
        )

        dense_retriever = DenseRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
        )

        sparse_retriever = SparseRetriever(
            vector_store=vector_store,
        )

        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        )

        search_service = SearchService(
            retriever=hybrid_retriever,
        )

        app.state.redis = redis_client
        app.state.qdrant = qdrant_client
        app.state.gemini = gemini_client
        app.state.embedding_service = embedding_service
        app.state.vector_store = vector_store
        app.state.search_service = search_service

        yield

    finally:
        gemini_client.close()
        await redis_client.aclose()
        await qdrant_client.close()
        await engine.dispose()


app = FastAPI(
    title="Knowledge Intelligence Assistant",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok"}


app.include_router(
    search_router,
    prefix="/api/v1",
)