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
from backend.app.retrieval import hybrid_retriever
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

from backend.app.retrieval.relationship_aware_retriever import (
    RelationshipAwareRetriever,
)
from backend.app.retrieval.relationship_context_expander import (
    RelationshipContextExpander,
)
from backend.app.retrieval.relationship_store import (
    SqlAlchemyRelationshipStore,
)

from backend.app.generation.answer_service import (
    AnswerService,
)
from backend.app.generation.gemini_answer_generator import (
    GeminiAnswerGenerator,
)

from backend.app.api.v1.ask import (
    router as ask_router,
)

from pathlib import Path

from backend.app.ingestion.loader_registry import (
    LoaderRegistry,
)
from backend.app.services.document_upload_service import (
    DocumentUploadService,
)
from backend.app.services.file_storage import (
    LocalFileStorage,
)

from backend.app.api.v1.documents import (
    document_router,
    router as documents_router,
)

from backend.app.api.v1.chats import router as chats_router

from backend.app.queues.document_queue import (
    DocumentJobQueue,
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

        # --------------------------------------------------
        # Relationship-aware retrieval
        # --------------------------------------------------

        relationship_store = SqlAlchemyRelationshipStore()

        context_expander = RelationshipContextExpander(
            store=relationship_store,
        )

        relationship_aware_retriever = RelationshipAwareRetriever(
            retriever=hybrid_retriever,
            context_expander=context_expander,
        )

        # --------------------------------------------------
        # Application services
        # --------------------------------------------------

        search_service = SearchService(
            retriever=relationship_aware_retriever,
        )

        answer_generator = GeminiAnswerGenerator(
            client=gemini_client,
        )

        answer_service = AnswerService(
            retriever=relationship_aware_retriever,
            answer_generator=answer_generator,
        )

        # --------------------------------------------------
        # Document upload infrastructure
        # --------------------------------------------------

        project_root = Path(
            __file__
        ).resolve().parents[2]

        upload_storage = LocalFileStorage(
            root_dir=(
                project_root
                / "storage"
                / "uploads"
            ),
            max_file_size_bytes=50 * 1024 * 1024,
        )

        loader_registry = LoaderRegistry()

        document_job_queue = DocumentJobQueue(
            redis_url=settings.redis_url,
        )

        document_upload_service = DocumentUploadService(
            storage=upload_storage,
            loader_registry=loader_registry,
            job_queue=document_job_queue,
        )

        app.state.document_upload_service = (
            document_upload_service
        )

        app.state.document_job_queue = (
            document_job_queue
        )

        app.state.redis = redis_client
        app.state.qdrant = qdrant_client
        app.state.gemini = gemini_client
        app.state.embedding_service = embedding_service
        app.state.vector_store = vector_store
        app.state.search_service = search_service
        app.state.answer_service = answer_service

        yield

    finally:
        document_job_queue.close()
        gemini_client.close()
        await redis_client.aclose()
        await qdrant_client.close()
        await engine.dispose()


app = FastAPI(
    title="Knowledge Intelligence Assistant",
    version="0.1.0",
    lifespan=lifespan,
)

from fastapi.openapi.utils import get_openapi


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        openapi_version="3.0.3",
    )

    # Swagger UI compatibility:
    # Convert OpenAPI 3.1 binary schemas generated by FastAPI
    # into the OpenAPI 3.0 style expected by many Swagger UIs.
    for component in schema.get("components", {}).get("schemas", {}).values():
        properties = component.get("properties", {})

        for property_schema in properties.values():
            if property_schema.get("type") != "array":
                continue

            items = property_schema.get("items", {})

            if (
                items.get("type") == "string"
                and items.get("contentMediaType")
            ):
                items.pop("contentMediaType", None)
                items["format"] = "binary"

    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok"}


app.include_router(
    search_router,
    prefix="/api/v1",
)

app.include_router(
    ask_router,
    prefix="/api/v1",
)

app.include_router(
    documents_router,
    prefix="/api/v1",
)

app.include_router(
    document_router,
    prefix="/api/v1",
)

app.include_router(
    chats_router,
    prefix="/api/v1",
)