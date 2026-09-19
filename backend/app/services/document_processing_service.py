from __future__ import annotations

from pathlib import Path

from google import genai
from sqlalchemy import select

from backend.app.core.config import settings
from backend.app.db.models.source_document import SourceDocument
from backend.app.db.session import AsyncSessionLocal
from backend.app.embeddings.gemini import (
    GeminiEmbeddingProvider,
)
from backend.app.embeddings.service import EmbeddingService
from backend.app.ingestion.chunking.text_chunker import (
    chunk_documents,
)
from backend.app.ingestion.loader_registry import (
    LoaderRegistry,
)
from backend.app.ingestion.pipeline import (
    IngestionPipeline,
)
from backend.app.ingestion.preprocessing.pipeline import (
    preprocess_ingestion,
)
from backend.app.ingestion.service import (
    IngestionService,
)
from backend.app.retrieval.qdrant import (
    create_qdrant_client,
)
from backend.app.retrieval.qdrant_store import (
    QdrantVectorStore,
)
from backend.app.retrieval.vector_indexer import (
    VectorIndexer,
)


class DocumentProcessingService:
    async def process_document(
        self,
        document_id: str,
    ):
        # --------------------------------------------------
        # Read uploaded document metadata
        # --------------------------------------------------

        async with AsyncSessionLocal() as session:
            document = await session.scalar(
                select(SourceDocument).where(
                    SourceDocument.document_id
                    == document_id
                )
            )

            if document is None:
                raise ValueError(
                    f"Document not found: {document_id}"
                )

            metadata = dict(
                document.metadata_json or {}
            )

            source_type = document.source_type
            title = document.title

            storage_path = metadata.get(
                "storage_path"
            )

            if not storage_path:
                raise ValueError(
                    "Document does not have a storage_path: "
                    f"{document_id}"
                )

        file_path = Path(storage_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Uploaded file not found: {file_path}"
            )

        output_dir = file_path.parent / "processed"
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------
        # Mark processing
        # --------------------------------------------------

        await self._update_status(
            document_id,
            "processing",
        )

        gemini_client = genai.Client(
            api_key=settings.gemini_api_key,
        )

        qdrant_client = create_qdrant_client()

        try:
            embedding_provider = (
                GeminiEmbeddingProvider(
                    client=gemini_client,
                )
            )

            embedding_service = (
                EmbeddingService(
                    provider=embedding_provider,
                )
            )

            vector_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=(
                    settings.qdrant_collection_name
                ),
            )

            vector_indexer = VectorIndexer(
                embedding_service=embedding_service,
                vector_store=vector_store,
            )

            loader_registry = LoaderRegistry()

            loader = loader_registry.create_loader(
                source_type=source_type,
                file_path=str(file_path),
                document_id=document_id,
                output_dir=str(output_dir),
            )

            async with AsyncSessionLocal() as session:
                ingestion_service = IngestionService(
                    session=session,
                )

                pipeline = IngestionPipeline(
                    ingestion_service=(
                        ingestion_service
                    ),
                    vector_indexer=vector_indexer,
                    preprocessor=(
                        preprocess_ingestion
                    ),
                    chunker=chunk_documents,
                )

                result = await pipeline.ingest(
                    loader,
                    source_type=source_type,
                    title=title,
                    document_metadata={
                        **metadata,
                        "processing_status": (
                            "processing"
                        ),
                    },
                    allow_existing_document=True,
                )

            await self._update_status(
                document_id,
                "completed",
            )

            return result

        except Exception as exc:
            await self._update_status(
                document_id,
                "failed",
                error_message=str(exc),
            )

            raise

        finally:
            gemini_client.close()
            await qdrant_client.close()

    async def _update_status(
        self,
        document_id: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        async with AsyncSessionLocal() as session:
            ingestion_service = IngestionService(
                session=session,
            )

            await ingestion_service.update_document_status(
                document_id,
                status=status,
                error_message=error_message,
            )