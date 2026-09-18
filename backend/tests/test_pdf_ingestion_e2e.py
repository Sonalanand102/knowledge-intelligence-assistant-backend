from __future__ import annotations

import uuid

import fitz
import pytest
from google import genai
from sqlalchemy import delete, select

from backend.app.core.config import settings
from backend.app.db.models import (
    IngestionRun,
    SourceDocument,
    SourceElement,
)
from backend.app.db.session import AsyncSessionLocal
from backend.app.embeddings.gemini import GeminiEmbeddingProvider
from backend.app.embeddings.service import EmbeddingService
from backend.app.ingestion.chunking.text_chunker import chunk_documents
from backend.app.ingestion.loaders.pdf_loader import load_pdf
from backend.app.ingestion.pipeline import IngestionPipeline
from backend.app.ingestion.preprocessing.pipeline import preprocess_ingestion
from backend.app.ingestion.service import IngestionService
from backend.app.retrieval.dense_retriever import DenseRetriever
from backend.app.retrieval.qdrant import create_qdrant_client
from backend.app.retrieval.qdrant_store import QdrantVectorStore
from backend.app.retrieval.vector_indexer import VectorIndexer


def _create_test_pdf(path: str) -> None:
    doc = fitz.open()

    page = doc.new_page()

    page.insert_text(
        (72, 72),
        (
            "Retrieval Augmented Generation (RAG) combines "
            "retrieval with language models. "
            "A retriever finds relevant knowledge and the "
            "language model generates an answer using that context."
        ),
    )

    doc.save(path)
    doc.close()


@pytest.mark.asyncio
async def test_pdf_ingestion_end_to_end(tmp_path):
    document_id = (
        f"e2e-pdf-{uuid.uuid4().hex}"
    )

    pdf_path = tmp_path / "rag_test.pdf"

    _create_test_pdf(str(pdf_path))

    # -------------------------
    # Infrastructure
    # -------------------------

    async with AsyncSessionLocal() as session:
        ingestion_service = IngestionService(
            session=session,
        )

        qdrant_client = create_qdrant_client()

        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=settings.qdrant_collection_name,
        )

        gemini_client = genai.Client(
            api_key=settings.gemini_api_key,
        )

        embedding_provider = GeminiEmbeddingProvider(
            client=gemini_client,
        )

        embedding_service = EmbeddingService(
            provider=embedding_provider,
        )

        vector_indexer = VectorIndexer(
            embedding_service=embedding_service,
            vector_store=vector_store,
        )

        retriever = DenseRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
        )


        text_chunker = chunk_documents

        pipeline = IngestionPipeline(
            ingestion_service=ingestion_service,
            vector_indexer=vector_indexer,
            preprocessor=preprocess_ingestion,
            chunker=text_chunker,
        )

        # -------------------------
        # Real ingestion
        # -------------------------

        result = await pipeline.ingest(
            loader=lambda: load_pdf(
                file_path=str(pdf_path),
                document_id=document_id,
                output_dir=str(tmp_path / "pdf_assets"),
            ),
            source_type="pdf",
            title="RAG Test Document",
            document_metadata={
                "file_name": "rag_test.pdf",
            },
        )

        assert result.document_id

        assert result.elements_persisted > 0
        assert result.chunks_indexed > 0

        # -------------------------
        # Verify ingestion run
        # -------------------------

        run = await session.scalar(
            select(IngestionRun).where(
                IngestionRun.id == result.run_id
            )
        )

        assert run is not None
        assert run.status == "completed"
        assert run.document_id == result.document_id
        assert run.completed_at is not None

        assert run.stats is not None
        assert run.stats["elements_persisted"] > 0
        assert run.stats["chunks_indexed"] > 0

        # -------------------------
        # Verify PostgreSQL
        # -------------------------

        document = await session.scalar(
            select(SourceDocument).where(
                SourceDocument.document_id
                == result.document_id
            )
        )

        assert document is not None
        assert document.source_type == "pdf"
        assert document.title == "RAG Test Document"

        elements = (
            await session.scalars(
                select(SourceElement).where(
                    SourceElement.document_id
                    == result.document_id
                )
            )
        ).all()

        assert len(elements) > 0

        # -------------------------
        # Verify Qdrant retrieval
        # -------------------------

        search_results = await retriever.retrieve(
            "What is retrieval augmented generation?",
            top_k=5,
        )

        assert len(search_results) > 0

        assert any(
            item.document_id == result.document_id
            for item in search_results
        )

        # -------------------------
        # Cleanup Qdrant
        # -------------------------

        await vector_store.delete_by_document_id(
            result.document_id
        )

        # -------------------------
        # Cleanup PostgreSQL
        # -------------------------

        await session.execute(
            delete(SourceElement).where(
                SourceElement.document_id
                == result.document_id
            )
        )

        # -------------------------
        # Cleanup PostgreSQL
        # -------------------------

        await session.execute(
            delete(SourceElement).where(
                SourceElement.document_id
                == result.document_id
            )
        )

        await session.execute(
            delete(SourceDocument).where(
                SourceDocument.document_id
                == result.document_id
            )
        )

        await session.execute(
            delete(IngestionRun).where(
                IngestionRun.id == result.run_id
            )
        )

        await session.commit()

        await qdrant_client.close()