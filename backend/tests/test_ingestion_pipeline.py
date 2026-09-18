from __future__ import annotations

import uuid

import pytest

from backend.app.ingestion.models.chunk_document import (
    ChunkDocument,
)
from backend.app.ingestion.models.content import (
    TextContent,
)
from backend.app.ingestion.models.ingestion_result import (
    IngestionResult,
)
from backend.app.ingestion.models.source_element import (
    SourceElement,
)
from backend.app.ingestion.pipeline import (
    IngestionPipeline,
)
from backend.app.ingestion.service import (
    IngestionPersistenceResult,
)


class FakeIngestionService:
    def __init__(self) -> None:
        self.run_id = uuid.uuid4()
        self.persisted_result = None
        self.source_type = None
        self.attached_document_id = None
        self.completed_run_id = None
        self.failed_run_id = None
        self.failure_message = None
        self.completed_stats = None

    async def start_run(
        self,
        *,
        stats: dict | None = None,
    ):
        self.started_stats = stats
        return self.run_id

    async def persist(
        self,
        result,
        *,
        source_type,
        title=None,
        document_metadata=None,
    ):
        self.persisted_result = result
        self.source_type = source_type
        self.title = title
        self.document_metadata = document_metadata

        return IngestionPersistenceResult(
            document_id=result.document_id,
            elements_persisted=len(result.elements),
            relationships_persisted=len(
                result.relationships
            ),
        )

    async def attach_document(
        self,
        run_id,
        document_id,
    ):
        self.attached_run_id = run_id
        self.attached_document_id = document_id

    async def mark_completed(
        self,
        run_id,
        *,
        stats=None,
    ):
        self.completed_run_id = run_id
        self.completed_stats = stats

    async def mark_failed(
        self,
        run_id,
        *,
        error_message,
        stats=None,
    ):
        self.failed_run_id = run_id
        self.failure_message = error_message
        self.failed_stats = stats


class FailingIngestionService(FakeIngestionService):
    async def persist(
        self,
        result,
        *,
        source_type,
        title=None,
        document_metadata=None,
    ):
        raise RuntimeError(
            "Database persistence failed"
        )


class FakeVectorIndexer:
    def __init__(self) -> None:
        self.indexed_chunks = None

    async def index(self, chunks):
        self.indexed_chunks = chunks


class FailingVectorIndexer(FakeVectorIndexer):
    async def index(self, chunks):
        raise RuntimeError(
            "Vector indexing failed"
        )


def make_ingestion_result() -> IngestionResult:
    return IngestionResult(
        document_id="pipeline-doc",
        elements=[
            SourceElement(
                element_id="pipeline-element-001",
                document_id="pipeline-doc",
                element_type="paragraph",
                content=TextContent(
                    text="RAG retrieves external context."
                ),
                metadata={
                    "page_number": 1,
                },
            )
        ],
        relationships=[],
    )


def make_chunks(
    result: IngestionResult,
) -> list[ChunkDocument]:
    return [
        ChunkDocument(
            content=(
                "RAG retrieves external context."
            ),
            document_id=result.document_id,
            chunk_index=0,
        )
    ]


@pytest.mark.asyncio
async def test_ingestion_pipeline_orchestrates_full_flow():
    ingestion_service = FakeIngestionService()
    vector_indexer = FakeVectorIndexer()

    pipeline = IngestionPipeline(
        ingestion_service=ingestion_service,
        vector_indexer=vector_indexer,
        preprocessor=lambda result: result,
        chunker=make_chunks,
    )

    loader_called = False

    def loader():
        nonlocal loader_called

        loader_called = True

        return make_ingestion_result()

    result = await pipeline.ingest(
        loader,
        source_type="pdf",
        title="Test PDF",
        document_metadata={
            "filename": "test.pdf",
        },
    )

    assert loader_called is True

    assert result.run_id == ingestion_service.run_id
    assert result.document_id == "pipeline-doc"
    assert result.elements_persisted == 1
    assert result.relationships_persisted == 0
    assert result.chunks_indexed == 1

    assert (
        ingestion_service.persisted_result
        is not None
    )

    assert (
        ingestion_service.source_type
        == "pdf"
    )

    assert (
        ingestion_service.attached_run_id
        == ingestion_service.run_id
    )

    assert (
        ingestion_service.attached_document_id
        == "pipeline-doc"
    )

    assert (
        ingestion_service.completed_run_id
        == ingestion_service.run_id
    )

    assert (
        ingestion_service.completed_stats[
            "source_type"
        ]
        == "pdf"
    )

    assert (
        ingestion_service.completed_stats[
            "elements_persisted"
        ]
        == 1
    )

    assert (
        ingestion_service.completed_stats[
            "relationships_persisted"
        ]
        == 0
    )

    assert (
        ingestion_service.completed_stats[
            "chunks_indexed"
        ]
        == 1
    )

    assert (
        ingestion_service.failed_run_id
        is None
    )

    assert (
        vector_indexer.indexed_chunks
        is not None
    )

    assert len(
        vector_indexer.indexed_chunks
    ) == 1


@pytest.mark.asyncio
async def test_ingestion_pipeline_marks_failure_when_persistence_fails():
    ingestion_service = FailingIngestionService()
    vector_indexer = FakeVectorIndexer()

    pipeline = IngestionPipeline(
        ingestion_service=ingestion_service,
        vector_indexer=vector_indexer,
        preprocessor=lambda result: result,
        chunker=make_chunks,
    )

    with pytest.raises(
        RuntimeError,
        match="Database persistence failed",
    ):
        await pipeline.ingest(
            lambda: make_ingestion_result(),
            source_type="pdf",
        )

    assert (
        ingestion_service.failed_run_id
        == ingestion_service.run_id
    )

    assert (
        ingestion_service.failure_message
        == "Database persistence failed"
    )

    assert (
        vector_indexer.indexed_chunks
        is None
    )


@pytest.mark.asyncio
async def test_ingestion_pipeline_rejects_invalid_loader_result():
    ingestion_service = FakeIngestionService()
    vector_indexer = FakeVectorIndexer()

    pipeline = IngestionPipeline(
        ingestion_service=ingestion_service,
        vector_indexer=vector_indexer,
        preprocessor=lambda result: result,
        chunker=make_chunks,
    )

    with pytest.raises(
        TypeError,
        match="Loader must return an IngestionResult",
    ):
        await pipeline.ingest(
            lambda: "not-an-ingestion-result",
            source_type="pdf",
        )

    assert (
        ingestion_service.failed_run_id
        == ingestion_service.run_id
    )

    assert (
        "Loader must return an IngestionResult"
        in ingestion_service.failure_message
    )


@pytest.mark.asyncio
async def test_ingestion_pipeline_rejects_invalid_preprocessor_result():
    ingestion_service = FakeIngestionService()
    vector_indexer = FakeVectorIndexer()

    pipeline = IngestionPipeline(
        ingestion_service=ingestion_service,
        vector_indexer=vector_indexer,
        preprocessor=lambda result: "invalid",
        chunker=make_chunks,
    )

    with pytest.raises(
        TypeError,
        match="Preprocessor must return an IngestionResult",
    ):
        await pipeline.ingest(
            lambda: make_ingestion_result(),
            source_type="pdf",
        )

    assert (
        ingestion_service.failed_run_id
        == ingestion_service.run_id
    )

    assert (
        "Preprocessor must return an IngestionResult"
        in ingestion_service.failure_message
    )


@pytest.mark.asyncio
async def test_ingestion_pipeline_rejects_invalid_chunk():
    ingestion_service = FakeIngestionService()
    vector_indexer = FakeVectorIndexer()

    pipeline = IngestionPipeline(
        ingestion_service=ingestion_service,
        vector_indexer=vector_indexer,
        preprocessor=lambda result: result,
        chunker=lambda result: ["invalid"],
    )

    with pytest.raises(
        TypeError,
        match="Chunker returned an invalid chunk",
    ):
        await pipeline.ingest(
            lambda: make_ingestion_result(),
            source_type="pdf",
        )

    assert (
        ingestion_service.failed_run_id
        == ingestion_service.run_id
    )

    assert (
        "Chunker returned an invalid chunk"
        in ingestion_service.failure_message
    )


@pytest.mark.asyncio
async def test_ingestion_pipeline_marks_failure_when_vector_indexing_fails():
    ingestion_service = FakeIngestionService()
    vector_indexer = FailingVectorIndexer()

    pipeline = IngestionPipeline(
        ingestion_service=ingestion_service,
        vector_indexer=vector_indexer,
        preprocessor=lambda result: result,
        chunker=make_chunks,
    )

    with pytest.raises(
        RuntimeError,
        match="Vector indexing failed",
    ):
        await pipeline.ingest(
            lambda: make_ingestion_result(),
            source_type="pdf",
        )

    # PostgreSQL persistence happened before Qdrant indexing.
    assert (
        ingestion_service.persisted_result
        is not None
    )

    assert (
        ingestion_service.attached_document_id
        == "pipeline-doc"
    )

    assert (
        ingestion_service.failed_run_id
        == ingestion_service.run_id
    )

    assert (
        ingestion_service.failure_message
        == "Vector indexing failed"
    )

    assert (
        ingestion_service.completed_run_id
        is None
    )