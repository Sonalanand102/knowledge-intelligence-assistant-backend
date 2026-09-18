from __future__ import annotations

import uuid

import pytest
from sqlalchemy import delete, select

from backend.app.db.models import (
    IngestionRun,
    SourceDocument,
)
from backend.app.db.session import AsyncSessionLocal
from backend.app.ingestion.service import IngestionService


@pytest.mark.asyncio
async def test_ingestion_run_lifecycle_completed():
    document_id = (
        f"lifecycle-doc-{uuid.uuid4().hex}"
    )

    async with AsyncSessionLocal() as session:
        service = IngestionService(
            session=session,
        )

        run_id = await service.start_run(
            stats={
                "source_type": "pdf",
            }
        )

        run = await session.scalar(
            select(IngestionRun).where(
                IngestionRun.id == run_id
            )
        )

        assert run is not None
        assert run.status == "running"
        assert run.document_id is None

        document = SourceDocument(
            document_id=document_id,
            source_type="pdf",
            title="Lifecycle Test",
            metadata_json={
                "filename": "lifecycle.pdf",
            },
        )

        session.add(document)
        await session.commit()

        await service.attach_document(
            run_id,
            document_id,
        )

        await service.mark_completed(
            run_id,
            stats={
                "source_type": "pdf",
                "elements_persisted": 3,
                "relationships_persisted": 1,
                "chunks_indexed": 5,
            },
        )

    async with AsyncSessionLocal() as session:
        run = await session.scalar(
            select(IngestionRun).where(
                IngestionRun.id == run_id
            )
        )

        assert run is not None
        assert run.status == "completed"
        assert run.document_id == document_id
        assert run.completed_at is not None

        assert run.stats == {
            "source_type": "pdf",
            "elements_persisted": 3,
            "relationships_persisted": 1,
            "chunks_indexed": 5,
        }

        await session.execute(
            delete(SourceDocument).where(
                SourceDocument.document_id == document_id
            )
        )

        await session.execute(
            delete(IngestionRun).where(
                IngestionRun.id == run_id
            )
        )

        await session.commit()


@pytest.mark.asyncio
async def test_ingestion_run_lifecycle_failed():
    async with AsyncSessionLocal() as session:
        service = IngestionService(
            session=session,
        )

        run_id = await service.start_run(
            stats={
                "source_type": "pdf",
            }
        )

        await service.mark_failed(
            run_id,
            error_message="Vector indexing failed",
            stats={
                "source_type": "pdf",
                "elements_persisted": 3,
                "chunks_indexed": 5,
            },
        )

    async with AsyncSessionLocal() as session:
        run = await session.scalar(
            select(IngestionRun).where(
                IngestionRun.id == run_id
            )
        )

        assert run is not None
        assert run.status == "failed"
        assert run.document_id is None
        assert run.completed_at is not None
        assert run.error_message == (
            "Vector indexing failed"
        )

        assert run.stats == {
            "source_type": "pdf",
            "elements_persisted": 3,
            "chunks_indexed": 5,
        }

        await session.execute(
            delete(IngestionRun).where(
                IngestionRun.id == run_id
            )
        )

        await session.commit()


@pytest.mark.asyncio
async def test_ingestion_run_can_start_without_document():
    async with AsyncSessionLocal() as session:
        service = IngestionService(
            session=session,
        )

        run_id = await service.start_run()

        run = await session.scalar(
            select(IngestionRun).where(
                IngestionRun.id == run_id
            )
        )

        assert run is not None
        assert run.status == "running"
        assert run.document_id is None

        await session.execute(
            delete(IngestionRun).where(
                IngestionRun.id == run_id
            )
        )

        await session.commit()