from __future__ import annotations

import uuid

import pytest
from sqlalchemy import delete, select

from backend.app.db.models import SourceDocument, SourceElement
from backend.app.db.session import AsyncSessionLocal
from backend.app.ingestion.models.content import TextContent
from backend.app.ingestion.models.ingestion_result import IngestionResult
from backend.app.ingestion.models.source_element import SourceElement as SourceElementModel
from backend.app.ingestion.service import IngestionService


@pytest.mark.asyncio
async def test_duplicate_document_ingestion_is_rejected():
    document_id = f"duplicate-test-{uuid.uuid4().hex}"

    ingestion_result = IngestionResult(
        document_id=document_id,
        elements=[
            SourceElementModel(
                element_id=f"{document_id}-element",
                document_id=document_id,
                element_type="paragraph",
                content=TextContent(
                    text="Test document content"
                ),
                metadata={},
            )
        ],
    )

    async with AsyncSessionLocal() as session:
        ingestion_service = IngestionService(
            session=session,
        )

        try:
            # -------------------------
            # First ingestion
            # -------------------------

            result = await ingestion_service.persist(
                ingestion_result,
                source_type="txt",
                title="Duplicate Test",
            )

            assert result.document_id == document_id
            assert result.elements_persisted == 1

            # -------------------------
            # Duplicate ingestion
            # -------------------------

            with pytest.raises(
                ValueError,
                match=f"Document already exists: {document_id}",
            ):
                await ingestion_service.persist(
                    ingestion_result,
                    source_type="txt",
                    title="Duplicate Test",
                )

            # -------------------------
            # Verify no duplicate data
            # -------------------------

            documents = (
                await session.scalars(
                    select(SourceDocument).where(
                        SourceDocument.document_id
                        == document_id
                    )
                )
            ).all()

            elements = (
                await session.scalars(
                    select(SourceElement).where(
                        SourceElement.document_id
                        == document_id
                    )
                )
            ).all()

            assert len(documents) == 1
            assert len(elements) == 1

        finally:
            await session.execute(
                delete(SourceElement).where(
                    SourceElement.document_id
                    == document_id
                )
            )

            await session.execute(
                delete(SourceDocument).where(
                    SourceDocument.document_id
                    == document_id
                )
            )

            await session.commit()