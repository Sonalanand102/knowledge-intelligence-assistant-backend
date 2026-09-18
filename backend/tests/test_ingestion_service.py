from __future__ import annotations

import uuid

import pytest
from sqlalchemy import delete, select

from backend.app.db.models import (
    ElementRelationship,
    SourceDocument,
    SourceElement,
)
from backend.app.db.session import AsyncSessionLocal
from backend.app.ingestion.models.content import (
    ImageContent,
    TextContent,
)
from backend.app.ingestion.models.element_relationship import (
    ElementRelationship as IngestionRelationship,
)
from backend.app.ingestion.models.ingestion_result import (
    IngestionResult,
)
from backend.app.ingestion.models.source_element import (
    SourceElement as IngestionSourceElement,
)
from backend.app.ingestion.service import (
    IngestionService,
)


@pytest.mark.asyncio
async def test_ingestion_service_persists_result():
    document_id = (
        f"ingestion-service-{uuid.uuid4().hex}"
    )

    paragraph_id = f"{document_id}-paragraph"
    image_id = f"{document_id}-image"

    ingestion_result = IngestionResult(
        document_id=document_id,
        elements=[
            IngestionSourceElement(
                element_id=paragraph_id,
                document_id=document_id,
                element_type="paragraph",
                content=TextContent(
                    text="Figure 1 shows the architecture."
                ),
                metadata={
                    "content_type": "text",
                    "page_number": 1,
                },
            ),
            IngestionSourceElement(
                element_id=image_id,
                document_id=document_id,
                element_type="image",
                content=ImageContent(
                    path="/tmp/figure-1.png"
                ),
                metadata={
                    "content_type": "image",
                    "page_number": 1,
                },
            ),
        ],
        relationships=[
            IngestionRelationship(
                source_element_id=paragraph_id,
                relationship_type="refers_to",
                target_element_id=image_id,
                metadata={
                    "confidence": 1.0,
                    "method": "explicit_reference",
                },
            )
        ],
    )

    async with AsyncSessionLocal() as session:
        service = IngestionService(
            session=session,
        )

        persisted = await service.persist(
            ingestion_result,
            source_type="pdf",
            title="Test Document",
            document_metadata={
                "filename": "test.pdf",
            },
        )

        assert persisted.document_id == document_id
        assert persisted.elements_persisted == 2
        assert persisted.relationships_persisted == 1

    async with AsyncSessionLocal() as session:
        document = await session.scalar(
            select(SourceDocument).where(
                SourceDocument.document_id
                == document_id
            )
        )

        assert document is not None
        assert document.source_type == "pdf"
        assert document.title == "Test Document"
        assert document.metadata_json["filename"] == (
            "test.pdf"
        )

        elements = (
            await session.scalars(
                select(SourceElement).where(
                    SourceElement.document_id
                    == document_id
                )
            )
        ).all()

        assert len(elements) == 2

        relationship = await session.scalar(
            select(ElementRelationship).where(
                ElementRelationship.source_element_id
                == paragraph_id
            )
        )

        assert relationship is not None
        assert (
            relationship.relationship_type
            == "refers_to"
        )

        await session.execute(
            delete(SourceDocument).where(
                SourceDocument.document_id
                == document_id
            )
        )

        await session.commit()


@pytest.mark.asyncio
async def test_ingestion_service_rejects_duplicate_document():
    document_id = (
        f"duplicate-test-{uuid.uuid4().hex}"
    )

    ingestion_result = IngestionResult(
        document_id=document_id,
    )

    async with AsyncSessionLocal() as session:
        service = IngestionService(
            session=session,
        )

        await service.persist(
            ingestion_result,
            source_type="pdf",
        )

    try:
        async with AsyncSessionLocal() as session:
            service = IngestionService(
                session=session,
            )

            with pytest.raises(ValueError):
                await service.persist(
                    ingestion_result,
                    source_type="pdf",
                )

    finally:
        async with AsyncSessionLocal() as session:
            await session.execute(
                delete(SourceDocument).where(
                    SourceDocument.document_id
                    == document_id
                )
            )

            await session.commit()