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


@pytest.mark.asyncio
async def test_source_document_and_elements_persist():
    document_id = f"test-doc-{uuid.uuid4().hex}"

    async with AsyncSessionLocal() as session:
        document = SourceDocument(
            document_id=document_id,
            source_type="pdf",
            title="Test Document",
            metadata_json={
                "filename": "test.pdf",
            },
        )

        element = SourceElement(
            element_id=f"{document_id}-element-001",
            document_id=document_id,
            element_type="paragraph",
            content_type="text",
            text_content="This is a test paragraph.",
            metadata_json={
                "page_number": 1,
            },
        )

        session.add(document)
        session.add(element)

        await session.commit()

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(SourceDocument).where(
                SourceDocument.document_id == document_id
            )
        )

        stored_document = result.scalar_one()

        assert stored_document.document_id == document_id
        assert stored_document.source_type == "pdf"

        result = await session.execute(
            select(SourceElement).where(
                SourceElement.document_id == document_id
            )
        )

        stored_element = result.scalar_one()

        assert stored_element.element_id == (
            f"{document_id}-element-001"
        )
        assert stored_element.text_content == (
            "This is a test paragraph."
        )

        await session.execute(
            delete(SourceDocument).where(
                SourceDocument.document_id == document_id
            )
        )

        await session.commit()

@pytest.mark.asyncio
async def test_element_relationship_persists_and_cascades():
    document_id = f"relationship-doc-{uuid.uuid4().hex}"

    element_1_id = f"{document_id}-element-001"
    element_2_id = f"{document_id}-element-002"

    async with AsyncSessionLocal() as session:
        document = SourceDocument(
            document_id=document_id,
            source_type="pdf",
            title="Relationship Test",
            metadata_json={},
        )

        element_1 = SourceElement(
            element_id=element_1_id,
            document_id=document_id,
            element_type="paragraph",
            content_type="text",
            text_content="Figure 1 shows the architecture.",
            metadata_json={
                "page_number": 1,
            },
        )

        element_2 = SourceElement(
            element_id=element_2_id,
            document_id=document_id,
            element_type="image",
            content_type="image",
            asset_path="/tmp/figure-1.png",
            metadata_json={
                "page_number": 1,
                "figure_number": 1,
            },
        )

        session.add(document)
        session.add(element_1)
        session.add(element_2)

        # Ensure document and source elements exist
        # before inserting the foreign-key relationship.
        await session.flush()

        relationship = ElementRelationship(
            source_element_id=element_1_id,
            relationship_type="refers_to",
            target_element_id=element_2_id,
            metadata_json={
                "confidence": 1.0,
                "method": "explicit_reference",
            },
        )

        session.add(relationship)

        await session.commit()

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ElementRelationship).where(
                ElementRelationship.source_element_id
                == element_1_id
            )
        )

        stored_relationship = result.scalar_one()

        assert stored_relationship.relationship_type == "refers_to"
        assert stored_relationship.source_element_id == element_1_id
        assert stored_relationship.target_element_id == element_2_id
        assert stored_relationship.metadata_json["confidence"] == 1.0
        assert (
            stored_relationship.metadata_json["method"]
            == "explicit_reference"
        )

        await session.execute(
            delete(SourceDocument).where(
                SourceDocument.document_id == document_id
            )
        )

        await session.commit()

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ElementRelationship)
            .where(
                ElementRelationship.source_element_id
                == element_1_id
            )
        )

        assert result.scalar_one_or_none() is None