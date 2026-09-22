from __future__ import annotations

import uuid

import pytest
from sqlalchemy import delete

from backend.app.db.models.element_relationship import (
    ElementRelationship,
)
from backend.app.db.models.source_document import (
    SourceDocument,
)
from backend.app.db.models.source_element import (
    SourceElement,
)
from backend.app.db.session import AsyncSessionLocal
from backend.app.retrieval.relationship_store import (
    SqlAlchemyRelationshipStore,
)


@pytest.mark.asyncio
async def test_get_relationships_returns_same_document_relationships():
    document_id = f"relationship-store-{uuid.uuid4().hex}"

    semantic_id = f"{document_id}:image-1:semantic"
    image_id = f"{document_id}:image-1"
    paragraph_id = f"{document_id}:paragraph-1"

    async with AsyncSessionLocal() as session:
        document = SourceDocument(
            document_id=document_id,
            source_type="image",
            title="Relationship Store Test",
            metadata_json={},
        )

        semantic = SourceElement(
            element_id=semantic_id,
            document_id=document_id,
            element_type="image_semantic",
            content_type="text",
            text_content="Image semantic description.",
            asset_path=None,
            metadata_json={},
        )

        image = SourceElement(
            element_id=image_id,
            document_id=document_id,
            element_type="image",
            content_type="image",
            text_content=None,
            asset_path="/tmp/image.webp",
            metadata_json={},
        )

        paragraph = SourceElement(
            element_id=paragraph_id,
            document_id=document_id,
            element_type="paragraph",
            content_type="text",
            text_content="Figure 1 explains the image.",
            asset_path=None,
            metadata_json={},
        )

        relationship_1 = ElementRelationship(
            source_element_id=semantic_id,
            relationship_type="derived_from",
            target_element_id=image_id,
            metadata_json={
                "derivation_type": (
                    "visual_semantic_representation"
                ),
            },
        )

        relationship_2 = ElementRelationship(
            source_element_id=paragraph_id,
            relationship_type="refers_to",
            target_element_id=image_id,
            metadata_json={},
        )

        session.add(document)
        await session.flush()

        session.add_all(
            [
                semantic,
                image,
                paragraph,
            ]
        )
        await session.flush()

        session.add_all(
            [
                relationship_1,
                relationship_2,
            ]
        )
        await session.commit()

    try:
        store = SqlAlchemyRelationshipStore(
            session_factory=AsyncSessionLocal,
        )

        relationships = await store.get_relationships(
            element_ids={semantic_id},
            document_id=document_id,
        )

        relationship_types = {
            relationship.relationship_type
            for relationship in relationships
        }

        assert "derived_from" in relationship_types

    finally:
        async with AsyncSessionLocal() as session:
            await session.execute(
                delete(ElementRelationship).where(
                    ElementRelationship.source_element_id.in_(
                        {
                            semantic_id,
                            paragraph_id,
                        }
                    )
                )
            )

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


@pytest.mark.asyncio
async def test_get_elements_is_document_scoped():
    document_id = f"relationship-store-{uuid.uuid4().hex}"
    other_document_id = f"other-{uuid.uuid4().hex}"

    element_id = f"{document_id}:image-1"
    other_element_id = f"{other_document_id}:image-1"

    async with AsyncSessionLocal() as session:
        document = SourceDocument(
            document_id=document_id,
            source_type="image",
            title="Test",
            metadata_json={},
        )

        other_document = SourceDocument(
            document_id=other_document_id,
            source_type="image",
            title="Other",
            metadata_json={},
        )

        session.add_all(
            [
                document,
                other_document,
            ]
        )
        await session.flush()

        session.add_all(
            [
                SourceElement(
                    element_id=element_id,
                    document_id=document_id,
                    element_type="image",
                    content_type="image",
                    text_content=None,
                    asset_path="/tmp/image.webp",
                    metadata_json={
                        "name": "correct",
                    },
                ),
                SourceElement(
                    element_id=other_element_id,
                    document_id=other_document_id,
                    element_type="image",
                    content_type="image",
                    text_content=None,
                    asset_path="/tmp/other.webp",
                    metadata_json={
                        "name": "wrong-document",
                    },
                ),
            ]
        )

        await session.commit()

    try:
        store = SqlAlchemyRelationshipStore(
            session_factory=AsyncSessionLocal,
        )

        elements = await store.get_elements(
            element_ids={
                element_id,
                other_element_id,
            },
            document_id=document_id,
        )

        assert len(elements) == 1
        assert elements[0].element_id == element_id

    finally:
        async with AsyncSessionLocal() as session:
            await session.execute(
                delete(SourceElement).where(
                    SourceElement.document_id.in_(
                        {
                            document_id,
                            other_document_id,
                        }
                    )
                )
            )

            await session.execute(
                delete(SourceDocument).where(
                    SourceDocument.document_id.in_(
                        {
                            document_id,
                            other_document_id,
                        }
                    )
                )
            )

            await session.commit()


@pytest.mark.asyncio
async def test_get_relationships_does_not_cross_document_boundary():
    document_id = f"relationship-store-{uuid.uuid4().hex}"
    other_document_id = f"other-{uuid.uuid4().hex}"

    source_id = f"{document_id}:image-1"
    foreign_target_id = f"{other_document_id}:image-1"

    async with AsyncSessionLocal() as session:
        document = SourceDocument(
            document_id=document_id,
            source_type="image",
            title="Test",
            metadata_json={},
        )

        other_document = SourceDocument(
            document_id=other_document_id,
            source_type="image",
            title="Other",
            metadata_json={},
        )

        session.add_all(
            [
                document,
                other_document,
            ]
        )
        await session.flush()

        # Deliberately create a cross-document relationship at DB level.
        # The store must never expose it during scoped retrieval.
        source = SourceElement(
            element_id=source_id,
            document_id=document_id,
            element_type="image",
            content_type="image",
            text_content=None,
            asset_path="/tmp/image.webp",
            metadata_json={},
        )

        foreign_target = SourceElement(
            element_id=foreign_target_id,
            document_id=other_document_id,
            element_type="image",
            content_type="image",
            text_content=None,
            asset_path="/tmp/other.webp",
            metadata_json={},
        )

        session.add_all(
            [
                source,
                foreign_target,
            ]
        )
        await session.flush()

        relationship = ElementRelationship(
            source_element_id=source_id,
            relationship_type="derived_from",
            target_element_id=foreign_target_id,
            metadata_json={},
        )

        session.add(relationship)
        await session.commit()

    try:
        store = SqlAlchemyRelationshipStore(
            session_factory=AsyncSessionLocal,
        )

        relationships = await store.get_relationships(
            element_ids={source_id},
            document_id=document_id,
        )

        assert relationships == []

    finally:
        async with AsyncSessionLocal() as session:
            await session.execute(
                delete(ElementRelationship).where(
                    ElementRelationship.source_element_id
                    == source_id
                )
            )

            await session.execute(
                delete(SourceElement).where(
                    SourceElement.document_id.in_(
                        {
                            document_id,
                            other_document_id,
                        }
                    )
                )
            )

            await session.execute(
                delete(SourceDocument).where(
                    SourceDocument.document_id.in_(
                        {
                            document_id,
                            other_document_id,
                        }
                    )
                )
            )

            await session.commit()