from __future__ import annotations

from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.element_relationship import (
    ElementRelationship,
)
from backend.app.db.models.source_element import (
    SourceElement,
)
from backend.app.db.session import AsyncSessionLocal


class SqlAlchemyRelationshipStore:
    """
    PostgreSQL-backed relationship store.

    Provides document-scoped access to source elements
    and their relationships.
    """

    def __init__(
        self,
        session_factory: Any = AsyncSessionLocal,
    ) -> None:
        self.session_factory = session_factory

    async def get_relationships(
        self,
        element_ids: set[str],
        document_id: str,
    ) -> list[ElementRelationship]:
        if not element_ids:
            return []

        if not document_id.strip():
            raise ValueError(
                "document_id cannot be empty"
            )

        source_element = SourceElement.__table__.alias(
            "relationship_source"
        )

        target_element = SourceElement.__table__.alias(
            "relationship_target"
        )

        statement = (
            select(ElementRelationship)
            .join(
                source_element,
                source_element.c.element_id
                == ElementRelationship.source_element_id,
            )
            .join(
                target_element,
                target_element.c.element_id
                == ElementRelationship.target_element_id,
            )
            .where(
                source_element.c.document_id
                == document_id,
                target_element.c.document_id
                == document_id,
                or_(
                    ElementRelationship.source_element_id.in_(
                        element_ids
                    ),
                    ElementRelationship.target_element_id.in_(
                        element_ids
                    ),
                ),
            )
        )

        async with self.session_factory() as session:
            result = await session.execute(
                statement
            )

            return list(
                result.scalars().all()
            )

    async def get_elements(
        self,
        element_ids: set[str],
        document_id: str,
    ) -> list[SourceElement]:
        if not element_ids:
            return []

        if not document_id.strip():
            raise ValueError(
                "document_id cannot be empty"
            )

        statement = select(
            SourceElement
        ).where(
            SourceElement.document_id
            == document_id,
            SourceElement.element_id.in_(
                element_ids
            ),
        )

        async with self.session_factory() as session:
            result = await session.execute(
                statement
            )

            return list(
                result.scalars().all()
            )