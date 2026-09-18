from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models import (
    ElementRelationship,
    IngestionRun,
    SourceDocument,
    SourceElement,
)
from backend.app.ingestion.models.content import (
    AudioContent,
    ImageContent,
    TableContent,
    TextContent,
    VideoContent,
)
from backend.app.ingestion.models.ingestion_result import (
    IngestionResult,
)


@dataclass(frozen=True)
class IngestionPersistenceResult:
    document_id: str
    elements_persisted: int
    relationships_persisted: int


class IngestionService:
    """
    Handles relational persistence and ingestion-run lifecycle.
    """

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self.session = session

    async def start_run(
        self,
        *,
        stats: dict | None = None,
    ) -> uuid.UUID:
        run = IngestionRun(
            status="running",
            stats=stats or {},
        )

        self.session.add(run)

        await self.session.commit()

        return run.id

    async def attach_document(
        self,
        run_id: uuid.UUID,
        document_id: str,
    ) -> None:
        run = await self._get_run(run_id)

        run.document_id = document_id

        await self.session.commit()

    async def mark_completed(
        self,
        run_id: uuid.UUID,
        *,
        stats: dict | None = None,
    ) -> None:
        run = await self._get_run(run_id)

        run.status = "completed"
        run.completed_at = datetime.now(
            timezone.utc
        )

        if stats is not None:
            run.stats = stats

        await self.session.commit()

    async def mark_failed(
        self,
        run_id: uuid.UUID,
        *,
        error_message: str,
        stats: dict | None = None,
    ) -> None:
        # An earlier failed DB operation may have left
        # the session in a rollback-required state.
        await self.session.rollback()

        run = await self._get_run(run_id)

        run.status = "failed"
        run.completed_at = datetime.now(
            timezone.utc
        )
        run.error_message = error_message

        if stats is not None:
            run.stats = stats

        await self.session.commit()

    async def persist(
        self,
        result: IngestionResult,
        *,
        source_type: str,
        title: str | None = None,
        document_metadata: dict | None = None,
    ) -> IngestionPersistenceResult:
        if not result.document_id.strip():
            raise ValueError(
                "IngestionResult must contain a document_id"
            )

        if not source_type.strip():
            raise ValueError(
                "source_type cannot be empty"
            )

        existing_document = await self.session.scalar(
            select(SourceDocument).where(
                SourceDocument.document_id
                == result.document_id
            )
        )

        if existing_document is not None:
            raise ValueError(
                "Document already exists: "
                f"{result.document_id}"
            )

        document = SourceDocument(
            document_id=result.document_id,
            source_type=source_type,
            title=title,
            metadata_json=document_metadata or {},
        )

        self.session.add(document)

        elements = [
            self._to_source_element(
                element,
            )
            for element in result.elements
        ]

        self.session.add_all(elements)

        await self.session.flush()

        relationships = [
            ElementRelationship(
                source_element_id=relationship.source_element_id,
                relationship_type=relationship.relationship_type,
                target_element_id=relationship.target_element_id,
                metadata_json=relationship.metadata,
            )
            for relationship in result.relationships
        ]

        self.session.add_all(relationships)

        await self.session.commit()

        return IngestionPersistenceResult(
            document_id=result.document_id,
            elements_persisted=len(elements),
            relationships_persisted=len(relationships),
        )

    async def _get_run(
        self,
        run_id: uuid.UUID,
    ) -> IngestionRun:
        run = await self.session.scalar(
            select(IngestionRun).where(
                IngestionRun.id == run_id
            )
        )

        if run is None:
            raise ValueError(
                f"Ingestion run not found: {run_id}"
            )

        return run

    @staticmethod
    def _to_source_element(
        element,
    ) -> SourceElement:
        content = element.content

        text_content: str | None = None
        asset_path: str | None = None

        if isinstance(content, TextContent):
            text_content = content.text

        elif isinstance(content, TableContent):
            text_content = content.text

        elif isinstance(content, ImageContent):
            asset_path = content.path

        elif isinstance(content, AudioContent):
            asset_path = content.path

        elif isinstance(content, VideoContent):
            asset_path = content.path

        else:
            raise ValueError(
                "Unsupported source element content type: "
                f"{type(content).__name__}"
            )

        content_type = element.metadata.get(
            "content_type"
        )

        if not content_type:
            content_type = (
                type(content).__name__
                .removesuffix("Content")
                .lower()
            )

        return SourceElement(
            element_id=element.element_id,
            document_id=element.document_id,
            element_type=element.element_type,
            content_type=content_type,
            text_content=text_content,
            asset_path=asset_path,
            metadata_json=dict(element.metadata),
        )