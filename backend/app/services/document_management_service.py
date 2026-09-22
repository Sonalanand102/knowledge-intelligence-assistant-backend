from __future__ import annotations

import asyncio
import uuid

from sqlalchemy import delete, select

from backend.app.db.models.chat import Chat
from backend.app.db.models.chat_document import ChatDocument
from backend.app.db.models.source_document import SourceDocument
from backend.app.db.session import AsyncSessionLocal
from backend.app.queues.document_queue import DocumentJobQueue
from backend.app.schemas.document_management import (
    ChatDocumentListResponse,
    DocumentResponse,
    DocumentRetryResponse,
)


class ChatNotFoundError(ValueError):
    pass


class DocumentNotFoundError(ValueError):
    pass


class DocumentNotRetryableError(ValueError):
    pass


class DocumentManagementService:
    """
    Handles document listing, inspection, chat detachment,
    and failed-document retry.
    """

    def __init__(
        self,
        job_queue: DocumentJobQueue,
    ) -> None:
        self.job_queue = job_queue

    async def list_chat_documents(
        self,
        chat_id: str,
    ) -> ChatDocumentListResponse:
        parsed_chat_id = self._parse_chat_id(
            chat_id
        )

        async with AsyncSessionLocal() as session:
            chat = await session.scalar(
                select(Chat).where(
                    Chat.id == parsed_chat_id
                )
            )

            if chat is None:
                raise ChatNotFoundError(
                    f"Chat not found: {chat_id}"
                )

            result = await session.execute(
                select(
                    SourceDocument,
                    ChatDocument.created_at,
                )
                .join(
                    ChatDocument,
                    ChatDocument.document_id
                    == SourceDocument.document_id,
                )
                .where(
                    ChatDocument.chat_id
                    == parsed_chat_id
                )
                .order_by(
                    ChatDocument.created_at.asc()
                )
            )

            rows = result.all()

            documents = [
                self._to_response(
                    document
                )
                for document, _ in rows
            ]

            return ChatDocumentListResponse(
                chat_id=str(parsed_chat_id),
                documents=documents,
            )

    async def get_document(
        self,
        document_id: str,
    ) -> DocumentResponse:
        if not document_id or not document_id.strip():
            raise DocumentNotFoundError(
                "Document ID cannot be empty"
            )

        normalized_document_id = (
            document_id.strip()
        )

        async with AsyncSessionLocal() as session:
            document = await session.scalar(
                select(SourceDocument).where(
                    SourceDocument.document_id
                    == normalized_document_id
                )
            )

            if document is None:
                raise DocumentNotFoundError(
                    f"Document not found: "
                    f"{normalized_document_id}"
                )

            return self._to_response(
                document
            )

    async def detach_document(
        self,
        chat_id: str,
        document_id: str,
    ) -> None:
        parsed_chat_id = self._parse_chat_id(
            chat_id
        )

        if not document_id or not document_id.strip():
            raise DocumentNotFoundError(
                "Document ID cannot be empty"
            )

        normalized_document_id = (
            document_id.strip()
        )

        async with AsyncSessionLocal() as session:
            chat = await session.scalar(
                select(Chat).where(
                    Chat.id == parsed_chat_id
                )
            )

            if chat is None:
                raise ChatNotFoundError(
                    f"Chat not found: {chat_id}"
                )

            chat_document = await session.scalar(
                select(ChatDocument).where(
                    ChatDocument.chat_id
                    == parsed_chat_id,
                    ChatDocument.document_id
                    == normalized_document_id,
                )
            )

            if chat_document is None:
                raise DocumentNotFoundError(
                    "Document is not attached to "
                    f"chat: {normalized_document_id}"
                )

            await session.delete(
                chat_document
            )

            await session.commit()

    async def retry_document(
        self,
        document_id: str,
    ) -> DocumentRetryResponse:
        if not document_id or not document_id.strip():
            raise DocumentNotFoundError(
                "Document ID cannot be empty"
            )

        normalized_document_id = (
            document_id.strip()
        )

        async with AsyncSessionLocal() as session:
            document = await session.scalar(
                select(SourceDocument).where(
                    SourceDocument.document_id
                    == normalized_document_id
                )
            )

            if document is None:
                raise DocumentNotFoundError(
                    f"Document not found: "
                    f"{normalized_document_id}"
                )

            metadata = dict(
                document.metadata_json or {}
            )

            current_status = metadata.get(
                "processing_status"
            )

            if current_status not in {
                "failed",
                "pending",
            }:
                raise DocumentNotRetryableError(
                    "Only failed or pending documents "
                    "can be retried. Current status: "
                    f"{current_status}"
                )

            metadata[
                "processing_status"
            ] = "pending"

            metadata.pop(
                "processing_error",
                None,
            )

            document.metadata_json = metadata

            await session.commit()

        try:
            await asyncio.to_thread(
                self.job_queue.enqueue,
                normalized_document_id,
                retry=True,
            )

        except Exception as exc:
            await self._mark_retry_enqueue_failed(
                normalized_document_id,
                str(exc),
            )

            raise

        return DocumentRetryResponse(
            document_id=normalized_document_id,
            status="pending",
        )

    async def _mark_retry_enqueue_failed(
        self,
        document_id: str,
        error_message: str,
    ) -> None:
        async with AsyncSessionLocal() as session:
            document = await session.scalar(
                select(SourceDocument).where(
                    SourceDocument.document_id
                    == document_id
                )
            )

            if document is None:
                return

            metadata = dict(
                document.metadata_json or {}
            )

            metadata[
                "processing_status"
            ] = "failed"

            metadata[
                "processing_error"
            ] = (
                "Failed to enqueue retry job: "
                f"{error_message}"
            )

            document.metadata_json = metadata

            await session.commit()

    @staticmethod
    def _parse_chat_id(
        chat_id: str,
    ) -> uuid.UUID:
        try:
            return uuid.UUID(chat_id)
        except (
            ValueError,
            AttributeError,
        ) as exc:
            raise ChatNotFoundError(
                f"Chat not found: {chat_id}"
            ) from exc

    @staticmethod
    def _to_response(
        document: SourceDocument,
    ) -> DocumentResponse:
        metadata = dict(
            document.metadata_json or {}
        )

        return DocumentResponse(
            document_id=document.document_id,
            filename=(
                metadata.get(
                    "original_filename"
                )
                or document.title
                or document.document_id
            ),
            source_type=document.source_type,
            status=metadata.get(
                "processing_status",
                "unknown",
            ),
            error=metadata.get(
                "processing_error"
            ),
            size_bytes=metadata.get(
                "size_bytes"
            ),
            created_at=document.created_at,
            updated_at=document.updated_at,
        )