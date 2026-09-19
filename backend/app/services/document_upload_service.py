from __future__ import annotations

import uuid
from typing import Any

from fastapi import UploadFile
from sqlalchemy import select

from backend.app.db.models.chat import Chat
from backend.app.db.models.chat_document import ChatDocument
from backend.app.db.models.source_document import SourceDocument
from backend.app.db.session import AsyncSessionLocal
from backend.app.ingestion.loader_registry import (
    LoaderRegistry,
)
from backend.app.schemas.document_upload import (
    MultiDocumentUploadResponse,
    UploadedDocumentResponse,
)
from backend.app.services.file_storage import (
    LocalFileStorage,
)

import asyncio

from backend.app.queues.document_queue import (
    DocumentJobQueue,
)

class ChatNotFoundError(ValueError):
    pass


class DocumentUploadService:
    """
    Handles the synchronous part of document upload.

    Responsibilities:
        1. Validate chat
        2. Validate file types
        3. Persist uploaded files
        4. Create SourceDocument records
        5. Create ChatDocument associations

    Heavy ingestion is intentionally not executed here.
    """

    def __init__(
        self,
        storage: LocalFileStorage,
        loader_registry: LoaderRegistry,
        job_queue: DocumentJobQueue
    ) -> None:
        self.storage = storage
        self.loader_registry = loader_registry
        self.job_queue = job_queue

    async def upload_documents(
        self,
        chat_id: str,
        files: list[UploadFile],
    ) -> MultiDocumentUploadResponse:
        if not chat_id or not chat_id.strip():
            raise ValueError(
                "chat_id cannot be empty"
            )

        if not files:
            raise ValueError(
                "At least one file is required"
            )

        normalized_chat_id = chat_id.strip()

        # --------------------------------------------------
        # Validate all filenames BEFORE writing anything.
        # --------------------------------------------------

        detected_source_types: list[str] = []

        for file in files:
            if not file.filename:
                raise ValueError(
                    "Every uploaded file must have a filename"
                )

            source_type = (
                self.loader_registry.detect_source_type(
                    file.filename
                )
            )

            detected_source_types.append(
                source_type
            )

        stored_files = []
        created_document_ids: list[str] = []

        try:
            async with AsyncSessionLocal() as session:
                # ------------------------------------------
                # 1. Verify chat exists
                # ------------------------------------------

                chat_result = await session.execute(
                    select(Chat).where(
                        Chat.id
                        == uuid.UUID(normalized_chat_id)
                    )
                )

                chat = chat_result.scalar_one_or_none()

                if chat is None:
                    raise ChatNotFoundError(
                        f"Chat not found: {normalized_chat_id}"
                    )

                # ------------------------------------------
                # 2. Store files + DB records
                # ------------------------------------------

                responses: list[
                    UploadedDocumentResponse
                ] = []

                for file, source_type in zip(
                    files,
                    detected_source_types,
                ):
                    document_id = str(
                        uuid.uuid4()
                    )

                    stored_file = (
                        await self.storage.save(
                            file,
                            document_id=document_id,
                        )
                    )

                    stored_files.append(
                        stored_file
                    )

                    source_document = SourceDocument(
                        document_id=document_id,
                        source_type=source_type,
                        title=stored_file.original_filename,
                        metadata_json={
                            "original_filename": (
                                stored_file.original_filename
                            ),
                            "storage_path": (
                                stored_file.storage_path
                            ),
                            "content_type": (
                                file.content_type
                            ),
                            "size_bytes": (
                                stored_file.size_bytes
                            ),
                            "processing_status": (
                                "pending"
                            ),
                        },
                    )

                    chat_document = ChatDocument(
                        chat_id=chat.id,
                        document_id=document_id,
                    )

                    session.add(source_document)

                    await session.flush()  # Ensure source_document is persisted before creating ChatDocument

                    session.add(chat_document)

                    created_document_ids.append(
                        document_id
                    )

                    responses.append(
                        UploadedDocumentResponse(
                            document_id=document_id,
                            filename=(
                                stored_file.original_filename
                            ),
                            source_type=source_type,
                            status="pending",
                        )
                    )

                await session.commit()

                for document_id in created_document_ids:
                    await asyncio.to_thread(
                        self.job_queue.enqueue,
                        document_id,
                    )

                return MultiDocumentUploadResponse(
                    chat_id=normalized_chat_id,
                    documents=responses,
                )


        except Exception:
            # ------------------------------------------
            # Cleanup files if DB transaction fails.
            # ------------------------------------------

            for stored_file in stored_files:
                try:
                    from pathlib import Path

                    Path(
                        stored_file.storage_path
                    ).unlink(
                        missing_ok=True
                    )
                except OSError:
                    pass

            raise