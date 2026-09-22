from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    source_type: str
    status: str
    error: str | None = None
    size_bytes: int | None = None
    created_at: datetime
    updated_at: datetime


class ChatDocumentListResponse(BaseModel):
    chat_id: str
    documents: list[DocumentResponse] = Field(
        default_factory=list
    )


class DocumentRetryResponse(BaseModel):
    document_id: str
    status: str