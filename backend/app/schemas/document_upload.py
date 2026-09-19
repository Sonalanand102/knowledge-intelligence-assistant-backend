from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class UploadedDocumentResponse(BaseModel):
    document_id: str
    filename: str
    source_type: str
    status: str


class MultiDocumentUploadResponse(BaseModel):
    chat_id: str

    documents: list[
        UploadedDocumentResponse
    ] = Field(default_factory=list)