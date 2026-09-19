from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="Natural-language question",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of retrieved chunks used for generation",
    )


class CitationResponse(BaseModel):
    citation_id: int
    chunk_id: str
    document_id: str
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: list[CitationResponse] = Field(
        default_factory=list
    )


class ChatAskResponse(BaseModel):
    chat_id: UUID
    query: str
    answer: str
    citations: list[CitationResponse] = Field(
        default_factory=list
    )