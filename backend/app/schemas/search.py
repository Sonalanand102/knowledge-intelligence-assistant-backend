from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchResultResponse(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    score: float
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultResponse]