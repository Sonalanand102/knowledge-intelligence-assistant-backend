from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ChatMessageResponse(BaseModel):
    message_id: UUID
    chat_id: UUID
    role: str
    content: str
    citations: list[dict[str, Any]] = Field(
        default_factory=list
    )
    created_at: datetime


class ChatMessageListResponse(BaseModel):
    messages: list[ChatMessageResponse] = Field(
        default_factory=list
    )